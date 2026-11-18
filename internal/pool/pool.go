package pool

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"acp2api/internal/acp"
	"acp2api/internal/bridge"
	"acp2api/internal/config"
)

// Pool manages ACP stdio worker processes, a background pre-warmed idle session pool,
// and an LRU multi-turn conversation cache.
type Pool struct {
	cfg       config.Config
	logger    *slog.Logger
	poolCtx   context.Context
	poolStop  context.CancelFunc
	activeSem chan struct{}

	mu               sync.Mutex
	workers          []*acpWorker
	idleSessions     []*ManagedSession
	cachedSessions   []*ManagedSession          // ordered oldest -> newest (LRU at index 0)
	inFlightByHash   map[string]*ManagedSession // currently checked-out sessions eligible for native session/fork
	discoveredModels map[string]string          // modelId -> name
	warmingCount     int
	closed           bool
	workerSeq        int

	replenishCh chan struct{}
	wg          sync.WaitGroup

	// Metrics
	activeTurns      atomic.Int64
	totalRequests    atomic.Int64
	idlePoolHits     atomic.Int64
	conversationHits atomic.Int64
	sessionForkHits  atomic.Int64
	coldCreations    atomic.Int64
	workerRestarts   atomic.Int64
}

func New(cfg config.Config, logger *slog.Logger) *Pool {
	if logger == nil {
		logger = slog.Default()
	}
	ctx, cancel := context.WithCancel(context.Background())
	var sem chan struct{}
	if cfg.Pool.MaxActiveSessions > 0 {
		sem = make(chan struct{}, cfg.Pool.MaxActiveSessions)
	}
	return &Pool{
		cfg:              cfg,
		logger:           logger,
		poolCtx:          ctx,
		poolStop:         cancel,
		activeSem:        sem,
		inFlightByHash:   make(map[string]*ManagedSession),
		discoveredModels: make(map[string]string),
		replenishCh:      make(chan struct{}, 1),
	}
}

// Start launches ACP worker processes and pre-warms the initial idle session pool.
func (p *Pool) Start(ctx context.Context) error {
	count := p.cfg.Pool.ProcessCount
	if count <= 0 {
		count = 1
	}

	type startRes struct {
		slot int
		w    *acpWorker
		err  error
	}
	resCh := make(chan startRes, count)
	for slot := 0; slot < count; slot++ {
		s := slot
		go func() {
			w, err := p.spawnWorker(ctx, s)
			resCh <- startRes{slot: s, w: w, err: err}
		}()
	}

	p.mu.Lock()
	p.workers = make([]*acpWorker, count)
	p.mu.Unlock()

	var firstErr error
	started := 0
	for i := 0; i < count; i++ {
		r := <-resCh
		if r.err != nil {
			if firstErr == nil {
				firstErr = r.err
			}
			p.logger.Error("failed starting initial ACP worker", "slot", r.slot, "error", r.err)
			continue
		}
		started++
		p.mu.Lock()
		p.workers[r.slot] = r.w
		p.mu.Unlock()
		p.wg.Add(1)
		go p.watchWorker(r.w)
	}

	if started == 0 {
		p.poolStop()
		return fmt.Errorf("failed to start any ACP stdio worker: %w", firstErr)
	}

	targetIdle := p.cfg.Pool.IdlePoolSize
	if targetIdle > 0 {
		warmCtx, cancel := context.WithTimeout(ctx, p.cfg.SessionNewTimeout())
		var warmWG sync.WaitGroup
		for i := 0; i < targetIdle; i++ {
			warmWG.Add(1)
			p.mu.Lock()
			p.warmingCount++
			p.mu.Unlock()
			go func() {
				defer warmWG.Done()
				sess, err := p.createFreshSession(warmCtx)
				p.mu.Lock()
				p.warmingCount--
				if err == nil && !p.closed {
					p.idleSessions = append(p.idleSessions, sess)
				}
				p.mu.Unlock()
				if err != nil {
					p.logger.Warn("initial idle session warmup failed", "error", err)
				} else if p.closed {
					p.retireSession(sess)
				}
			}()
		}
		warmWG.Wait()
		cancel()
	}

	p.logger.Info("ACP session pool ready",
		"workers", started,
		"idle_sessions_ready", p.Stats().ReadyIdleSessions,
		"target_idle_sessions", targetIdle,
		"reuse_conversations", p.cfg.Pool.ReuseConversations,
	)

	p.wg.Add(2)
	go p.replenishLoop()
	go p.janitorLoop()
	p.signalReplenish()
	return nil
}

// Close gracefully shuts down background loops and all ACP worker subprocesses.
func (p *Pool) Close() error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	workers := append([]*acpWorker(nil), p.workers...)
	p.idleSessions = nil
	p.cachedSessions = nil
	p.inFlightByHash = make(map[string]*ManagedSession)
	p.mu.Unlock()

	p.poolStop()
	for _, w := range workers {
		if w != nil && w.client != nil {
			_ = w.client.Close()
		}
	}
	p.wg.Wait()
	return nil
}

// Stats returns a snapshot of current pool health and session metrics.
func (p *Pool) Stats() Stats {
	p.mu.Lock()
	defer p.mu.Unlock()

	healthy := 0
	totalW := 0
	for _, w := range p.workers {
		if w != nil {
			totalW++
			if w.client.Alive() && !w.draining.Load() {
				healthy++
			}
		}
	}

	return Stats{
		TotalWorkers:        totalW,
		HealthyWorkers:      healthy,
		TargetIdleSessions:  p.cfg.Pool.IdlePoolSize,
		ReadyIdleSessions:   len(p.idleSessions),
		WarmingSessions:     p.warmingCount,
		CachedConversations: len(p.cachedSessions),
		ActiveTurns:         p.activeTurns.Load(),
		TotalRequests:       p.totalRequests.Load(),
		IdlePoolHits:        p.idlePoolHits.Load(),
		ConversationHits:    p.conversationHits.Load(),
		SessionForkHits:     p.sessionForkHits.Load(),
		ColdCreations:       p.coldCreations.Load(),
		WorkerRestarts:      p.workerRestarts.Load(),
	}
}

// AvailableModels returns the deduplicated list of models exposed by this gateway.
func (p *Pool) AvailableModels() []string {
	seen := make(map[string]struct{})
	var out []string
	add := func(id string) {
		id = strings.TrimSpace(id)
		if id == "" {
			return
		}
		if _, ok := seen[id]; !ok {
			seen[id] = struct{}{}
			out = append(out, id)
		}
	}

	add(p.cfg.ACP.DefaultModel)
	for alias := range p.cfg.ACP.ModelAliases {
		add(alias)
	}

	p.mu.Lock()
	discovered := make([]string, 0, len(p.discoveredModels))
	for id := range p.discoveredModels {
		discovered = append(discovered, id)
	}
	p.mu.Unlock()

	sort.Strings(discovered)
	for _, id := range discovered {
		add(id)
	}
	return out
}

// ExecuteTurn runs a single chat/messages API request against a reused, forked, resumed, or pre-warmed ACP session.
func (p *Pool) ExecuteTurn(
	ctx context.Context,
	req *bridge.UnifiedRequest,
	onEvent func(bridge.StreamEvent),
) (*bridge.TurnResponse, error) {
	p.totalRequests.Add(1)

	if p.activeSem != nil {
		select {
		case p.activeSem <- struct{}{}:
			defer func() { <-p.activeSem }()
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	cleanModel, suffixEffort := p.cfg.SplitModelThinkingSuffix(req.Model)
	targetACPModel, displayModel := p.cfg.ResolveModel(cleanModel)
	if strings.TrimSpace(req.Model) != "" {
		displayModel = strings.TrimSpace(req.Model)
	}

	effectiveThinking := req.Thinking
	if !effectiveThinking.Specified && suffixEffort != "" {
		effectiveThinking = acp.NormalizeThinkingIntent(nil, suffixEffort, 0)
	}

	activeTools := req.Tools
	if req.ToolChoice.Mode == "none" {
		activeTools = nil
	}

	sess, startIdx, reused, err := p.acquireSession(ctx, req, targetACPModel, activeTools)
	if err != nil {
		return nil, err
	}

	priorHash := sess.HistoryHash
	if priorHash != "" {
		p.mu.Lock()
		p.inFlightByHash[priorHash] = sess
		p.mu.Unlock()
		defer func() {
			p.mu.Lock()
			if p.inFlightByHash[priorHash] == sess {
				delete(p.inFlightByHash, priorHash)
			}
			p.mu.Unlock()
		}()
	}

	sess.Worker.activeTurns.Add(1)
	p.activeTurns.Add(1)
	defer func() {
		sess.Worker.activeTurns.Add(-1)
		p.activeTurns.Add(-1)
	}()

	// 1. Switch ACP session model if requested and different from current session state.
	if targetACPModel != "" && targetACPModel != sess.CurrentModel {
		updatedOpts, err := sess.Worker.client.SetSessionModel(ctx, sess.ID, targetACPModel, sess.Meta)
		if err != nil {
			var rpcErr *acp.RPCError
			if errors.As(err, &rpcErr) && rpcErr.Code == -32601 {
				p.logger.Debug("ACP agent does not implement model switching; using agent default", "requested_model", targetACPModel)
			} else {
				p.logger.Warn("ACP SetSessionModel returned warning", "session_id", sess.ID, "model", targetACPModel, "error", err)
			}
		} else {
			sess.CurrentModel = targetACPModel
			if len(updatedOpts) > 0 {
				sess.Meta.ConfigOptions = updatedOpts
				p.recordDiscoveredModels(sess.Meta)
			}
		}
	}

	// 2. Switch operational mode if explicitly requested (e.g. via X-ACP-Mode header).
	if desiredMode := strings.TrimSpace(req.Mode); desiredMode != "" && desiredMode != sess.CurrentMode {
		updatedOpts, err := sess.Worker.client.SetSessionMode(ctx, sess.ID, desiredMode, sess.Meta)
		if err != nil {
			p.logger.Warn("ACP SetSessionMode returned warning", "session_id", sess.ID, "mode", desiredMode, "error", err)
		} else {
			sess.CurrentMode = desiredMode
			if len(updatedOpts) > 0 {
				sess.Meta.ConfigOptions = updatedOpts
			}
		}
	}

	// 3. Resolve & apply Thinking / Reasoning Effort across standard and non-standard ACP servers.
	thinkingRes, err := acp.ResolveThinkingAction(sess.Meta, effectiveThinking, targetACPModel, p.cfg.ACP.Thinking)
	if err != nil {
		// StrictError mode: return session back to cache/idle if untouched or retire it, then surface error.
		p.retireSession(sess)
		return nil, err
	}

	thinkingStrategy := thinkingRes.Strategy
	if thinkingRes.ConfigID != "" {
		desiredSig := fmt.Sprintf("%s=%v", thinkingRes.ConfigID, thinkingRes.ConfigValue)
		if desiredSig != sess.CurrentThinking {
			updatedOpts, cfgErr := sess.Worker.client.SetConfigOption(ctx, sess.ID, thinkingRes.ConfigID, thinkingRes.ConfigValue)
			if cfgErr != nil {
				// Graceful degradation when optimistic override_config_id is rejected by the ACP agent.
				p.logger.Debug("ACP SetConfigOption for thinking level not accepted; degrading gracefully",
					"session_id", sess.ID,
					"config_id", thinkingRes.ConfigID,
					"value", thinkingRes.ConfigValue,
					"error", cfgErr,
				)
				thinkingStrategy = thinkingStrategy + "_degraded"
			} else {
				sess.CurrentThinking = desiredSig
				if len(updatedOpts) > 0 {
					sess.Meta.ConfigOptions = updatedOpts
				}
			}
		}
	}
	if thinkingRes.ModelOverride != "" && thinkingRes.ModelOverride != sess.CurrentModel {
		if updatedOpts, mErr := sess.Worker.client.SetSessionModel(ctx, sess.ID, thinkingRes.ModelOverride, sess.Meta); mErr == nil {
			sess.CurrentModel = thinkingRes.ModelOverride
			if len(updatedOpts) > 0 {
				sess.Meta.ConfigOptions = updatedOpts
			}
		}
	}
	if thinkingRes.ModeOverride != "" && thinkingRes.ModeOverride != sess.CurrentMode {
		if updatedOpts, modeErr := sess.Worker.client.SetSessionMode(ctx, sess.ID, thinkingRes.ModeOverride, sess.Meta); modeErr == nil {
			sess.CurrentMode = thinkingRes.ModeOverride
			if len(updatedOpts) > 0 {
				sess.Meta.ConfigOptions = updatedOpts
			}
		}
	}

	promptCaps := sess.Worker.client.InitResult().AgentCapabilities.PromptCapabilities
	// Note: many ACP servers support text+image without setting promptCapabilities.image=true; only downgrade image
	// if explicitly false AND URI fallback exists, or keep true when Data is present unless strict.
	// Here we pass actual negotiated Audio & EmbeddedContext capabilities while allowing base64 Image when advertised.
	blocks := bridge.BuildPromptBlocksWithCaps(req, startIdx, promptCaps, thinkingRes.PromptHint)

	var (
		thinkingBuf   strings.Builder
		textBuf       strings.Builder
		toolCalls     []bridge.ToolCallValue
		agentUsage    *acp.UsageInfo
		stoppedSeq    string
		stopTriggered bool
	)

	exposeThoughts := p.cfg.ACP.ExposeThoughts
	if effectiveThinking.Specified && !effectiveThinking.Enabled {
		exposeThoughts = false
	}

	turnCtx, cancelTurn := context.WithCancel(ctx)
	defer cancelTurn()

	emitEvent := func(ev bridge.StreamEvent) {
		switch ev.Kind {
		case bridge.EventThinkingDelta:
			thinkingBuf.WriteString(ev.Text)
			if exposeThoughts && onEvent != nil {
				onEvent(ev)
			}
		case bridge.EventTextDelta:
			if stopTriggered {
				return
			}
			chunk := ev.Text
			if len(req.StopSequences) > 0 {
				combined := textBuf.String() + chunk
				if cutText, matched, ok := checkStopSequences(combined, textBuf.Len(), req.StopSequences); ok {
					delta := cutText[textBuf.Len():]
					textBuf.WriteString(delta)
					stoppedSeq = matched
					stopTriggered = true
					if delta != "" && onEvent != nil {
						onEvent(bridge.StreamEvent{Kind: bridge.EventTextDelta, Text: delta})
					}
					cancelTurn()
					return
				}
			}
			textBuf.WriteString(chunk)
			if onEvent != nil {
				onEvent(ev)
			}
		case bridge.EventToolCall:
			if stopTriggered || ev.ToolCall == nil {
				return
			}
			toolCalls = append(toolCalls, *ev.ToolCall)
			if onEvent != nil {
				onEvent(ev)
			}
		}
	}

	toolsEnabled := len(activeTools) > 0
	toolParser := bridge.NewToolStreamParser(toolsEnabled, emitEvent)
	thinkParser := bridge.NewInlineThinkParser(
		p.cfg.ACP.Thinking.ExtractInlineThinkTags,
		func(thoughtChunk string) {
			emitEvent(bridge.StreamEvent{Kind: bridge.EventThinkingDelta, Text: thoughtChunk})
		},
		func(textChunk string) {
			toolParser.Feed(textChunk)
		},
	)

	var metaMu sync.Mutex
	promptRes, _, promptErr := sess.Worker.client.Prompt(turnCtx, sess.ID, blocks, thinkingRes.MetaHint, func(u acp.ParsedUpdate) {
		switch u.Kind {
		case "agent_thought_chunk":
			thinkParser.MarkNativeThoughtSeen()
			if u.Text != "" {
				emitEvent(bridge.StreamEvent{Kind: bridge.EventThinkingDelta, Text: u.Text})
			}
		case "agent_message_chunk":
			if u.Text != "" {
				thinkParser.Feed(u.Text)
			}
		case "config_option_update":
			if len(u.ConfigOptions) > 0 {
				metaMu.Lock()
				sess.Meta.ConfigOptions = u.ConfigOptions
				metaCopy := sess.Meta
				metaMu.Unlock()
				p.recordDiscoveredModels(metaCopy)
			}
		case "current_mode_update":
			if u.CurrentModeID != "" {
				metaMu.Lock()
				sess.CurrentMode = u.CurrentModeID
				metaMu.Unlock()
			}
		case "usage_update":
			if u.Usage != nil {
				metaMu.Lock()
				agentUsage = u.Usage
				metaMu.Unlock()
			}
		}
	})

	thinkParser.Finish()
	toolParser.Finish()

	if promptErr != nil && stopTriggered && ctx.Err() == nil {
		promptErr = nil
	}

	if promptErr != nil {
		p.retireSession(sess)
		return nil, promptErr
	}

	if promptRes.Usage != nil {
		agentUsage = promptRes.Usage
	}

	stopReason := normalizeStopReason(promptRes.StopReason, len(toolCalls) > 0, stoppedSeq != "")
	inTokens, outTokens := computeTokenUsage(req, thinkingBuf.String(), textBuf.String(), toolCalls, agentUsage)

	exposedThinking := ""
	if exposeThoughts {
		exposedThinking = thinkingBuf.String()
	}

	resp := &bridge.TurnResponse{
		Model:            displayModel,
		SessionID:        sess.ID,
		SessionReused:    reused,
		ThinkingStrategy: thinkingStrategy,
		Thinking:         exposedThinking,
		Text:             textBuf.String(),
		ToolCalls:        toolCalls,
		StopReason:       stopReason,
		StopSequence:     stoppedSeq,
		InputTokens:      inTokens,
		OutputTokens:     outTokens,
	}

	if !stopTriggered && p.cfg.Pool.ReuseConversations && p.cfg.Pool.MaxCachedSessions > 0 && sess.Worker.client.Alive() && !sess.Worker.draining.Load() {
		updatedMsgs := make([]bridge.UnifiedMessage, 0, len(req.Messages)+1)
		updatedMsgs = append(updatedMsgs, req.Messages...)
		updatedMsgs = append(updatedMsgs, resp.AssistantMessage())

		sess.HistoryHash = bridge.ComputeConversationHash(targetACPModel, req.System, activeTools, updatedMsgs)
		sess.SessionHint = req.SessionHint
		sess.TurnCount++
		sess.LastUsedAt = time.Now()
		p.storeCachedSession(sess)
	} else {
		p.retireSession(sess)
	}

	return resp, nil
}

func (p *Pool) acquireSession(
	ctx context.Context,
	req *bridge.UnifiedRequest,
	targetACPModel string,
	activeTools []bridge.ToolDefinition,
) (*ManagedSession, int, bool, error) {
	// 0. Explicit session resume/load request via X-ACP-Resume-Session-ID header.
	if resumeID := strings.TrimSpace(req.ResumeSessionID); resumeID != "" {
		if sess, err := p.resumeOrLoadSession(ctx, resumeID); err == nil {
			startIdx := len(req.Messages) - 1
			if startIdx < 0 {
				startIdx = 0
			}
			return sess, startIdx, true, nil
		}
	}

	// 1. Try smart multi-turn prefix match against cached conversation sessions (or native session/fork on concurrent branches!).
	if p.cfg.Pool.ReuseConversations && len(req.Messages) > 1 {
		for splitIdx := len(req.Messages) - 1; splitIdx >= 1; splitIdx-- {
			if req.Messages[splitIdx-1].Role != "assistant" {
				continue
			}
			prefixHash := bridge.ComputeConversationHash(targetACPModel, req.System, activeTools, req.Messages[:splitIdx])
			if sess := p.checkoutCachedByHash(prefixHash, req.SessionHint); sess != nil {
				p.conversationHits.Add(1)
				p.logger.Debug("reusing cached ACP session for multi-turn request",
					"session_id", sess.ID,
					"split_idx", splitIdx,
					"delta_messages", len(req.Messages)-splitIdx,
				)
				return sess, splitIdx, true, nil
			}
			// If an identical prefix session is currently in-flight on a fork-capable worker, fork it natively!
			if p.cfg.Pool.EnableNativeSessionFork {
				if forked := p.tryForkInFlightSession(ctx, prefixHash); forked != nil {
					p.sessionForkHits.Add(1)
					p.logger.Debug("forked in-flight ACP session via session/fork",
						"new_session_id", forked.ID,
						"split_idx", splitIdx,
					)
					return forked, splitIdx, true, nil
				}
			}
		}
	}

	// 2. Pop a pre-warmed idle session for instant response, and trigger async replenishment.
	if sess := p.popIdleSession(); sess != nil {
		p.idlePoolHits.Add(1)
		p.signalReplenish()
		p.logger.Debug("acquired pre-warmed idle ACP session", "session_id", sess.ID)
		return sess, 0, false, nil
	}

	// 3. Pool momentarily empty under burst load: create session on demand and trigger background replenishment.
	p.signalReplenish()
	p.coldCreations.Add(1)
	sess, err := p.createFreshSession(ctx)
	if err != nil {
		return nil, 0, false, err
	}
	p.logger.Debug("created cold ACP session under burst load", "session_id", sess.ID)
	return sess, 0, false, nil
}

func (p *Pool) resumeOrLoadSession(ctx context.Context, sessionID string) (*ManagedSession, error) {
	w, err := p.pickHealthyWorker()
	if err != nil {
		return nil, err
	}
	caps := w.client.InitResult().AgentCapabilities
	var meta acp.SessionNewResult
	switch {
	case caps.SessionCapabilities.HasResume():
		meta, err = w.client.ResumeSession(ctx, sessionID)
	case caps.LoadSession:
		meta, err = w.client.LoadSession(ctx, sessionID, nil)
	default:
		return nil, errors.New("agent supports neither session/resume nor session/load")
	}
	if err != nil {
		return nil, err
	}
	w.liveSessions.Add(1)
	w.totalSessions.Add(1)
	now := time.Now()
	return &ManagedSession{
		ID:         meta.SessionID,
		Worker:     w,
		Meta:       meta,
		CreatedAt:  now,
		LastUsedAt: now,
	}, nil
}

func (p *Pool) tryForkInFlightSession(ctx context.Context, prefixHash string) *ManagedSession {
	p.mu.Lock()
	src := p.inFlightByHash[prefixHash]
	p.mu.Unlock()

	if src == nil || src.Worker == nil || !src.Worker.client.Alive() || src.Worker.draining.Load() || !src.Worker.client.SupportsFork() {
		return nil
	}
	forkCtx, cancel := context.WithTimeout(ctx, p.cfg.SessionNewTimeout())
	defer cancel()
	meta, err := src.Worker.client.ForkSession(forkCtx, src.ID)
	if err != nil {
		return nil
	}
	src.Worker.liveSessions.Add(1)
	src.Worker.totalSessions.Add(1)
	now := time.Now()
	return &ManagedSession{
		ID:              meta.SessionID,
		Worker:          src.Worker,
		Meta:            meta,
		CurrentModel:    src.CurrentModel,
		CurrentMode:     src.CurrentMode,
		CurrentThinking: src.CurrentThinking,
		CreatedAt:       now,
		LastUsedAt:      now,
	}
}

func (p *Pool) checkoutCachedByHash(hash, sessionHint string) *ManagedSession {
	now := time.Now()
	ttl := p.cfg.CachedSessionTTL()

	p.mu.Lock()
	defer p.mu.Unlock()

	bestIdx := -1
	for i := len(p.cachedSessions) - 1; i >= 0; i-- {
		s := p.cachedSessions[i]
		if !s.Worker.client.Alive() || s.Worker.draining.Load() || now.Sub(s.LastUsedAt) > ttl {
			continue
		}
		if s.HistoryHash != hash {
			continue
		}
		if sessionHint != "" && s.SessionHint == sessionHint {
			bestIdx = i
			break
		}
		if bestIdx == -1 {
			bestIdx = i
		}
	}

	if bestIdx < 0 {
		return nil
	}
	chosen := p.cachedSessions[bestIdx]
	p.cachedSessions = append(p.cachedSessions[:bestIdx], p.cachedSessions[bestIdx+1:]...)
	return chosen
}

func (p *Pool) storeCachedSession(sess *ManagedSession) {
	maxCached := p.cfg.Pool.MaxCachedSessions
	var evicted []*ManagedSession

	p.mu.Lock()
	if p.closed || maxCached <= 0 || !sess.Worker.client.Alive() || sess.Worker.draining.Load() {
		p.mu.Unlock()
		p.retireSession(sess)
		return
	}
	p.cachedSessions = append(p.cachedSessions, sess)
	for len(p.cachedSessions) > maxCached {
		evicted = append(evicted, p.cachedSessions[0])
		p.cachedSessions = p.cachedSessions[1:]
	}
	p.mu.Unlock()

	for _, ev := range evicted {
		p.retireSession(ev)
	}
}

func (p *Pool) popIdleSession() *ManagedSession {
	now := time.Now()
	ttl := p.cfg.IdleSessionTTL()

	p.mu.Lock()
	var stale []*ManagedSession
	var chosen *ManagedSession
	for len(p.idleSessions) > 0 {
		s := p.idleSessions[0]
		p.idleSessions = p.idleSessions[1:]
		if !s.Worker.client.Alive() || s.Worker.draining.Load() || now.Sub(s.CreatedAt) > ttl {
			stale = append(stale, s)
			continue
		}
		chosen = s
		break
	}
	p.mu.Unlock()

	for _, st := range stale {
		p.retireSession(st)
	}
	return chosen
}

func (p *Pool) retireSession(sess *ManagedSession) {
	if sess == nil || sess.Worker == nil {
		return
	}
	w := sess.Worker
	sid := sess.ID

	// Asynchronously release session state on the ACP agent via session/close (if supported by agent).
	if sid != "" && w.client.Alive() {
		go func() {
			closeCtx, cancel := context.WithTimeout(p.poolCtx, 3*time.Second)
			defer cancel()
			_ = w.client.CloseSession(closeCtx, sid)
		}()
	}

	remaining := w.liveSessions.Add(-1)
	if w.draining.Load() && remaining <= 0 && w.activeTurns.Load() <= 0 {
		go func(target *acpWorker) {
			_ = target.client.Close()
		}(w)
	}
	p.maybeScheduleWorkerRotation(w)
}

func (p *Pool) maybeScheduleWorkerRotation(w *acpWorker) {
	maxPerProc := p.cfg.Pool.MaxSessionsPerProcess
	if maxPerProc <= 0 || w == nil || w.draining.Load() {
		return
	}
	if int(w.totalSessions.Load()) < maxPerProc {
		return
	}
	if w.rotating.CompareAndSwap(false, true) {
		p.logger.Info("ACP worker reached max_sessions_per_process; rotating worker",
			"worker", w.id,
			"total_sessions", w.totalSessions.Load(),
		)
		go p.rotateWorker(w)
	}
}

func (p *Pool) rotateWorker(oldW *acpWorker) {
	ctx, cancel := context.WithTimeout(p.poolCtx, p.cfg.InitializeTimeout())
	newW, err := p.spawnWorker(ctx, oldW.slot)
	cancel()
	if err != nil {
		oldW.rotating.Store(false)
		p.logger.Warn("failed spawning replacement worker during rotation", "error", err)
		return
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		_ = newW.client.Close()
		return
	}
	if oldW.slot >= 0 && oldW.slot < len(p.workers) && p.workers[oldW.slot] == oldW {
		p.workers[oldW.slot] = newW
	}
	oldW.draining.Store(true)
	p.mu.Unlock()

	p.wg.Add(1)
	go p.watchWorker(newW)
	p.signalReplenish()

	if oldW.liveSessions.Load() <= 0 && oldW.activeTurns.Load() <= 0 {
		_ = oldW.client.Close()
	}
}

func (p *Pool) pickHealthyWorker() (*acpWorker, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil, errors.New("acp session pool is closed")
	}

	var best *acpWorker
	var bestScore int64
	for _, w := range p.workers {
		if w == nil || !w.client.Alive() || w.draining.Load() {
			continue
		}
		score := w.activeTurns.Load()*1000 + w.liveSessions.Load()
		if best == nil || score < bestScore {
			best = w
			bestScore = score
		}
	}
	if best == nil {
		return nil, errors.New("no healthy ACP worker subprocess is currently available")
	}
	return best, nil
}

func (p *Pool) createFreshSession(ctx context.Context) (*ManagedSession, error) {
	w, err := p.pickHealthyWorker()
	if err != nil {
		return nil, err
	}

	newCtx, cancel := context.WithTimeout(ctx, p.cfg.SessionNewTimeout())
	defer cancel()

	meta, err := w.client.NewSession(newCtx)
	if err != nil {
		return nil, err
	}

	w.liveSessions.Add(1)
	w.totalSessions.Add(1)

	currentModel := ""
	if meta.Models != nil {
		currentModel = meta.Models.CurrentModelID
	}
	currentMode := ""
	if meta.Modes != nil {
		currentMode = meta.Modes.CurrentModeID
	}
	p.recordDiscoveredModels(meta)

	now := time.Now()
	sess := &ManagedSession{
		ID:           meta.SessionID,
		Worker:       w,
		Meta:         meta,
		CurrentModel: currentModel,
		CurrentMode:  currentMode,
		CreatedAt:    now,
		LastUsedAt:   now,
	}
	p.maybeScheduleWorkerRotation(w)
	return sess, nil
}

func (p *Pool) recordDiscoveredModels(meta acp.SessionNewResult) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if meta.Models != nil {
		if meta.Models.CurrentModelID != "" {
			if _, ok := p.discoveredModels[meta.Models.CurrentModelID]; !ok {
				p.discoveredModels[meta.Models.CurrentModelID] = meta.Models.CurrentModelID
			}
		}
		for _, m := range meta.Models.AvailableModels {
			if m.ModelID != "" {
				p.discoveredModels[m.ModelID] = m.Name
			}
		}
	}
	for _, opt := range meta.ConfigOptions {
		if strings.EqualFold(opt.Category, "model") || strings.EqualFold(opt.ID, "model") {
			if cur := opt.EffectiveValue(); cur != "" {
				p.discoveredModels[cur] = cur
			}
			for _, ch := range opt.ExtractChoiceValues() {
				if ch.Value != "" {
					p.discoveredModels[ch.Value] = ch.Name
				}
			}
		}
	}
}

func (p *Pool) spawnWorker(ctx context.Context, slot int) (*acpWorker, error) {
	p.mu.Lock()
	p.workerSeq++
	id := fmt.Sprintf("worker-%d-%d", slot, p.workerSeq)
	p.mu.Unlock()

	client := acp.NewClient(id, p.cfg.ACP, p.logger)
	if err := client.Start(ctx); err != nil {
		return nil, err
	}
	return &acpWorker{
		id:     id,
		slot:   slot,
		client: client,
	}, nil
}

func (p *Pool) watchWorker(w *acpWorker) {
	defer p.wg.Done()
	select {
	case <-w.client.Done():
	case <-p.poolCtx.Done():
		return
	}

	p.purgeWorkerSessions(w)

	p.mu.Lock()
	stillAssigned := !p.closed && w.slot >= 0 && w.slot < len(p.workers) && p.workers[w.slot] == w
	p.mu.Unlock()
	if !stillAssigned {
		return
	}

	backoff := 200 * time.Millisecond
	for {
		if p.poolCtx.Err() != nil {
			return
		}
		p.workerRestarts.Add(1)
		startCtx, cancel := context.WithTimeout(p.poolCtx, p.cfg.InitializeTimeout())
		replacement, err := p.spawnWorker(startCtx, w.slot)
		cancel()
		if err == nil {
			p.mu.Lock()
			if p.closed {
				p.mu.Unlock()
				_ = replacement.client.Close()
				return
			}
			p.workers[w.slot] = replacement
			p.mu.Unlock()

			p.logger.Info("restarted crashed ACP stdio worker", "slot", w.slot, "new_worker", replacement.id)
			p.wg.Add(1)
			go p.watchWorker(replacement)
			p.signalReplenish()
			return
		}

		p.logger.Error("failed restarting ACP worker; retrying", "slot", w.slot, "backoff", backoff, "error", err)
		select {
		case <-time.After(backoff):
		case <-p.poolCtx.Done():
			return
		}
		backoff *= 2
		if backoff > 5*time.Second {
			backoff = 5 * time.Second
		}
	}
}

func (p *Pool) purgeWorkerSessions(dead *acpWorker) {
	p.mu.Lock()
	defer p.mu.Unlock()

	filteredIdle := p.idleSessions[:0]
	for _, s := range p.idleSessions {
		if s.Worker != dead {
			filteredIdle = append(filteredIdle, s)
		}
	}
	p.idleSessions = filteredIdle

	filteredCached := p.cachedSessions[:0]
	for _, s := range p.cachedSessions {
		if s.Worker != dead {
			filteredCached = append(filteredCached, s)
		}
	}
	p.cachedSessions = filteredCached

	for k, s := range p.inFlightByHash {
		if s.Worker == dead {
			delete(p.inFlightByHash, k)
		}
	}
}

func (p *Pool) signalReplenish() {
	select {
	case p.replenishCh <- struct{}{}:
	default:
	}
}

func (p *Pool) replenishLoop() {
	defer p.wg.Done()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-p.poolCtx.Done():
			return
		case <-p.replenishCh:
			p.topUpIdleSessions()
		case <-ticker.C:
			p.topUpIdleSessions()
		}
	}
}

func (p *Pool) topUpIdleSessions() {
	target := p.cfg.Pool.IdlePoolSize
	if target <= 0 {
		return
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	deficit := target - (len(p.idleSessions) + p.warmingCount)
	if deficit <= 0 {
		p.mu.Unlock()
		return
	}
	p.warmingCount += deficit
	p.mu.Unlock()

	for i := 0; i < deficit; i++ {
		p.wg.Add(1)
		go func() {
			defer p.wg.Done()
			sess, err := p.createFreshSession(p.poolCtx)

			p.mu.Lock()
			p.warmingCount--
			if err == nil && !p.closed && len(p.idleSessions) < target {
				p.idleSessions = append(p.idleSessions, sess)
				p.mu.Unlock()
				return
			}
			p.mu.Unlock()

			if err == nil && sess != nil {
				p.retireSession(sess)
			} else if err != nil && p.poolCtx.Err() == nil {
				p.logger.Debug("background idle session warmup failed", "error", err)
			}
		}()
	}
}

func (p *Pool) janitorLoop() {
	defer p.wg.Done()
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-p.poolCtx.Done():
			return
		case <-ticker.C:
			p.evictExpiredSessions()
		}
	}
}

func (p *Pool) evictExpiredSessions() {
	now := time.Now()
	idleTTL := p.cfg.IdleSessionTTL()
	cachedTTL := p.cfg.CachedSessionTTL()

	var expired []*ManagedSession

	p.mu.Lock()
	keptIdle := p.idleSessions[:0]
	for _, s := range p.idleSessions {
		if !s.Worker.client.Alive() || s.Worker.draining.Load() || now.Sub(s.CreatedAt) > idleTTL {
			expired = append(expired, s)
		} else {
			keptIdle = append(keptIdle, s)
		}
	}
	p.idleSessions = keptIdle

	keptCached := p.cachedSessions[:0]
	for _, s := range p.cachedSessions {
		if !s.Worker.client.Alive() || s.Worker.draining.Load() || now.Sub(s.LastUsedAt) > cachedTTL {
			expired = append(expired, s)
		} else {
			keptCached = append(keptCached, s)
		}
	}
	p.cachedSessions = keptCached
	needReplenish := len(p.idleSessions)+p.warmingCount < p.cfg.Pool.IdlePoolSize
	p.mu.Unlock()

	for _, s := range expired {
		p.retireSession(s)
	}
	if needReplenish {
		p.signalReplenish()
	}
}

func checkStopSequences(combined string, prevLen int, stops []string) (string, string, bool) {
	bestIdx := -1
	bestSeq := ""
	for _, seq := range stops {
		if seq == "" {
			continue
		}
		searchStart := prevLen - len(seq)
		if searchStart < 0 {
			searchStart = 0
		}
		if rel := strings.Index(combined[searchStart:], seq); rel >= 0 {
			idx := searchStart + rel
			if bestIdx == -1 || idx < bestIdx {
				bestIdx = idx
				bestSeq = seq
			}
		}
	}
	if bestIdx >= 0 {
		return combined[:bestIdx], bestSeq, true
	}
	return "", "", false
}

func normalizeStopReason(acpStop string, hasToolCalls bool, hitStopSeq bool) string {
	if hasToolCalls {
		return "tool_calls"
	}
	if hitStopSeq {
		return "stop_sequence"
	}
	switch acpStop {
	case "max_tokens":
		return "max_tokens"
	case "cancelled":
		return "cancelled"
	default:
		return "end_turn"
	}
}

func computeTokenUsage(
	req *bridge.UnifiedRequest,
	thinking, text string,
	toolCalls []bridge.ToolCallValue,
	agentUsage *acp.UsageInfo,
) (int, int) {
	if agentUsage != nil {
		inTok := agentUsage.InputTokens
		if inTok <= 0 {
			inTok = agentUsage.PromptTokens
		}
		outTok := agentUsage.OutputTokens
		if inTok > 0 || outTok > 0 {
			if inTok <= 0 {
				inTok = bridge.EstimateRequestTokens(req)
			}
			if outTok <= 0 {
				outTok = estimateOutputTokens(thinking, text, toolCalls)
			}
			return inTok, outTok
		}
	}
	return bridge.EstimateRequestTokens(req), estimateOutputTokens(thinking, text, toolCalls)
}

func estimateOutputTokens(thinking, text string, toolCalls []bridge.ToolCallValue) int {
	out := bridge.EstimateTextTokens(thinking) + bridge.EstimateTextTokens(text)
	for _, tc := range toolCalls {
		out += bridge.EstimateTextTokens(tc.Name) + bridge.EstimateTextTokens(tc.Arguments) + 6
	}
	if out < 1 {
		out = 1
	}
	return out
}
