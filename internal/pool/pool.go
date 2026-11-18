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
	cachedSessions   []*ManagedSession // ordered oldest -> newest (LRU at index 0)
	discoveredModels map[string]string // modelId -> name
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

	// Pre-warm initial idle sessions synchronously (bounded by SessionNewTimeout) so first requests are instant.
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

// ExecuteTurn runs a single chat/messages API request against a reused or pre-warmed ACP session.
// If onEvent is non-nil, normalized StreamEvents are delivered in real time as chunks arrive.
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

	targetACPModel, displayModel := p.cfg.ResolveModel(req.Model)
	activeTools := req.Tools
	if req.ToolChoice.Mode == "none" {
		activeTools = nil
	}

	sess, startIdx, reused, err := p.acquireSession(ctx, req, targetACPModel, activeTools)
	if err != nil {
		return nil, err
	}

	sess.Worker.activeTurns.Add(1)
	p.activeTurns.Add(1)
	defer func() {
		sess.Worker.activeTurns.Add(-1)
		p.activeTurns.Add(-1)
	}()

	// Switch ACP session model if requested and different from current session state.
	if targetACPModel != "" && targetACPModel != sess.CurrentModel {
		if err := sess.Worker.client.SetSessionModel(ctx, sess.ID, targetACPModel, sess.Meta); err != nil {
			var rpcErr *acp.RPCError
			if errors.As(err, &rpcErr) && rpcErr.Code == -32601 {
				p.logger.Debug("ACP agent does not implement model switching; using agent default", "requested_model", targetACPModel)
			} else {
				p.logger.Warn("ACP SetSessionModel returned warning", "session_id", sess.ID, "model", targetACPModel, "error", err)
			}
		} else {
			sess.CurrentModel = targetACPModel
		}
	}

	blocks := bridge.BuildPromptBlocks(req, startIdx)

	var (
		thinkingBuf   strings.Builder
		textBuf       strings.Builder
		toolCalls     []bridge.ToolCallValue
		agentUsage    *acp.UsageInfo
		stoppedSeq    string
		stopTriggered bool
	)

	turnCtx, cancelTurn := context.WithCancel(ctx)
	defer cancelTurn()

	emitEvent := func(ev bridge.StreamEvent) {
		switch ev.Kind {
		case bridge.EventThinkingDelta:
			thinkingBuf.WriteString(ev.Text)
			if onEvent != nil {
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
	parser := bridge.NewToolStreamParser(toolsEnabled, emitEvent)

	promptRes, _, promptErr := sess.Worker.client.Prompt(turnCtx, sess.ID, blocks, func(u acp.ParsedUpdate) {
		switch u.Kind {
		case "agent_thought_chunk":
			if p.cfg.ACP.ExposeThoughts && u.Text != "" {
				emitEvent(bridge.StreamEvent{Kind: bridge.EventThinkingDelta, Text: u.Text})
			}
		case "agent_message_chunk":
			if u.Text != "" {
				parser.Feed(u.Text)
			}
		case "usage_update":
			if u.Usage != nil {
				agentUsage = u.Usage
			}
		}
	})

	parser.Finish()

	// If we intentionally cancelled turnCtx due to matching a stop_sequence, treat as clean completion!
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

	resp := &bridge.TurnResponse{
		Model:         displayModel,
		SessionID:     sess.ID,
		SessionReused: reused,
		Thinking:      thinkingBuf.String(),
		Text:          textBuf.String(),
		ToolCalls:     toolCalls,
		StopReason:    stopReason,
		StopSequence:  stoppedSeq,
		InputTokens:   inTokens,
		OutputTokens:  outTokens,
	}

	// Decide whether to store session in conversation cache for subsequent turns or retire it.
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
	// 1. Try smart multi-turn prefix match against cached conversation sessions.
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

func (p *Pool) checkoutCachedByHash(hash, sessionHint string) *ManagedSession {
	now := time.Now()
	ttl := p.cfg.CachedSessionTTL()

	p.mu.Lock()
	defer p.mu.Unlock()

	bestIdx := -1
	// Search newest -> oldest so most recent matching session wins; prefer matching SessionHint if provided.
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
	remaining := sess.Worker.liveSessions.Add(-1)
	if sess.Worker.draining.Load() && remaining <= 0 && sess.Worker.activeTurns.Load() <= 0 {
		go func(w *acpWorker) {
			_ = w.client.Close()
		}(sess.Worker)
	}
	p.maybeScheduleWorkerRotation(sess.Worker)
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
	p.recordDiscoveredModels(meta)

	now := time.Now()
	sess := &ManagedSession{
		ID:           meta.SessionID,
		Worker:       w,
		Meta:         meta,
		CurrentModel: currentModel,
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
		if opt.Category == "model" || opt.ID == "model" {
			if opt.CurrentValue != "" {
				p.discoveredModels[opt.CurrentValue] = opt.CurrentValue
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

	// Purge dead sessions belonging to w.
	p.purgeWorkerSessions(w)

	p.mu.Lock()
	stillAssigned := !p.closed && w.slot >= 0 && w.slot < len(p.workers) && p.workers[w.slot] == w
	p.mu.Unlock()
	if !stillAssigned {
		return
	}

	// Respawn worker with exponential backoff.
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
