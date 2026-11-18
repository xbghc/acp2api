package acp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"acp2api/internal/config"
)

const (
	maxStderrTailLines = 25
	cancelGracePeriod  = 2 * time.Second
)

type callOutcome struct {
	result json.RawMessage
	rpcErr *RPCError
	err    error
}

// Client manages a single ACP agent subprocess over JSON-RPC 2.0 stdio.
// Multiple ACP sessions can be safely multiplexed over a single Client.
type Client struct {
	id         string
	cfg        config.ACPConfig
	logger     *slog.Logger
	host       *HostHandler
	procCtx    context.Context
	procCancel context.CancelFunc

	cmd   *exec.Cmd
	stdin io.WriteCloser

	writeMu sync.Mutex
	nextID  atomic.Int64

	pendingMu sync.Mutex
	pending   map[string]chan callOutcome

	reverseMu      sync.Mutex
	reverseCancels map[string]context.CancelFunc

	listenersMu sync.RWMutex
	listeners   map[string]func(ParsedUpdate)

	initMu     sync.RWMutex
	initResult InitializeResult

	unsupportedClose atomic.Bool
	unsupportedFork  atomic.Bool

	stderrMu   sync.Mutex
	stderrTail []string

	done      chan struct{}
	closeOnce sync.Once
	exitMu    sync.RWMutex
	exitErr   error
}

// NewClient constructs a new ACP stdio client (not started until Start is called).
func NewClient(id string, cfg config.ACPConfig, logger *slog.Logger) *Client {
	if logger == nil {
		logger = slog.Default()
	}
	l := logger.With("acp_worker", id)
	procCtx, procCancel := context.WithCancel(context.Background())
	return &Client{
		id:             id,
		cfg:            cfg,
		logger:         l,
		host:           NewHostHandlerWithContext(procCtx, cfg, l),
		procCtx:        procCtx,
		procCancel:     procCancel,
		pending:        make(map[string]chan callOutcome),
		reverseCancels: make(map[string]context.CancelFunc),
		listeners:      make(map[string]func(ParsedUpdate)),
		done:           make(chan struct{}),
	}
}

func (c *Client) ID() string {
	return c.id
}

// Done returns a channel that closes when the underlying ACP subprocess exits.
func (c *Client) Done() <-chan struct{} {
	return c.done
}

// Alive reports whether the ACP subprocess is still running.
func (c *Client) Alive() bool {
	select {
	case <-c.done:
		return false
	default:
		return true
	}
}

// InitResult returns the negotiated capabilities and agent info after Start.
func (c *Client) InitResult() InitializeResult {
	c.initMu.RLock()
	defer c.initMu.RUnlock()
	return c.initResult
}

// SupportsFork reports whether the ACP agent can fork existing sessions.
func (c *Client) SupportsFork() bool {
	if c.unsupportedFork.Load() {
		return false
	}
	return c.InitResult().AgentCapabilities.SessionCapabilities.HasFork()
}

// Start launches the ACP subprocess via stdio, completes initialize, and runs authenticate if configured.
func (c *Client) Start(ctx context.Context) error {
	cmd := exec.CommandContext(c.procCtx, c.cfg.Command, c.cfg.Args...)
	cmd.Dir = c.cfg.Cwd

	env := os.Environ()
	for k, v := range c.cfg.Env {
		env = append(env, k+"="+v)
	}
	cmd.Env = env

	stdin, err := cmd.StdinPipe()
	if err != nil {
		c.procCancel()
		return fmt.Errorf("create stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		c.procCancel()
		return fmt.Errorf("create stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		c.procCancel()
		return fmt.Errorf("create stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		c.procCancel()
		return fmt.Errorf("start acp subprocess %q: %w", c.cfg.Command, err)
	}
	c.cmd = cmd
	c.stdin = stdin

	go c.readStderrLoop(stderr)
	go c.readStdoutLoop(stdout)
	go c.waitLoop()

	initTimeout := time.Duration(c.cfg.InitializeTimeoutSecs) * time.Second
	if initTimeout <= 0 {
		initTimeout = 30 * time.Second
	}
	initCtx, cancel := context.WithTimeout(ctx, initTimeout)
	defer cancel()

	params := InitializeParams{
		ProtocolVersion: ProtocolVersion,
		ClientInfo: ImplementationInfo{
			Name:    "acp2api",
			Title:   "ACP to OpenAI/Anthropic API Gateway",
			Version: "1.1.0",
		},
		ClientCapabilities: ClientCapabilities{
			FS: FSCapabilities{
				ReadTextFile:  c.cfg.Capabilities.EnableFSRead,
				WriteTextFile: c.cfg.Capabilities.EnableFSWrite,
			},
			Terminal: c.cfg.Capabilities.EnableTerminal,
		},
	}

	var initRes InitializeResult
	if err := c.call(initCtx, "initialize", params, &initRes); err != nil {
		_ = c.Close()
		return fmt.Errorf("acp initialize failed: %w%s", err, c.formatStderrSuffix())
	}

	c.initMu.Lock()
	c.initResult = initRes
	c.initMu.Unlock()

	if authID := strings.TrimSpace(c.cfg.AuthMethodID); authID != "" {
		if err := c.Authenticate(initCtx, authID); err != nil {
			_ = c.Close()
			return fmt.Errorf("acp authenticate(%q) failed: %w%s", authID, err, c.formatStderrSuffix())
		}
	}

	c.logger.Info("ACP stdio worker initialized",
		"pid", cmd.Process.Pid,
		"agent_name", initRes.AgentInfo.Name,
		"agent_version", initRes.AgentInfo.Version,
		"protocol_version", initRes.ProtocolVersion,
		"image_support", initRes.AgentCapabilities.PromptCapabilities.Image,
		"audio_support", initRes.AgentCapabilities.PromptCapabilities.Audio,
		"embedded_context", initRes.AgentCapabilities.PromptCapabilities.EmbeddedContext,
	)
	return nil
}

// Authenticate invokes the ACP authenticate method with the selected methodId.
func (c *Client) Authenticate(ctx context.Context, methodID string) error {
	var ignored json.RawMessage
	return c.call(ctx, "authenticate", AuthenticateParams{MethodID: methodID}, &ignored)
}

func (c *Client) compatibleMCPServers() []config.MCPServerConfig {
	caps := c.InitResult().AgentCapabilities.MCPCapabilities
	out := make([]config.MCPServerConfig, 0, len(c.cfg.MCPServers))
	for _, m := range c.cfg.MCPServers {
		t := strings.ToLower(strings.TrimSpace(m.Type))
		if t == "http" && !caps.HTTP {
			c.logger.Warn("skipping HTTP MCP server because agent did not advertise mcpCapabilities.http", "mcp_name", m.Name)
			continue
		}
		if t == "sse" && !caps.SSE {
			c.logger.Warn("skipping SSE MCP server because agent did not advertise mcpCapabilities.sse", "mcp_name", m.Name)
			continue
		}
		out = append(out, m)
	}
	return out
}

// NewSession invokes session/new to create a fresh conversation session on this worker.
func (c *Client) NewSession(ctx context.Context) (SessionNewResult, error) {
	params := SessionNewParams{
		Cwd:        c.cfg.Cwd,
		MCPServers: c.compatibleMCPServers(),
	}
	var res SessionNewResult
	if err := c.call(ctx, "session/new", params, &res); err != nil {
		return SessionNewResult{}, fmt.Errorf("acp session/new failed: %w%s", err, c.formatStderrSuffix())
	}
	if strings.TrimSpace(res.SessionID) == "" {
		return SessionNewResult{}, errors.New("acp session/new returned empty sessionId")
	}
	if defMode := strings.TrimSpace(c.cfg.DefaultMode); defMode != "" {
		if updated, err := c.SetSessionMode(ctx, res.SessionID, defMode, res); err == nil && len(updated) > 0 {
			res.ConfigOptions = updated
		}
	}
	return res, nil
}

// LoadSession invokes session/load to restore a persisted ACP session (requires agent loadSession capability).
func (c *Client) LoadSession(ctx context.Context, sessionID string, onReplayUpdate func(ParsedUpdate)) (SessionNewResult, error) {
	if onReplayUpdate != nil {
		c.listenersMu.Lock()
		c.listeners[sessionID] = onReplayUpdate
		c.listenersMu.Unlock()
		defer func() {
			c.listenersMu.Lock()
			delete(c.listeners, sessionID)
			c.listenersMu.Unlock()
		}()
	}

	params := SessionLoadParams{
		SessionID:  sessionID,
		Cwd:        c.cfg.Cwd,
		MCPServers: c.compatibleMCPServers(),
	}
	var res SessionNewResult
	if err := c.call(ctx, "session/load", params, &res); err != nil {
		return SessionNewResult{}, fmt.Errorf("acp session/load failed: %w", err)
	}
	if res.SessionID == "" {
		res.SessionID = sessionID
	}
	return res, nil
}

// ResumeSession invokes session/resume to re-attach to an existing session without history replay.
func (c *Client) ResumeSession(ctx context.Context, sessionID string) (SessionNewResult, error) {
	params := SessionResumeParams{
		SessionID:  sessionID,
		Cwd:        c.cfg.Cwd,
		MCPServers: c.compatibleMCPServers(),
	}
	var res SessionNewResult
	if err := c.call(ctx, "session/resume", params, &res); err != nil {
		return SessionNewResult{}, fmt.Errorf("acp session/resume failed: %w", err)
	}
	if res.SessionID == "" {
		res.SessionID = sessionID
	}
	return res, nil
}

// ForkSession invokes session/fork to clone an existing session's state into a new independent session ID.
func (c *Client) ForkSession(ctx context.Context, sessionID string) (SessionNewResult, error) {
	if c.unsupportedFork.Load() {
		return SessionNewResult{}, &RPCError{Code: rpcErrMethodNotFound, Message: "session/fork not supported"}
	}
	params := SessionForkParams{
		SessionID:  sessionID,
		Cwd:        c.cfg.Cwd,
		MCPServers: c.compatibleMCPServers(),
	}
	var res SessionNewResult
	if err := c.call(ctx, "session/fork", params, &res); err != nil {
		var rpcErr *RPCError
		if errors.As(err, &rpcErr) && rpcErr.Code == rpcErrMethodNotFound {
			c.unsupportedFork.Store(true)
		}
		return SessionNewResult{}, err
	}
	if strings.TrimSpace(res.SessionID) == "" {
		return SessionNewResult{}, errors.New("acp session/fork returned empty sessionId")
	}
	return res, nil
}

// ListSessions invokes session/list to enumerate sessions known to the ACP agent.
func (c *Client) ListSessions(ctx context.Context, cursor string) (SessionListResult, error) {
	params := SessionListParams{
		Cwd:    c.cfg.Cwd,
		Cursor: cursor,
	}
	var res SessionListResult
	if err := c.call(ctx, "session/list", params, &res); err != nil {
		return SessionListResult{}, err
	}
	return res, nil
}

// CloseSession invokes session/close so the ACP agent can release memory associated with a retired session.
// If the agent does not implement session/close (-32601), it caches that fact and becomes a silent no-op.
func (c *Client) CloseSession(ctx context.Context, sessionID string) error {
	if strings.TrimSpace(sessionID) == "" || !c.Alive() || c.unsupportedClose.Load() {
		return nil
	}
	var ignored json.RawMessage
	err := c.call(ctx, "session/close", SessionCloseParams{SessionID: sessionID}, &ignored)
	if err != nil {
		var rpcErr *RPCError
		if errors.As(err, &rpcErr) && rpcErr.Code == rpcErrMethodNotFound {
			c.unsupportedClose.Store(true)
			return nil
		}
	}
	return err
}

// SetConfigOption updates a session configuration option via session/set_config_option and returns
// the updated ConfigOptions list if returned by the agent.
func (c *Client) SetConfigOption(ctx context.Context, sessionID, configID string, value any) ([]SessionConfigOption, error) {
	params := SetConfigOptionParams{
		SessionID: sessionID,
		ConfigID:  configID,
		Value:     value,
	}
	var res SetConfigOptionResult
	if err := c.call(ctx, "session/set_config_option", params, &res); err != nil {
		return nil, err
	}
	return res.ConfigOptions, nil
}

// SetSessionModel switches the model on an existing ACP session using session/set_config_option
// when advertised, falling back to session/set_model. Returns any updated ConfigOptions.
func (c *Client) SetSessionModel(ctx context.Context, sessionID, modelID string, meta SessionNewResult) ([]SessionConfigOption, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, nil
	}
	for _, opt := range meta.ConfigOptions {
		if strings.EqualFold(opt.Category, "model") || strings.EqualFold(opt.ID, "model") {
			return c.SetConfigOption(ctx, sessionID, opt.ID, modelID)
		}
	}
	params := SetModelParams{
		SessionID: sessionID,
		ModelID:   modelID,
	}
	var ignored json.RawMessage
	if err := c.call(ctx, "session/set_model", params, &ignored); err != nil {
		return nil, err
	}
	return nil, nil
}

// SetSessionMode switches the operational mode of an ACP session using session/set_config_option
// (if a category=="mode" configOption is present) or session/set_mode.
func (c *Client) SetSessionMode(ctx context.Context, sessionID, modeID string, meta SessionNewResult) ([]SessionConfigOption, error) {
	if strings.TrimSpace(modeID) == "" {
		return nil, nil
	}
	for _, opt := range meta.ConfigOptions {
		if strings.EqualFold(opt.Category, "mode") || strings.EqualFold(opt.ID, "mode") {
			return c.SetConfigOption(ctx, sessionID, opt.ID, modeID)
		}
	}
	params := SetSessionModeParams{
		SessionID: sessionID,
		ModeID:    modeID,
	}
	var ignored json.RawMessage
	if err := c.call(ctx, "session/set_mode", params, &ignored); err != nil {
		return nil, err
	}
	return nil, nil
}

// Prompt executes a user turn on sessionID, streaming session/update events to onUpdate.
// Returns (result, cancelledCleanly, error).
func (c *Client) Prompt(
	ctx context.Context,
	sessionID string,
	blocks []ContentBlock,
	meta map[string]any,
	onUpdate func(ParsedUpdate),
) (SessionPromptResult, bool, error) {
	if onUpdate != nil {
		c.listenersMu.Lock()
		c.listeners[sessionID] = onUpdate
		c.listenersMu.Unlock()
		defer func() {
			c.listenersMu.Lock()
			delete(c.listeners, sessionID)
			c.listenersMu.Unlock()
		}()
	}

	reqID := c.nextID.Add(1)
	idKey := strconv.FormatInt(reqID, 10)
	ch := make(chan callOutcome, 1)

	c.pendingMu.Lock()
	if !c.Alive() {
		c.pendingMu.Unlock()
		return SessionPromptResult{}, false, c.deadError()
	}
	c.pending[idKey] = ch
	c.pendingMu.Unlock()

	defer func() {
		c.pendingMu.Lock()
		delete(c.pending, idKey)
		c.pendingMu.Unlock()
	}()

	req := rpcRequest{
		JSONRPC: "2.0",
		ID:      reqID,
		Method:  "session/prompt",
		Params: SessionPromptParams{
			SessionID: sessionID,
			Prompt:    blocks,
			Meta:      meta,
		},
	}
	if err := c.writeJSONLine(req); err != nil {
		return SessionPromptResult{}, false, err
	}

	select {
	case outcome := <-ch:
		if outcome.err != nil {
			return SessionPromptResult{}, false, outcome.err
		}
		if outcome.rpcErr != nil {
			return SessionPromptResult{}, false, outcome.rpcErr
		}
		var res SessionPromptResult
		if len(outcome.result) > 0 {
			if err := json.Unmarshal(outcome.result, &res); err != nil {
				return SessionPromptResult{}, false, fmt.Errorf("decode session/prompt result: %w", err)
			}
		}
		return res, false, nil

	case <-ctx.Done():
		_ = c.CancelSession(sessionID)
		timer := time.NewTimer(cancelGracePeriod)
		defer timer.Stop()
		select {
		case <-ch:
			return SessionPromptResult{StopReason: "cancelled"}, true, ctx.Err()
		case <-timer.C:
			return SessionPromptResult{StopReason: "cancelled"}, false, ctx.Err()
		case <-c.done:
			return SessionPromptResult{StopReason: "cancelled"}, false, ctx.Err()
		}

	case <-c.done:
		return SessionPromptResult{}, false, c.deadError()
	}
}

// CancelSession sends a session/cancel notification for the given session.
func (c *Client) CancelSession(sessionID string) error {
	return c.writeJSONLine(rpcNotification{
		JSONRPC: "2.0",
		Method:  "session/cancel",
		Params:  SessionCancelParams{SessionID: sessionID},
	})
}

// Close terminates the ACP subprocess and releases all host resources.
func (c *Client) Close() error {
	c.closeOnce.Do(func() {
		c.host.CloseAll()
		c.writeMu.Lock()
		if c.stdin != nil {
			_ = c.stdin.Close()
		}
		c.writeMu.Unlock()

		if c.cmd != nil && c.cmd.Process != nil {
			select {
			case <-c.done:
			case <-time.After(500 * time.Millisecond):
				c.procCancel()
			}
		} else {
			c.procCancel()
		}
	})
	if c.cmd != nil {
		<-c.done
	}
	return nil
}

func (c *Client) call(ctx context.Context, method string, params any, out any) error {
	reqID := c.nextID.Add(1)
	idKey := strconv.FormatInt(reqID, 10)
	ch := make(chan callOutcome, 1)

	c.pendingMu.Lock()
	if !c.Alive() {
		c.pendingMu.Unlock()
		return c.deadError()
	}
	c.pending[idKey] = ch
	c.pendingMu.Unlock()

	defer func() {
		c.pendingMu.Lock()
		delete(c.pending, idKey)
		c.pendingMu.Unlock()
	}()

	if err := c.writeJSONLine(rpcRequest{
		JSONRPC: "2.0",
		ID:      reqID,
		Method:  method,
		Params:  params,
	}); err != nil {
		return err
	}

	select {
	case outcome := <-ch:
		if outcome.err != nil {
			return outcome.err
		}
		if outcome.rpcErr != nil {
			return outcome.rpcErr
		}
		if out != nil && len(outcome.result) > 0 && string(bytes.TrimSpace(outcome.result)) != "null" {
			if err := json.Unmarshal(outcome.result, out); err != nil {
				return fmt.Errorf("decode %s result: %w", method, err)
			}
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-c.done:
		return c.deadError()
	}
}

func (c *Client) writeJSONLine(v any) error {
	payload, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal json-rpc message: %w", err)
	}
	payload = bytes.ReplaceAll(payload, []byte{'\n'}, []byte{' '})
	payload = append(payload, '\n')

	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	if c.stdin == nil {
		return errors.New("acp stdin is not open")
	}
	_, err = c.stdin.Write(payload)
	return err
}

func (c *Client) readStdoutLoop(r io.Reader) {
	reader := bufio.NewReaderSize(r, 64*1024)
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			c.handleStdoutLine(bytes.TrimSpace(line))
		}
		if err != nil {
			if !errors.Is(err, io.EOF) && c.procCtx.Err() == nil {
				c.logger.Warn("ACP stdout read error", "error", err)
			}
			return
		}
	}
}

func (c *Client) handleStdoutLine(line []byte) {
	if len(line) == 0 {
		return
	}
	// Support JSON-RPC 2.0 batch arrays ([...]) as well as single objects ({...}).
	if line[0] == '[' {
		var batch []json.RawMessage
		if err := json.Unmarshal(line, &batch); err == nil {
			for _, item := range batch {
				c.handleSingleEnvelope(bytes.TrimSpace(item))
			}
			return
		}
	}
	if line[0] != '{' {
		c.logger.Debug("ignoring non-JSON line on ACP stdout", "line", string(line))
		return
	}
	c.handleSingleEnvelope(line)
}

func (c *Client) handleSingleEnvelope(raw []byte) {
	if len(raw) == 0 || raw[0] != '{' {
		return
	}
	var env rpcEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		c.logger.Warn("invalid JSON on ACP stdout", "error", err)
		return
	}

	hasID := len(env.ID) > 0 && string(bytes.TrimSpace(env.ID)) != "null"
	hasMethod := env.Method != ""

	switch {
	case hasMethod && hasID:
		idCopy := append(json.RawMessage(nil), env.ID...)
		paramsCopy := append(json.RawMessage(nil), env.Params...)
		method := env.Method
		go c.dispatchReverseRequest(idCopy, method, paramsCopy)

	case hasMethod && !hasID:
		switch env.Method {
		case "session/update":
			c.dispatchSessionUpdate(env.Params)
		case "$/cancelRequest", "$cancelRequest":
			c.handleCancelReverseRequest(env.Params)
		default:
			c.logger.Debug("received ACP notification", "method", env.Method)
		}

	case !hasMethod && hasID:
		idKey := normalizeRPCID(env.ID)
		c.pendingMu.Lock()
		ch, ok := c.pending[idKey]
		if ok {
			delete(c.pending, idKey)
		}
		c.pendingMu.Unlock()
		if ok {
			ch <- callOutcome{
				result: env.Result,
				rpcErr: env.Error,
			}
		}
	}
}

func (c *Client) handleCancelReverseRequest(raw json.RawMessage) {
	var p CancelRequestParams
	if err := json.Unmarshal(raw, &p); err != nil {
		return
	}
	idKey := normalizeRPCID(p.ID)
	c.reverseMu.Lock()
	cancelFn, ok := c.reverseCancels[idKey]
	c.reverseMu.Unlock()
	if ok && cancelFn != nil {
		cancelFn()
	}
}

func (c *Client) dispatchReverseRequest(id json.RawMessage, method string, params json.RawMessage) {
	idKey := normalizeRPCID(id)
	reqCtx, cancel := context.WithCancel(c.procCtx)
	c.reverseMu.Lock()
	c.reverseCancels[idKey] = cancel
	c.reverseMu.Unlock()

	defer func() {
		cancel()
		c.reverseMu.Lock()
		delete(c.reverseCancels, idKey)
		c.reverseMu.Unlock()
	}()

	if method == "$/ping" || method == "ping" {
		_ = c.writeJSONLine(rpcResponse{JSONRPC: "2.0", ID: id, Result: map[string]any{}})
		return
	}

	res, rpcErr := c.host.Handle(reqCtx, method, params)
	resp := rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
	}
	if rpcErr != nil {
		resp.Error = rpcErr
	} else {
		if res == nil {
			res = map[string]any{}
		}
		resp.Result = res
	}
	if err := c.writeJSONLine(resp); err != nil {
		c.logger.Warn("failed sending reverse RPC response", "method", method, "error", err)
	}
}

func (c *Client) dispatchSessionUpdate(raw json.RawMessage) {
	var p SessionUpdateParams
	if err := json.Unmarshal(raw, &p); err != nil {
		c.logger.Warn("failed decoding session/update params", "error", err)
		return
	}
	parsed, err := ParseSessionUpdate(p.Update)
	if err != nil {
		c.logger.Debug("failed decoding session/update payload", "session_id", p.SessionID, "error", err)
		return
	}
	c.listenersMu.RLock()
	fn := c.listeners[p.SessionID]
	c.listenersMu.RUnlock()
	if fn != nil {
		fn(parsed)
	}
}

func (c *Client) readStderrLoop(r io.Reader) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 16*1024), 512*1024)
	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		c.stderrMu.Lock()
		if len(c.stderrTail) >= maxStderrTailLines {
			c.stderrTail = append(c.stderrTail[1:], text)
		} else {
			c.stderrTail = append(c.stderrTail, text)
		}
		c.stderrMu.Unlock()
		c.logger.Debug("acp stderr", "msg", text)
	}
}

func (c *Client) waitLoop() {
	err := c.cmd.Wait()
	c.host.CloseAll()

	c.exitMu.Lock()
	c.exitErr = err
	c.exitMu.Unlock()
	close(c.done)

	deadErr := c.deadError()
	c.pendingMu.Lock()
	for k, ch := range c.pending {
		delete(c.pending, k)
		ch <- callOutcome{err: deadErr}
	}
	c.pendingMu.Unlock()

	if c.procCtx.Err() == nil {
		c.logger.Warn("ACP worker subprocess exited", "error", err, "stderr_tail", c.stderrSummary())
	}
}

func (c *Client) deadError() error {
	c.exitMu.RLock()
	err := c.exitErr
	c.exitMu.RUnlock()
	if err != nil {
		return fmt.Errorf("acp subprocess exited: %w%s", err, c.formatStderrSuffix())
	}
	return fmt.Errorf("acp subprocess exited unexpectedly%s", c.formatStderrSuffix())
}

func (c *Client) stderrSummary() string {
	c.stderrMu.Lock()
	defer c.stderrMu.Unlock()
	return strings.Join(c.stderrTail, " | ")
}

func (c *Client) formatStderrSuffix() string {
	s := c.stderrSummary()
	if s == "" {
		return ""
	}
	return " [stderr: " + s + "]"
}

func normalizeRPCID(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	return strings.Trim(s, `"`)
}
