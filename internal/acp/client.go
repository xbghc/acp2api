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

	listenersMu sync.RWMutex
	listeners   map[string]func(ParsedUpdate)

	initMu     sync.RWMutex
	initResult InitializeResult

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
		id:         id,
		cfg:        cfg,
		logger:     l,
		host:       NewHostHandler(cfg, l),
		procCtx:    procCtx,
		procCancel: procCancel,
		pending:    make(map[string]chan callOutcome),
		listeners:  make(map[string]func(ParsedUpdate)),
		done:       make(chan struct{}),
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

// Start launches the ACP subprocess via stdio and completes the initialize handshake.
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
			Version: "1.0.0",
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

	c.logger.Info("ACP stdio worker initialized",
		"pid", cmd.Process.Pid,
		"agent_name", initRes.AgentInfo.Name,
		"agent_version", initRes.AgentInfo.Version,
		"protocol_version", initRes.ProtocolVersion,
		"image_support", initRes.AgentCapabilities.PromptCapabilities.Image,
	)
	return nil
}

// NewSession invokes session/new to create a fresh conversation session on this worker.
func (c *Client) NewSession(ctx context.Context) (SessionNewResult, error) {
	mcpServers := c.cfg.MCPServers
	if mcpServers == nil {
		mcpServers = []config.MCPServerConfig{}
	}
	params := SessionNewParams{
		Cwd:        c.cfg.Cwd,
		MCPServers: mcpServers,
	}
	var res SessionNewResult
	if err := c.call(ctx, "session/new", params, &res); err != nil {
		return SessionNewResult{}, fmt.Errorf("acp session/new failed: %w%s", err, c.formatStderrSuffix())
	}
	if strings.TrimSpace(res.SessionID) == "" {
		return SessionNewResult{}, errors.New("acp session/new returned empty sessionId")
	}
	return res, nil
}

// SetSessionModel switches the model on an existing ACP session using session/set_config_option
// when advertised, falling back to session/set_model.
func (c *Client) SetSessionModel(ctx context.Context, sessionID, modelID string, meta SessionNewResult) error {
	if strings.TrimSpace(modelID) == "" {
		return nil
	}
	for _, opt := range meta.ConfigOptions {
		if opt.Category == "model" || opt.ID == "model" {
			params := SetConfigOptionParams{
				SessionID: sessionID,
				ConfigID:  opt.ID,
				Value:     modelID,
			}
			var ignored json.RawMessage
			return c.call(ctx, "session/set_config_option", params, &ignored)
		}
	}
	params := SetModelParams{
		SessionID: sessionID,
		ModelID:   modelID,
	}
	var ignored json.RawMessage
	return c.call(ctx, "session/set_model", params, &ignored)
}

// Prompt executes a user turn on sessionID, streaming session/update events to onUpdate.
// Returns (result, cancelledCleanly, error).
func (c *Client) Prompt(
	ctx context.Context,
	sessionID string,
	blocks []ContentBlock,
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
		// Notify ACP agent immediately to cancel in-flight turn.
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

		// Give the subprocess a brief moment to exit cleanly on EOF before killing.
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
		if out != nil && len(outcome.result) > 0 {
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
	// Guarantee no embedded newline violates ACP stdio line framing.
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
	if line[0] != '{' {
		c.logger.Debug("ignoring non-JSON line on ACP stdout", "line", string(line))
		return
	}
	var env rpcEnvelope
	if err := json.Unmarshal(line, &env); err != nil {
		c.logger.Warn("invalid JSON on ACP stdout", "error", err)
		return
	}

	hasID := len(env.ID) > 0 && string(bytes.TrimSpace(env.ID)) != "null"
	hasMethod := env.Method != ""

	switch {
	case hasMethod && hasID:
		// Reverse request from Agent -> Client; handle in goroutine so stdout reader never blocks.
		idCopy := append(json.RawMessage(nil), env.ID...)
		paramsCopy := append(json.RawMessage(nil), env.Params...)
		method := env.Method
		go c.dispatchReverseRequest(idCopy, method, paramsCopy)

	case hasMethod && !hasID:
		// Notification from Agent -> Client.
		if env.Method == "session/update" {
			c.dispatchSessionUpdate(env.Params)
		} else {
			c.logger.Debug("received ACP notification", "method", env.Method)
		}

	case !hasMethod && hasID:
		// Response to a Client -> Agent request.
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

func (c *Client) dispatchReverseRequest(id json.RawMessage, method string, params json.RawMessage) {
	res, rpcErr := c.host.Handle(c.procCtx, method, params)
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
