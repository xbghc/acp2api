package acp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"

	"acp2api/internal/config"
)

const (
	rpcErrInvalidParams  = -32602
	rpcErrMethodNotFound = -32601
	rpcErrInternal       = -32603
	defaultTermByteLimit = 1024 * 1024 // 1 MiB
)

// HostHandler handles reverse JSON-RPC requests sent from the ACP agent to the client.
type HostHandler struct {
	cfg    config.ACPConfig
	logger *slog.Logger

	termSeq   atomic.Int64
	termMu    sync.Mutex
	terminals map[string]*managedTerminal
}

type managedTerminal struct {
	id        string
	cmd       *exec.Cmd
	cancel    context.CancelFunc
	done      chan struct{}
	mu        sync.Mutex
	buf       bytes.Buffer
	byteLimit int
	truncated bool
	exitCode  *int
	signal    string
}

func (t *managedTerminal) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	totalLen := len(p)
	remaining := t.byteLimit - t.buf.Len()
	if remaining <= 0 {
		t.truncated = true
		return totalLen, nil
	}
	if len(p) > remaining {
		t.buf.Write(p[:remaining])
		t.truncated = true
		return totalLen, nil
	}
	t.buf.Write(p)
	return totalLen, nil
}

func (t *managedTerminal) snapshot() (string, bool, *TerminalExitStatus) {
	t.mu.Lock()
	defer t.mu.Unlock()
	var status *TerminalExitStatus
	select {
	case <-t.done:
		status = &TerminalExitStatus{
			ExitCode: t.exitCode,
			Signal:   t.signal,
		}
	default:
	}
	return t.buf.String(), t.truncated, status
}

func NewHostHandler(cfg config.ACPConfig, logger *slog.Logger) *HostHandler {
	if logger == nil {
		logger = slog.Default()
	}
	return &HostHandler{
		cfg:       cfg,
		logger:    logger,
		terminals: make(map[string]*managedTerminal),
	}
}

// Handle dispatches a reverse JSON-RPC request from the ACP agent.
func (h *HostHandler) Handle(ctx context.Context, method string, params json.RawMessage) (any, *RPCError) {
	switch method {
	case "session/request_permission":
		return h.handlePermission(ctx, params)
	case "fs/read_text_file":
		return h.handleFSRead(params)
	case "fs/write_text_file":
		return h.handleFSWrite(params)
	case "terminal/create":
		return h.handleTerminalCreate(ctx, params)
	case "terminal/output":
		return h.handleTerminalOutput(params)
	case "terminal/wait_for_exit":
		return h.handleTerminalWait(ctx, params)
	case "terminal/kill":
		return h.handleTerminalKill(params)
	case "terminal/release":
		return h.handleTerminalRelease(params)
	default:
		return nil, &RPCError{
			Code:    rpcErrMethodNotFound,
			Message: fmt.Sprintf("unsupported client method: %s", method),
		}
	}
}

// CloseAll terminates and cleans up any terminals spawned by the agent.
func (h *HostHandler) CloseAll() {
	h.termMu.Lock()
	terms := make([]*managedTerminal, 0, len(h.terminals))
	for _, t := range h.terminals {
		terms = append(terms, t)
	}
	h.terminals = make(map[string]*managedTerminal)
	h.termMu.Unlock()

	for _, t := range terms {
		t.cancel()
	}
}

func (h *HostHandler) handlePermission(ctx context.Context, raw json.RawMessage) (any, *RPCError) {
	if ctx.Err() != nil {
		return RequestPermissionResult{
			Outcome: PermissionOutcome{Outcome: "cancelled"},
		}, nil
	}
	var req RequestPermissionParams
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
	}

	shouldAllow := false
	switch h.cfg.PermissionPolicy {
	case config.PermissionAllowAll:
		shouldAllow = true
	case config.PermissionAllowReadOnly:
		if req.ToolCall != nil {
			switch req.ToolCall.Kind {
			case "read", "search", "think", "fetch":
				shouldAllow = true
			}
		}
	case config.PermissionDenyAll:
		shouldAllow = false
	}

	optionID := pickPermissionOption(req.Options, shouldAllow)
	if optionID == "" {
		return RequestPermissionResult{
			Outcome: PermissionOutcome{Outcome: "cancelled"},
		}, nil
	}

	h.logger.Debug("answered ACP permission request",
		"session_id", req.SessionID,
		"policy", h.cfg.PermissionPolicy,
		"allow", shouldAllow,
		"option_id", optionID,
	)
	return RequestPermissionResult{
		Outcome: PermissionOutcome{
			Outcome:  "selected",
			OptionID: optionID,
		},
	}, nil
}

func pickPermissionOption(options []PermissionOption, allow bool) string {
	if len(options) == 0 {
		return ""
	}
	if allow {
		for _, k := range []string{"allow_once", "allow_always"} {
			for _, o := range options {
				if o.Kind == k {
					return o.OptionID
				}
			}
		}
		for _, o := range options {
			if strings.Contains(strings.ToLower(o.Kind), "allow") || strings.Contains(strings.ToLower(o.OptionID), "allow") {
				return o.OptionID
			}
		}
		return options[0].OptionID
	}

	for _, k := range []string{"reject_once", "reject_always"} {
		for _, o := range options {
			if o.Kind == k {
				return o.OptionID
			}
		}
	}
	for _, o := range options {
		lower := strings.ToLower(o.Kind + " " + o.OptionID)
		if strings.Contains(lower, "reject") || strings.Contains(lower, "deny") {
			return o.OptionID
		}
	}
	return ""
}

func (h *HostHandler) resolveAllowedPath(target string) (string, error) {
	if strings.TrimSpace(target) == "" {
		return "", errors.New("path is required")
	}
	resolved := target
	if !filepath.IsAbs(resolved) {
		resolved = filepath.Join(h.cfg.Cwd, resolved)
	}
	resolved = filepath.Clean(resolved)

	if !h.cfg.Capabilities.RestrictToCwd {
		return resolved, nil
	}
	base := filepath.Clean(h.cfg.Cwd)
	rel, err := filepath.Rel(base, resolved)
	if err != nil {
		return "", fmt.Errorf("path check failed: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q is outside allowed working directory %q", target, base)
	}
	return resolved, nil
}

func (h *HostHandler) handleFSRead(raw json.RawMessage) (any, *RPCError) {
	if !h.cfg.Capabilities.EnableFSRead {
		return nil, &RPCError{Code: rpcErrMethodNotFound, Message: "fs/read_text_file capability is disabled"}
	}
	var req FSReadTextFileParams
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
	}
	cleanPath, err := h.resolveAllowedPath(req.Path)
	if err != nil {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
	}
	data, err := os.ReadFile(cleanPath)
	if err != nil {
		return nil, &RPCError{Code: rpcErrInternal, Message: err.Error()}
	}
	content := string(data)
	startLine := req.Line
	if startLine <= 0 {
		startLine = req.LineStart
	}
	limitLines := req.Limit
	if limitLines <= 0 {
		limitLines = req.LineLimit
	}
	if startLine > 0 || limitLines > 0 {
		lines := strings.Split(content, "\n")
		idx := 0
		if startLine > 1 {
			idx = startLine - 1
		}
		if idx >= len(lines) {
			content = ""
		} else {
			end := len(lines)
			if limitLines > 0 && idx+limitLines < end {
				end = idx + limitLines
			}
			content = strings.Join(lines[idx:end], "\n")
		}
	}
	return FSReadTextFileResult{Content: content}, nil
}

func (h *HostHandler) handleFSWrite(raw json.RawMessage) (any, *RPCError) {
	if !h.cfg.Capabilities.EnableFSWrite {
		return nil, &RPCError{Code: rpcErrMethodNotFound, Message: "fs/write_text_file capability is disabled"}
	}
	var req FSWriteTextFileParams
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
	}
	cleanPath, err := h.resolveAllowedPath(req.Path)
	if err != nil {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
	}
	if err := os.MkdirAll(filepath.Dir(cleanPath), 0o755); err != nil {
		return nil, &RPCError{Code: rpcErrInternal, Message: err.Error()}
	}
	if err := os.WriteFile(cleanPath, []byte(req.Content), 0o644); err != nil {
		return nil, &RPCError{Code: rpcErrInternal, Message: err.Error()}
	}
	return map[string]any{}, nil
}

func (h *HostHandler) handleTerminalCreate(parentCtx context.Context, raw json.RawMessage) (any, *RPCError) {
	if !h.cfg.Capabilities.EnableTerminal {
		return nil, &RPCError{Code: rpcErrMethodNotFound, Message: "terminal capability is disabled"}
	}
	var req TerminalCreateParams
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
	}
	if strings.TrimSpace(req.Command) == "" {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: "command is required"}
	}

	dir := h.cfg.Cwd
	if strings.TrimSpace(req.Cwd) != "" {
		resolved, err := h.resolveAllowedPath(req.Cwd)
		if err != nil {
			return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
		}
		dir = resolved
	}

	limit := req.OutputByteLimit
	if limit <= 0 {
		limit = defaultTermByteLimit
	}

	cmdCtx, cancel := context.WithCancel(parentCtx)
	cmd := exec.CommandContext(cmdCtx, req.Command, req.Args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), parseTerminalEnv(req.Env)...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process != nil {
			return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		}
		return nil
	}

	id := fmt.Sprintf("term_%d", h.termSeq.Add(1))
	mt := &managedTerminal{
		id:        id,
		cmd:       cmd,
		cancel:    cancel,
		done:      make(chan struct{}),
		byteLimit: limit,
	}
	cmd.Stdout = mt
	cmd.Stderr = mt

	if err := cmd.Start(); err != nil {
		cancel()
		return nil, &RPCError{Code: rpcErrInternal, Message: err.Error()}
	}

	h.termMu.Lock()
	h.terminals[id] = mt
	h.termMu.Unlock()

	go func() {
		err := cmd.Wait()
		mt.mu.Lock()
		code := 0
		if err != nil {
			var exitErr *exec.ExitError
			if errors.As(err, &exitErr) {
				code = exitErr.ExitCode()
			} else {
				code = -1
			}
		}
		mt.exitCode = &code
		mt.mu.Unlock()
		close(mt.done)
	}()

	return TerminalCreateResult{TerminalID: id}, nil
}

func parseTerminalEnv(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var pairs []config.EnvPair
	if err := json.Unmarshal(raw, &pairs); err == nil && len(pairs) > 0 {
		out := make([]string, 0, len(pairs))
		for _, p := range pairs {
			if p.Name != "" {
				out = append(out, p.Name+"="+p.Value)
			}
		}
		return out
	}
	var m map[string]string
	if err := json.Unmarshal(raw, &m); err == nil && len(m) > 0 {
		out := make([]string, 0, len(m))
		for k, v := range m {
			out = append(out, k+"="+v)
		}
		return out
	}
	return nil
}

func (h *HostHandler) getTerminal(id string) (*managedTerminal, *RPCError) {
	h.termMu.Lock()
	defer h.termMu.Unlock()
	t, ok := h.terminals[id]
	if !ok {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: fmt.Sprintf("unknown terminalId %q", id)}
	}
	return t, nil
}

func (h *HostHandler) handleTerminalOutput(raw json.RawMessage) (any, *RPCError) {
	var req TerminalIDParams
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
	}
	t, rpcErr := h.getTerminal(req.TerminalID)
	if rpcErr != nil {
		return nil, rpcErr
	}
	output, truncated, status := t.snapshot()
	return TerminalOutputResult{
		Output:     output,
		Truncated:  truncated,
		ExitStatus: status,
	}, nil
}

func (h *HostHandler) handleTerminalWait(ctx context.Context, raw json.RawMessage) (any, *RPCError) {
	var req TerminalIDParams
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
	}
	t, rpcErr := h.getTerminal(req.TerminalID)
	if rpcErr != nil {
		return nil, rpcErr
	}
	select {
	case <-t.done:
		_, _, status := t.snapshot()
		if status == nil {
			zero := 0
			status = &TerminalExitStatus{ExitCode: &zero}
		}
		return status, nil
	case <-ctx.Done():
		return nil, &RPCError{Code: rpcErrInternal, Message: ctx.Err().Error()}
	}
}

func (h *HostHandler) handleTerminalKill(raw json.RawMessage) (any, *RPCError) {
	var req TerminalIDParams
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
	}
	t, rpcErr := h.getTerminal(req.TerminalID)
	if rpcErr != nil {
		return nil, rpcErr
	}
	t.cancel()
	return map[string]any{}, nil
}

func (h *HostHandler) handleTerminalRelease(raw json.RawMessage) (any, *RPCError) {
	var req TerminalIDParams
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: err.Error()}
	}
	h.termMu.Lock()
	t, ok := h.terminals[req.TerminalID]
	if ok {
		delete(h.terminals, req.TerminalID)
	}
	h.termMu.Unlock()
	if !ok {
		return nil, &RPCError{Code: rpcErrInvalidParams, Message: fmt.Sprintf("unknown terminalId %q", req.TerminalID)}
	}
	t.cancel()
	return map[string]any{}, nil
}
