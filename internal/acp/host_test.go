package acp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"acp2api/internal/config"
)

func TestHostHandlerPermissionsAndFSCapabilities(t *testing.T) {
	cwd := t.TempDir()
	if err := os.WriteFile(filepath.Join(cwd, "safe.txt"), []byte("l1\nl2\nl3\nl4"), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := config.Default().ACP
	cfg.Cwd = cwd
	cfg.PermissionPolicy = config.PermissionAllowReadOnly
	cfg.Capabilities.EnableFSRead = true
	cfg.Capabilities.EnableFSWrite = true
	cfg.Capabilities.RestrictToCwd = true

	host := NewHostHandler(cfg, nil)
	defer host.CloseAll()

	// 1. Path traversal outside Cwd must be rejected for read & write.
	if _, rpcErr := host.Handle(context.Background(), "fs/read_text_file", json.RawMessage(`{"sessionId":"s1","path":"../outside.txt"}`)); rpcErr == nil {
		t.Fatalf("expected path traversal outside cwd to fail")
	}
	if _, rpcErr := host.Handle(context.Background(), "fs/write_text_file", json.RawMessage(`{"sessionId":"s1","path":"../outside.txt","content":"bad"}`)); rpcErr == nil {
		t.Fatalf("expected write path traversal outside cwd to fail")
	}
	if _, rpcErr := host.Handle(context.Background(), "fs/read_text_file", json.RawMessage(`{"sessionId":"s1","path":""}`)); rpcErr == nil {
		t.Fatalf("expected empty path to fail")
	}

	// 2. Write nested file inside Cwd & read paginated lines using both line/limit and lineStart/lineLimit.
	if _, rpcErr := host.Handle(context.Background(), "fs/write_text_file", json.RawMessage(`{"sessionId":"s1","path":"sub/dir/notes.txt","content":"a\nb\nc\nd"}`)); rpcErr != nil {
		t.Fatalf("unexpected write error: %v", rpcErr)
	}
	res, rpcErr := host.Handle(context.Background(), "fs/read_text_file", json.RawMessage(`{"sessionId":"s1","path":"sub/dir/notes.txt","line":2,"limit":2}`))
	if rpcErr != nil || res.(FSReadTextFileResult).Content != "b\nc" {
		t.Fatalf("unexpected paginated read: %+v err=%v", res, rpcErr)
	}
	resAlias, rpcErr := host.Handle(context.Background(), "fs/read_text_file", json.RawMessage(`{"sessionId":"s1","path":"safe.txt","lineStart":3,"lineLimit":5}`))
	if rpcErr != nil || resAlias.(FSReadTextFileResult).Content != "l3\nl4" {
		t.Fatalf("unexpected alias paginated read: %+v err=%v", resAlias, rpcErr)
	}
	resPastEOF, rpcErr := host.Handle(context.Background(), "fs/read_text_file", json.RawMessage(`{"sessionId":"s1","path":"safe.txt","line":99}`))
	if rpcErr != nil || resPastEOF.(FSReadTextFileResult).Content != "" {
		t.Fatalf("expected empty string past EOF, got %+v err=%v", resPastEOF, rpcErr)
	}

	// 3. Permission policies: allow_read_only, allow_all, deny_all, cancelled ctx, fallback option matching.
	allowRes, rpcErr := host.Handle(context.Background(), "session/request_permission", json.RawMessage(`{
		"sessionId":"s1",
		"toolCall":{"kind":"search"},
		"options":[
			{"optionId":"allow1","name":"Allow","kind":"allow_once"},
			{"optionId":"deny1","name":"Deny","kind":"reject_once"}
		]
	}`))
	if rpcErr != nil || allowRes.(RequestPermissionResult).Outcome.OptionID != "allow1" {
		t.Fatalf("expected search permission to select allow1, got %+v err=%v", allowRes, rpcErr)
	}

	denyRes, rpcErr := host.Handle(context.Background(), "session/request_permission", json.RawMessage(`{
		"sessionId":"s1",
		"toolCall":{"kind":"execute"},
		"options":[
			{"optionId":"allow1","name":"Allow","kind":"allow_once"},
			{"optionId":"deny_custom","name":"Deny","kind":"custom_deny"}
		]
	}`))
	if rpcErr != nil || denyRes.(RequestPermissionResult).Outcome.OptionID != "deny_custom" {
		t.Fatalf("expected execute permission under allow_read_only to select deny_custom, got %+v err=%v", denyRes, rpcErr)
	}

	// Cancelled context -> outcome "cancelled"
	cancelCtx, cancel := context.WithCancel(context.Background())
	cancel()
	cancRes, rpcErr := host.Handle(cancelCtx, "session/request_permission", json.RawMessage(`{"sessionId":"s1","options":[{"optionId":"a","kind":"allow_once"}]}`))
	if rpcErr != nil || cancRes.(RequestPermissionResult).Outcome.Outcome != "cancelled" {
		t.Fatalf("expected cancelled outcome, got %+v err=%v", cancRes, rpcErr)
	}

	// Empty options -> outcome "cancelled"
	emptyOptRes, rpcErr := host.Handle(context.Background(), "session/request_permission", json.RawMessage(`{"sessionId":"s1","options":[]}`))
	if rpcErr != nil || emptyOptRes.(RequestPermissionResult).Outcome.Outcome != "cancelled" {
		t.Fatalf("expected cancelled outcome on empty options, got %+v", emptyOptRes)
	}

	// Unknown method -> -32601
	if _, rpcErr := host.Handle(context.Background(), "unknown/method", json.RawMessage(`{}`)); rpcErr == nil || rpcErr.Code != rpcErrMethodNotFound {
		t.Fatalf("expected method not found error, got %v", rpcErr)
	}

	// Disabled capabilities check
	disabledCfg := config.Default().ACP
	disabledCfg.Cwd = cwd
	disabledHost := NewHostHandler(disabledCfg, nil)
	if _, rpcErr := disabledHost.Handle(context.Background(), "fs/read_text_file", json.RawMessage(`{"path":"safe.txt"}`)); rpcErr == nil {
		t.Fatalf("expected error when EnableFSRead=false")
	}
	if _, rpcErr := disabledHost.Handle(context.Background(), "fs/write_text_file", json.RawMessage(`{"path":"safe.txt","content":"x"}`)); rpcErr == nil {
		t.Fatalf("expected error when EnableFSWrite=false")
	}
	if _, rpcErr := disabledHost.Handle(context.Background(), "terminal/create", json.RawMessage(`{"command":"echo"}`)); rpcErr == nil {
		t.Fatalf("expected error when EnableTerminal=false")
	}
}

func TestHostHandlerTerminalLifecycleTruncationAndKill(t *testing.T) {
	cwd := t.TempDir()
	cfg := config.Default().ACP
	cfg.Cwd = cwd
	cfg.Capabilities.EnableTerminal = true
	cfg.Capabilities.RestrictToCwd = true

	host := NewHostHandler(cfg, nil)
	defer host.CloseAll()

	// 1. Create terminal with map env and small OutputByteLimit to verify truncation + exit code capture.
	createRes, rpcErr := host.Handle(context.Background(), "terminal/create", json.RawMessage(`{
		"sessionId": "s1",
		"command": "sh",
		"args": ["-c", "printf '%s_1234567890' \"$MY_VAR\"; exit 7"],
		"env": {"MY_VAR": "HELLO"},
		"outputByteLimit": 8
	}`))
	if rpcErr != nil {
		t.Fatalf("terminal/create failed: %v", rpcErr)
	}
	tid := createRes.(TerminalCreateResult).TerminalID

	waitRes, rpcErr := host.Handle(context.Background(), "terminal/wait_for_exit", json.RawMessage(`{"terminalId":"`+tid+`"}`))
	if rpcErr != nil {
		t.Fatalf("terminal/wait_for_exit failed: %v", rpcErr)
	}
	exitStatus := waitRes.(*TerminalExitStatus)
	if exitStatus.ExitCode == nil || *exitStatus.ExitCode != 7 {
		t.Fatalf("expected exit code 7, got %+v", exitStatus)
	}

	outRes, rpcErr := host.Handle(context.Background(), "terminal/output", json.RawMessage(`{"terminalId":"`+tid+`"}`))
	if rpcErr != nil {
		t.Fatalf("terminal/output failed: %v", rpcErr)
	}
	tor := outRes.(TerminalOutputResult)
	if !tor.Truncated || tor.Output != "HELLO_12" || tor.ExitStatus == nil || *tor.ExitStatus.ExitCode != 7 {
		t.Fatalf("unexpected truncated terminal output: %+v", tor)
	}

	if _, rpcErr := host.Handle(context.Background(), "terminal/release", json.RawMessage(`{"terminalId":"`+tid+`"}`)); rpcErr != nil {
		t.Fatalf("terminal/release failed: %v", rpcErr)
	}
	// Accessing released terminal must error.
	if _, rpcErr := host.Handle(context.Background(), "terminal/output", json.RawMessage(`{"terminalId":"`+tid+`"}`)); rpcErr == nil {
		t.Fatalf("expected error after terminal/release")
	}

	// 2. Test long-running terminal + wait timeout + terminal/kill + []EnvPair env format.
	longRes, rpcErr := host.Handle(context.Background(), "terminal/create", json.RawMessage(`{
		"sessionId": "s1",
		"command": "sh",
		"args": ["-c", "sleep 10"],
		"env": [{"name": "FOO", "value": "BAR"}]
	}`))
	if rpcErr != nil {
		t.Fatalf("long terminal/create failed: %v", rpcErr)
	}
	longID := longRes.(TerminalCreateResult).TerminalID

	shortCtx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, rpcErr := host.Handle(shortCtx, "terminal/wait_for_exit", json.RawMessage(`{"terminalId":"`+longID+`"}`)); rpcErr == nil {
		t.Fatalf("expected wait_for_exit timeout error on sleeping process")
	}

	if _, rpcErr := host.Handle(context.Background(), "terminal/kill", json.RawMessage(`{"terminalId":"`+longID+`"}`)); rpcErr != nil {
		t.Fatalf("terminal/kill failed: %v", rpcErr)
	}
	// After kill, wait_for_exit returns promptly.
	waitCtx, waitCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer waitCancel()
	if _, rpcErr := host.Handle(waitCtx, "terminal/wait_for_exit", json.RawMessage(`{"terminalId":"`+longID+`"}`)); rpcErr != nil {
		t.Fatalf("expected wait_for_exit to complete after kill: %v", rpcErr)
	}

	// Validation error on empty command
	if _, rpcErr := host.Handle(context.Background(), "terminal/create", json.RawMessage(`{"command":"  "}`)); rpcErr == nil || !strings.Contains(rpcErr.Message, "command") {
		t.Fatalf("expected empty command error, got %v", rpcErr)
	}
}
