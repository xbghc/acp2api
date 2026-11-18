package acp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"acp2api/internal/config"
)

// TestHelperACPClientSubprocess acts as a configurable stdio ACP peer for unit-testing acp.Client.
func TestHelperACPClientSubprocess(t *testing.T) {
	mode := os.Getenv("GO_ACP_CLIENT_TEST_MODE")
	if mode == "" {
		return
	}

	if mode == "fail_init_exit" {
		fmt.Fprintln(os.Stderr, "fatal boot error: missing token")
		os.Exit(2)
	}

	var writeMu sync.Mutex
	writeRaw := func(s string) {
		writeMu.Lock()
		_, _ = os.Stdout.WriteString(s + "\n")
		writeMu.Unlock()
	}
	writeJSON := func(v any) {
		b, _ := json.Marshal(v)
		writeRaw(string(b))
	}

	// Emit non-JSON startup banner & malformed JSON line to ensure Client ignores them gracefully.
	writeRaw("Starting ACP test agent banner...")
	writeRaw("{malformed_json_line")

	cancelledSessions := sync.Map{}
	reader := bufio.NewReader(os.Stdin)
	for {
		line, err := reader.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			var req rpcEnvelope
			if json.Unmarshal(line, &req) == nil && req.Method != "" {
				go func(env rpcEnvelope) {
					idStr := strings.Trim(strings.TrimSpace(string(env.ID)), `"`)
					switch env.Method {
					case "initialize":
						// Respond with stringified ID to test normalizeRPCID handling!
						writeJSON(map[string]any{
							"jsonrpc": "2.0",
							"id":      idStr,
							"result": map[string]any{
								"protocolVersion": 1,
								"agentInfo":       map[string]any{"name": "unit-acp", "version": "0.1"},
							},
						})

					case "session/new":
						if mode == "empty_session_id" {
							writeJSON(map[string]any{"jsonrpc": "2.0", "id": env.ID, "result": map[string]any{"sessionId": ""}})
							return
						}
						writeJSON(map[string]any{
							"jsonrpc": "2.0",
							"id":      env.ID,
							"result": map[string]any{
								"sessionId": "sess_unit_1",
							},
						})

					case "session/set_model":
						var p SetModelParams
						_ = json.Unmarshal(env.Params, &p)
						if p.ModelID == "unsupported" {
							writeJSON(map[string]any{
								"jsonrpc": "2.0",
								"id":      env.ID,
								"error":   map[string]any{"code": -32601, "message": "Method not found"},
							})
							return
						}
						writeJSON(map[string]any{"jsonrpc": "2.0", "id": env.ID, "result": map[string]any{}})

					case "session/cancel":
						var p SessionCancelParams
						_ = json.Unmarshal(env.Params, &p)
						cancelledSessions.Store(p.SessionID, true)

					case "session/prompt":
						var p SessionPromptParams
						_ = json.Unmarshal(env.Params, &p)
						txt := ""
						if len(p.Prompt) > 0 {
							txt = p.Prompt[0].Text
						}

						if txt == "SLOW_CANCEL_ME" {
							for i := 0; i < 40; i++ {
								if _, ok := cancelledSessions.Load(p.SessionID); ok {
									writeJSON(map[string]any{
										"jsonrpc": "2.0",
										"id":      env.ID,
										"result":  map[string]any{"stopReason": "cancelled"},
									})
									return
								}
								time.Sleep(15 * time.Millisecond)
							}
						}

						writeJSON(map[string]any{
							"jsonrpc": "2.0",
							"method":  "session/update",
							"params": map[string]any{
								"sessionId": p.SessionID,
								"update": map[string]any{
									"sessionUpdate": "agent_message_chunk",
									"content":       map[string]any{"type": "text", "text": "Hello from unit agent"},
								},
							},
						})
						writeJSON(map[string]any{
							"jsonrpc": "2.0",
							"id":      env.ID,
							"result": map[string]any{
								"stopReason": "end_turn",
								"usage":      map[string]any{"inputTokens": 5, "outputTokens": 4},
							},
						})
					}
				}(req)
			}
		}
		if err != nil {
			os.Exit(0)
		}
	}
}

func TestACPClientFullLifecycleCancelAndErrors(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	// 1. Subprocess exits immediately during initialize -> error includes stderr tail!
	failCfg := config.Default().ACP
	failCfg.Command = os.Args[0]
	failCfg.Args = []string{"-test.run=TestHelperACPClientSubprocess", "--"}
	failCfg.Env = map[string]string{"GO_ACP_CLIENT_TEST_MODE": "fail_init_exit"}
	failCfg.Cwd = t.TempDir()

	failClient := NewClient("w-fail", failCfg, logger)
	if err := failClient.Start(context.Background()); err == nil || !strings.Contains(err.Error(), "missing token") {
		t.Fatalf("expected stderr diagnostic in Start error, got: %v", err)
	}

	// 2. Normal client lifecycle: Start, InitResult, NewSession, SetSessionModel fallback, Prompt, Cancel, Close.
	okCfg := failCfg
	okCfg.Env = map[string]string{"GO_ACP_CLIENT_TEST_MODE": "normal"}
	client := NewClient("w-ok", okCfg, logger)
	if client.ID() != "w-ok" {
		t.Fatalf("unexpected client ID: %s", client.ID())
	}
	if err := client.Start(context.Background()); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer client.Close()

	if !client.Alive() || client.InitResult().AgentInfo.Name != "unit-acp" {
		t.Fatalf("unexpected InitResult: %+v", client.InitResult())
	}

	sess, err := client.NewSession(context.Background())
	if err != nil || sess.SessionID != "sess_unit_1" {
		t.Fatalf("NewSession failed: %+v err=%v", sess, err)
	}

	// Fallback session/set_model (since sess.ConfigOptions is empty)
	if err := client.SetSessionModel(context.Background(), sess.SessionID, "pro", sess); err != nil {
		t.Fatalf("SetSessionModel failed: %v", err)
	}
	if err := client.SetSessionModel(context.Background(), sess.SessionID, "unsupported", sess); err == nil {
		t.Fatalf("expected RPCError on unsupported model")
	}

	// Normal Prompt
	var chunks []string
	res, _, err := client.Prompt(context.Background(), sess.SessionID, []ContentBlock{{Type: "text", Text: "hi"}}, func(u ParsedUpdate) {
		if u.Kind == "agent_message_chunk" {
			chunks = append(chunks, u.Text)
		}
	})
	if err != nil || res.StopReason != "end_turn" || strings.Join(chunks, "") != "Hello from unit agent" {
		t.Fatalf("unexpected Prompt result: res=%+v chunks=%v err=%v", res, chunks, err)
	}

	// Prompt Cancellation via context timeout -> sends session/cancel and settles cleanly!
	cancelCtx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	_, cleanCancel, err := client.Prompt(cancelCtx, sess.SessionID, []ContentBlock{{Type: "text", Text: "SLOW_CANCEL_ME"}}, nil)
	if err == nil || !cleanCancel {
		t.Fatalf("expected clean cancellation ack, got clean=%v err=%v", cleanCancel, err)
	}

	// Close client and verify Done() closes and Alive() becomes false.
	if err := client.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	select {
	case <-client.Done():
	case <-time.After(time.Second):
		t.Fatalf("expected client.Done() to close")
	}
	if client.Alive() {
		t.Fatalf("expected Alive() == false after Close()")
	}

	// 3. Empty sessionId validation error
	emptySessCfg := failCfg
	emptySessCfg.Env = map[string]string{"GO_ACP_CLIENT_TEST_MODE": "empty_session_id"}
	cEmpty := NewClient("w-empty", emptySessCfg, logger)
	if err := cEmpty.Start(context.Background()); err != nil {
		t.Fatalf("Start cEmpty: %v", err)
	}
	defer cEmpty.Close()
	if _, err := cEmpty.NewSession(context.Background()); err == nil || !strings.Contains(err.Error(), "empty sessionId") {
		t.Fatalf("expected empty sessionId error, got: %v", err)
	}
}
