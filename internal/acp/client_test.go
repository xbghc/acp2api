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
						writeJSON(map[string]any{
							"jsonrpc": "2.0",
							"id":      idStr,
							"result": map[string]any{
								"protocolVersion": 1,
								"agentInfo":       map[string]any{"name": "unit-acp", "version": "0.2"},
								"agentCapabilities": map[string]any{
									"loadSession": true,
									"mcpCapabilities": map[string]any{
										"http": true,
										"sse":  false,
									},
									"promptCapabilities": map[string]any{
										"image":           true,
										"audio":           true,
										"embeddedContext": true,
									},
									"sessionCapabilities": map[string]any{
										"fork":   map[string]any{},
										"resume": map[string]any{},
										"list":   map[string]any{},
										"close":  map[string]any{},
									},
								},
							},
						})

					case "authenticate":
						var p AuthenticateParams
						_ = json.Unmarshal(env.Params, &p)
						if p.MethodID == "bad_auth" {
							writeJSON(map[string]any{"jsonrpc": "2.0", "id": env.ID, "error": map[string]any{"code": -32000, "message": "Auth failed"}})
							return
						}
						writeJSON(map[string]any{"jsonrpc": "2.0", "id": env.ID, "result": map[string]any{}})

					case "session/new":
						if mode == "empty_session_id" {
							writeJSON(map[string]any{"jsonrpc": "2.0", "id": env.ID, "result": map[string]any{"sessionId": ""}})
							return
						}
						writeJSON(map[string]any{
							"jsonrpc": "2.0",
							"id":      env.ID,
							"result":  map[string]any{"sessionId": "sess_unit_1"},
						})

					case "session/load":
						var p SessionLoadParams
						_ = json.Unmarshal(env.Params, &p)
						writeJSON(map[string]any{
							"jsonrpc": "2.0",
							"method":  "session/update",
							"params": map[string]any{
								"sessionId": p.SessionID,
								"update": map[string]any{
									"sessionUpdate": "user_message_chunk",
									"content":       map[string]any{"type": "text", "text": "replayed turn"},
								},
							},
						})
						writeJSON(map[string]any{"jsonrpc": "2.0", "id": env.ID, "result": map[string]any{"sessionId": p.SessionID}})

					case "session/resume":
						var p SessionResumeParams
						_ = json.Unmarshal(env.Params, &p)
						writeJSON(map[string]any{"jsonrpc": "2.0", "id": env.ID, "result": map[string]any{"sessionId": p.SessionID}})

					case "session/fork":
						writeJSON(map[string]any{"jsonrpc": "2.0", "id": env.ID, "result": map[string]any{"sessionId": "sess_forked_2"}})

					case "session/list":
						writeJSON(map[string]any{
							"jsonrpc": "2.0",
							"id":      env.ID,
							"result": map[string]any{
								"sessions": []map[string]any{{"sessionId": "sess_unit_1", "title": "Unit Chat"}},
							},
						})

					case "session/close":
						writeJSON(map[string]any{"jsonrpc": "2.0", "id": env.ID, "result": map[string]any{}})

					case "session/set_mode":
						writeJSON(map[string]any{"jsonrpc": "2.0", "id": env.ID, "result": map[string]any{}})

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

						// Emit a JSON-RPC 2.0 batch notification array to verify batch parsing!
						writeJSON([]any{
							map[string]any{
								"jsonrpc": "2.0",
								"method":  "session/update",
								"params": map[string]any{
									"sessionId": p.SessionID,
									"update": map[string]any{
										"sessionUpdate": "agent_message_chunk",
										"content":       map[string]any{"type": "text", "text": "Hello from "},
									},
								},
							},
							map[string]any{
								"jsonrpc": "2.0",
								"method":  "session/update",
								"params": map[string]any{
									"sessionId": p.SessionID,
									"update": map[string]any{
										"sessionUpdate": "agent_message_chunk",
										"content":       map[string]any{"type": "text", "text": "unit agent"},
									},
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

func TestACPClientFullLifecycleCancelAndExtendedMethods(t *testing.T) {
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

	// 2. Full ACP lifecycle: Start + Authenticate + MCP filtering + New/Load/Resume/Fork/List/Close/SetMode/Prompt/Cancel!
	okCfg := failCfg
	okCfg.Env = map[string]string{"GO_ACP_CLIENT_TEST_MODE": "normal"}
	okCfg.AuthMethodID = "api_key"
	okCfg.DefaultMode = "code"
	okCfg.MCPServers = []config.MCPServerConfig{
		{Name: "local-stdio", Command: "mcp-local"},
		{Type: "http", Name: "remote-http", URL: "https://example.com/mcp"},
		{Type: "sse", Name: "unsupported-sse", URL: "https://example.com/sse"},
	}

	client := NewClient("w-ok", okCfg, logger)
	if err := client.Start(context.Background()); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer client.Close()

	if !client.Alive() || !client.SupportsFork() || client.InitResult().AgentInfo.Name != "unit-acp" {
		t.Fatalf("unexpected InitResult: %+v", client.InitResult())
	}

	sess, err := client.NewSession(context.Background())
	if err != nil || sess.SessionID != "sess_unit_1" {
		t.Fatalf("NewSession failed: %+v err=%v", sess, err)
	}

	// Test LoadSession with history replay
	var replayed []string
	if _, err := client.LoadSession(context.Background(), "sess_loaded", func(u ParsedUpdate) {
		if u.Kind == "user_message_chunk" {
			replayed = append(replayed, u.Text)
		}
	}); err != nil || len(replayed) != 1 || replayed[0] != "replayed turn" {
		t.Fatalf("LoadSession failed: replayed=%v err=%v", replayed, err)
	}

	// Test ResumeSession, ForkSession, ListSessions
	if resumed, err := client.ResumeSession(context.Background(), "sess_resumed"); err != nil || resumed.SessionID != "sess_resumed" {
		t.Fatalf("ResumeSession failed: %+v err=%v", resumed, err)
	}
	if forked, err := client.ForkSession(context.Background(), sess.SessionID); err != nil || forked.SessionID != "sess_forked_2" {
		t.Fatalf("ForkSession failed: %+v err=%v", forked, err)
	}
	if listed, err := client.ListSessions(context.Background(), ""); err != nil || len(listed.Sessions) != 1 {
		t.Fatalf("ListSessions failed: %+v err=%v", listed, err)
	}

	// Fallback session/set_model & session/set_mode
	if _, err := client.SetSessionModel(context.Background(), sess.SessionID, "pro", sess); err != nil {
		t.Fatalf("SetSessionModel failed: %v", err)
	}
	if _, err := client.SetSessionModel(context.Background(), sess.SessionID, "unsupported", sess); err == nil {
		t.Fatalf("expected RPCError on unsupported model")
	}

	// Normal Prompt (with JSON-RPC batch notification response!)
	var chunks []string
	res, _, err := client.Prompt(context.Background(), sess.SessionID, []ContentBlock{{Type: "text", Text: "hi"}}, map[string]any{"thoughtLevel": "high"}, func(u ParsedUpdate) {
		if u.Kind == "agent_message_chunk" {
			chunks = append(chunks, u.Text)
		}
	})
	if err != nil || res.StopReason != "end_turn" || strings.Join(chunks, "") != "Hello from unit agent" {
		t.Fatalf("unexpected Prompt result: res=%+v chunks=%v err=%v", res, chunks, err)
	}

	// Prompt Cancellation via context timeout
	cancelCtx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	_, cleanCancel, err := client.Prompt(cancelCtx, sess.SessionID, []ContentBlock{{Type: "text", Text: "SLOW_CANCEL_ME"}}, nil, nil)
	if err == nil || !cleanCancel {
		t.Fatalf("expected clean cancellation ack, got clean=%v err=%v", cleanCancel, err)
	}

	// CloseSession
	if err := client.CloseSession(context.Background(), sess.SessionID); err != nil {
		t.Fatalf("CloseSession failed: %v", err)
	}

	if err := client.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}
