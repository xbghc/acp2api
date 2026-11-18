package server_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"acp2api/internal/acp"
	"acp2api/internal/config"
	"acp2api/internal/pool"
	"acp2api/internal/server"
)

// TestHelperMockACPServer runs as an isolated stdio ACP v1 agent subprocess when GO_WANT_MOCK_ACP=1.
func TestHelperMockACPServer(t *testing.T) {
	if os.Getenv("GO_WANT_MOCK_ACP") != "1" {
		return
	}
	runMockACPStdioAgent()
	os.Exit(0)
}

func runMockACPStdioAgent() {
	var (
		writeMu  sync.Mutex
		sessSeq  atomic.Int64
		reqSeq   atomic.Int64
		sessions sync.Map // sessionId -> *mockSessState
		pending  sync.Map // reqId -> chan json.RawMessage
	)

	type mockSessState struct {
		mu        sync.Mutex
		model     string
		turnCount int
		memory    string
	}

	writeLine := func(v any) {
		b, _ := json.Marshal(v)
		b = append(b, '\n')
		writeMu.Lock()
		_, _ = os.Stdout.Write(b)
		writeMu.Unlock()
	}

	callClient := func(method string, params any) (json.RawMessage, error) {
		id := fmt.Sprintf("srv_%d", reqSeq.Add(1))
		ch := make(chan json.RawMessage, 1)
		pending.Store(id, ch)
		defer pending.Delete(id)

		writeLine(map[string]any{
			"jsonrpc": "2.0",
			"id":      id,
			"method":  method,
			"params":  params,
		})
		select {
		case res := <-ch:
			return res, nil
		case <-time.After(5 * time.Second):
			return nil, fmt.Errorf("timeout waiting for client callback %s", method)
		}
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		line, err := reader.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			var env struct {
				JSONRPC string          `json:"jsonrpc"`
				ID      json.RawMessage `json:"id"`
				Method  string          `json:"method"`
				Params  json.RawMessage `json:"params"`
				Result  json.RawMessage `json:"result"`
			}
			if json.Unmarshal(line, &env) == nil {
				hasID := len(env.ID) > 0 && string(bytes.TrimSpace(env.ID)) != "null"
				if env.Method == "" && hasID {
					// Response to reverse request
					idStr := strings.Trim(strings.TrimSpace(string(env.ID)), `"`)
					if v, ok := pending.Load(idStr); ok {
						v.(chan json.RawMessage) <- env.Result
					}
					continue
				}

				// Handle requests in goroutine so reverse requests can receive responses on stdout/stdin.
				go func(id json.RawMessage, method string, params json.RawMessage) {
					switch method {
					case "initialize":
						writeLine(map[string]any{
							"jsonrpc": "2.0",
							"id":      id,
							"result": map[string]any{
								"protocolVersion": 1,
								"agentInfo": map[string]any{
									"name":    "mock-acp-agent",
									"version": "1.0.0",
								},
								"agentCapabilities": map[string]any{
									"loadSession": false,
									"promptCapabilities": map[string]any{
										"image": true,
									},
								},
							},
						})

					case "session/new":
						sid := fmt.Sprintf("sess_%d_%d", os.Getpid(), sessSeq.Add(1))
						sessions.Store(sid, &mockSessState{model: "acp-fast-v1"})
						writeLine(map[string]any{
							"jsonrpc": "2.0",
							"id":      id,
							"result": map[string]any{
								"sessionId": sid,
								"models": map[string]any{
									"currentModelId": "acp-fast-v1",
									"availableModels": []map[string]any{
										{"modelId": "acp-fast-v1", "name": "ACP Fast V1"},
										{"modelId": "acp-pro-v1", "name": "ACP Pro V1"},
									},
								},
								"configOptions": []map[string]any{
									{
										"id":           "model",
										"name":         "Model",
										"category":     "model",
										"currentValue": "acp-fast-v1",
										"options": []map[string]any{
											{"value": "acp-fast-v1", "name": "ACP Fast V1"},
											{"value": "acp-pro-v1", "name": "ACP Pro V1"},
										},
									},
								},
							},
						})

					case "session/set_config_option":
						var p acp.SetConfigOptionParams
						_ = json.Unmarshal(params, &p)
						if st, ok := sessions.Load(p.SessionID); ok {
							ms := st.(*mockSessState)
							ms.mu.Lock()
							ms.model = fmt.Sprint(p.Value)
							ms.mu.Unlock()
						}
						writeLine(map[string]any{
							"jsonrpc": "2.0",
							"id":      id,
							"result":  map[string]any{"configOptions": []any{}},
						})

					case "session/prompt":
						var p acp.SessionPromptParams
						_ = json.Unmarshal(params, &p)
						var promptText strings.Builder
						for _, b := range p.Prompt {
							if b.Type == "text" {
								promptText.WriteString(b.Text)
							}
						}
						txt := promptText.String()

						if strings.Contains(txt, "TRIGGER_CRASH") {
							os.Exit(42)
						}

						var turnNum int
						var remembered string
						var curModel string
						if st, ok := sessions.Load(p.SessionID); ok {
							ms := st.(*mockSessState)
							ms.mu.Lock()
							ms.turnCount++
							turnNum = ms.turnCount
							curModel = ms.model
							if idx := strings.Index(txt, "REMEMBER:"); idx >= 0 {
								ms.memory = strings.TrimSpace(txt[idx+len("REMEMBER:"):])
							}
							remembered = ms.memory
							ms.mu.Unlock()
						}

						emitChunk := func(kind, chunk string) {
							writeLine(map[string]any{
								"jsonrpc": "2.0",
								"method":  "session/update",
								"params": map[string]any{
									"sessionId": p.SessionID,
									"update": map[string]any{
										"sessionUpdate": kind,
										"content": map[string]any{
											"type": "text",
											"text": chunk,
										},
									},
								},
							})
						}

						// Test reverse callbacks when requested!
						if strings.Contains(txt, "TEST_HOST_CALLBACKS") {
							permRes, err := callClient("session/request_permission", map[string]any{
								"sessionId": p.SessionID,
								"title":     "Allow reading test.txt",
								"toolCall":  map[string]any{"kind": "read"},
								"options": []map[string]any{
									{"optionId": "opt_allow", "name": "Allow", "kind": "allow_once"},
									{"optionId": "opt_deny", "name": "Deny", "kind": "reject_once"},
								},
							})
							if err != nil {
								emitChunk("agent_message_chunk", "ERR_PERM:"+err.Error())
							} else {
								var pr acp.RequestPermissionResult
								_ = json.Unmarshal(permRes, &pr)
								emitChunk("agent_message_chunk", "PERM="+pr.Outcome.OptionID+";")
							}

							_, _ = callClient("fs/write_text_file", map[string]any{
								"sessionId": p.SessionID,
								"path":      "hello_callback.txt",
								"content":   "line1\nline2_secret\nline3",
							})
							readRes, _ := callClient("fs/read_text_file", map[string]any{
								"sessionId": p.SessionID,
								"path":      "hello_callback.txt",
								"line":      2,
								"limit":     1,
							})
							var rr acp.FSReadTextFileResult
							_ = json.Unmarshal(readRes, &rr)
							emitChunk("agent_message_chunk", "FS="+rr.Content+";")

							termCreateRes, _ := callClient("terminal/create", map[string]any{
								"sessionId": p.SessionID,
								"command":   "sh",
								"args":      []string{"-c", "printf TERM_OK"},
							})
							var tcr acp.TerminalCreateResult
							_ = json.Unmarshal(termCreateRes, &tcr)
							_, _ = callClient("terminal/wait_for_exit", map[string]any{
								"sessionId":  p.SessionID,
								"terminalId": tcr.TerminalID,
							})
							termOutRes, _ := callClient("terminal/output", map[string]any{
								"sessionId":  p.SessionID,
								"terminalId": tcr.TerminalID,
							})
							var tor acp.TerminalOutputResult
							_ = json.Unmarshal(termOutRes, &tor)
							_, _ = callClient("terminal/release", map[string]any{
								"sessionId":  p.SessionID,
								"terminalId": tcr.TerminalID,
							})
							emitChunk("agent_message_chunk", "TERM="+tor.Output)
						} else if strings.Contains(txt, "INLINE_THINK_TEST") {
							// Simulate non-standard ACP server that streams <think>...</think> inside agent_message_chunk without agent_thought_chunk!
							emitChunk("agent_message_chunk", "<think>inline reasoning ")
							emitChunk("agent_message_chunk", "step 2</think>\nClean visible answer")
						} else if strings.Contains(txt, "CALL_WEATHER_TOOL") && !strings.Contains(txt, "<tool_response tool_call_id=") {
							emitChunk("agent_thought_chunk", "Need to call get_weather tool.")
							emitChunk("agent_message_chunk", "<tool_")
							emitChunk("agent_message_chunk", "call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Beijing\"}}\n</tool_call>")
						} else {
							emitChunk("agent_thought_chunk", "Thinking step 1... ")
							emitChunk("agent_thought_chunk", "done.")
							reply := fmt.Sprintf("Turn=%d Model=%s Memory=%s Echo=%s", turnNum, curModel, remembered, strings.TrimSpace(txt))
							mid := len(reply) / 2
							emitChunk("agent_message_chunk", reply[:mid])
							emitChunk("agent_message_chunk", reply[mid:])
						}

						writeLine(map[string]any{
							"jsonrpc": "2.0",
							"id":      id,
							"result": map[string]any{
								"stopReason": "end_turn",
							},
						})
					}
				}(append(json.RawMessage(nil), env.ID...), env.Method, append(json.RawMessage(nil), env.Params...))
			}
		}
		if err != nil {
			return
		}
	}
}

func newTestGateway(t *testing.T, modify func(*config.Config)) (*httptest.Server, *pool.Pool) {
	t.Helper()
	cfg := config.Default()
	cfg.Server.APIKeys = []string{"test-secret-key"}
	cfg.ACP.Command = os.Args[0]
	cfg.ACP.Args = []string{"-test.run=TestHelperMockACPServer", "--"}
	cfg.ACP.Env = map[string]string{"GO_WANT_MOCK_ACP": "1"}
	cfg.ACP.Cwd = t.TempDir()
	cfg.ACP.Capabilities.EnableFSRead = true
	cfg.ACP.Capabilities.EnableFSWrite = true
	cfg.ACP.Capabilities.EnableTerminal = true
	cfg.Pool.ProcessCount = 1
	cfg.Pool.IdlePoolSize = 3
	cfg.Pool.ReuseConversations = true
	if modify != nil {
		modify(&cfg)
	}
	if err := cfg.NormalizeAndValidate(); err != nil {
		t.Fatalf("NormalizeAndValidate: %v", err)
	}

	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	p := pool.New(cfg, logger)
	if err := p.Start(context.Background()); err != nil {
		t.Fatalf("pool.Start: %v", err)
	}
	srv := server.New(cfg, p, logger)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(func() {
		ts.Close()
		_ = p.Close()
	})
	return ts, p
}

func doJSON(t *testing.T, method, url, apiKey string, body any) (*http.Response, []byte) {
	t.Helper()
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			t.Fatalf("marshal request: %v", err)
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, url, r)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("http do: %v", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	return resp, raw
}

func TestIdleSessionWarmupAndMultiTurnReuse_OpenAI(t *testing.T) {
	ts, p := newTestGateway(t, nil)

	// 1. Verify 3 idle sessions are already pre-warmed BEFORE any API request arrives!
	st := p.Stats()
	if st.ReadyIdleSessions != 3 {
		t.Fatalf("expected 3 pre-warmed idle sessions at startup, got %+v", st)
	}

	// 2. Verify /v1/models includes both default and ACP-discovered models.
	resp, raw := doJSON(t, "GET", ts.URL+"/v1/models", "test-secret-key", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/models status=%d body=%s", resp.StatusCode, string(raw))
	}
	if !strings.Contains(string(raw), "acp-pro-v1") {
		t.Fatalf("expected discovered model acp-pro-v1 in /v1/models, got: %s", string(raw))
	}

	// 3. Turn 1: should pop from pre-warmed idle pool (0 cold creations!) and switch model to acp-pro-v1.
	resp1, raw1 := doJSON(t, "POST", ts.URL+"/v1/chat/completions", "test-secret-key", map[string]any{
		"model": "acp-pro-v1",
		"messages": []map[string]any{
			{"role": "user", "content": "REMEMBER:alpha-777"},
		},
	})
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("Turn 1 failed: %d %s", resp1.StatusCode, string(raw1))
	}
	sessID1 := resp1.Header.Get("X-ACP-Session-ID")
	if resp1.Header.Get("X-ACP-Session-Reused") != "false" {
		t.Fatalf("expected Turn 1 X-ACP-Session-Reused=false")
	}

	var chatResp1 struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw1, &chatResp1); err != nil {
		t.Fatalf("unmarshal Turn 1: %v", err)
	}
	ans1 := chatResp1.Choices[0].Message.Content
	if !strings.Contains(ans1, "Turn=1 Model=acp-pro-v1 Memory=alpha-777") {
		t.Fatalf("unexpected Turn 1 content: %q", ans1)
	}
	if chatResp1.Choices[0].Message.ReasoningContent == "" {
		t.Fatalf("expected reasoning_content from agent_thought_chunk")
	}

	// Wait briefly for background replenisher to restore ReadyIdleSessions back to 3.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if p.Stats().ReadyIdleSessions == 3 {
			break
		}
		time.Sleep(30 * time.Millisecond)
	}
	stAfter1 := p.Stats()
	if stAfter1.IdlePoolHits != 1 || stAfter1.ColdCreations != 0 || stAfter1.ReadyIdleSessions != 3 {
		t.Fatalf("unexpected pool stats after Turn 1: %+v", stAfter1)
	}

	// 4. Turn 2: pass [U1, A1, U2] (without reasoning_content, just like real OpenAI clients!)
	// Verify it reuses exact sessID1 and sends ONLY U2 ("WHAT_WAS_IT?") while sessID1 remembers alpha-777!
	resp2, raw2 := doJSON(t, "POST", ts.URL+"/v1/chat/completions", "test-secret-key", map[string]any{
		"model": "acp-pro-v1",
		"messages": []map[string]any{
			{"role": "user", "content": "REMEMBER:alpha-777"},
			{"role": "assistant", "content": ans1},
			{"role": "user", "content": "WHAT_WAS_IT?"},
		},
	})
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("Turn 2 failed: %d %s", resp2.StatusCode, string(raw2))
	}
	if resp2.Header.Get("X-ACP-Session-Reused") != "true" {
		t.Fatalf("expected Turn 2 X-ACP-Session-Reused=true")
	}
	if resp2.Header.Get("X-ACP-Session-ID") != sessID1 {
		t.Fatalf("expected reused session ID %q, got %q", sessID1, resp2.Header.Get("X-ACP-Session-ID"))
	}

	var chatResp2 struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	_ = json.Unmarshal(raw2, &chatResp2)
	ans2 := chatResp2.Choices[0].Message.Content
	// Verify Turn=2, Memory=alpha-777, and Echo=WHAT_WAS_IT? (proving only incremental delta was sent!).
	if !strings.Contains(ans2, "Turn=2 Model=acp-pro-v1 Memory=alpha-777 Echo=WHAT_WAS_IT?") {
		t.Fatalf("unexpected Turn 2 content: %q", ans2)
	}
}

func TestOpenAISSEStreamingAndToolCalls(t *testing.T) {
	ts, _ := newTestGateway(t, nil)

	resp, raw := doJSON(t, "POST", ts.URL+"/v1/chat/completions", "test-secret-key", map[string]any{
		"model":  "acp-fast-v1",
		"stream": true,
		"stream_options": map[string]any{
			"include_usage": true,
		},
		"tools": []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "get_weather",
					"description": "Get city weather",
					"parameters":  map[string]any{"type": "object"},
				},
			},
		},
		"messages": []map[string]any{
			{"role": "user", "content": "CALL_WEATHER_TOOL"},
		},
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("stream status=%d body=%s", resp.StatusCode, string(raw))
	}
	sse := string(raw)
	if !strings.Contains(sse, `"finish_reason":"tool_calls"`) {
		t.Fatalf("expected finish_reason=tool_calls in SSE stream:\n%s", sse)
	}
	if !strings.Contains(sse, `"get_weather"`) || !strings.Contains(sse, `\"city\":\"Beijing\"`) {
		t.Fatalf("expected get_weather tool_call delta in SSE stream:\n%s", sse)
	}
	if !strings.Contains(sse, `"total_tokens"`) || !strings.Contains(sse, "data: [DONE]") {
		t.Fatalf("expected usage chunk and [DONE] in SSE stream:\n%s", sse)
	}
}

func TestAnthropicMessagesSyncStreamAndToolLoop(t *testing.T) {
	ts, p := newTestGateway(t, nil)

	// 1. Test /v1/messages/count_tokens with x-api-key header
	reqCount, _ := http.NewRequest("POST", ts.URL+"/v1/messages/count_tokens", strings.NewReader(`{
		"model": "acp-fast-v1",
		"system": [{"type":"text","text":"You are helpful."}],
		"messages": [{"role":"user","content":"Hello world"}]
	}`))
	reqCount.Header.Set("Content-Type", "application/json")
	reqCount.Header.Set("x-api-key", "test-secret-key")
	resCount, err := http.DefaultClient.Do(reqCount)
	if err != nil {
		t.Fatalf("count_tokens request failed: %v", err)
	}
	defer resCount.Body.Close()
	if resCount.StatusCode != http.StatusOK {
		t.Fatalf("count_tokens status=%d", resCount.StatusCode)
	}

	// 2. Test Streaming Anthropic /v1/messages with Tool Use + Thinking
	reqBody, _ := json.Marshal(map[string]any{
		"model":      "acp-fast-v1",
		"max_tokens": 1024,
		"stream":     true,
		"tools": []map[string]any{
			{
				"name":         "get_weather",
				"description":  "Query weather",
				"input_schema": map[string]any{"type": "object"},
			},
		},
		"messages": []map[string]any{
			{"role": "user", "content": "CALL_WEATHER_TOOL"},
		},
	})
	reqStream, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(reqBody))
	reqStream.Header.Set("Content-Type", "application/json")
	reqStream.Header.Set("x-api-key", "test-secret-key")
	reqStream.Header.Set("anthropic-version", "2023-06-01")
	resStream, err := http.DefaultClient.Do(reqStream)
	if err != nil {
		t.Fatalf("anthropic stream failed: %v", err)
	}
	rawSSEBytes, _ := io.ReadAll(resStream.Body)
	resStream.Body.Close()
	sse := string(rawSSEBytes)

	for _, expected := range []string{
		"event: message_start",
		`"thinking_delta"`,
		`"signature_delta"`,
		`"type":"tool_use"`,
		`"input_json_delta"`,
		`"stop_reason":"tool_use"`,
		"event: message_stop",
	} {
		if !strings.Contains(sse, expected) {
			t.Fatalf("missing %q in Anthropic SSE stream:\n%s", expected, sse)
		}
	}

	// Extract tool_use id from SSE to test Turn 2 tool_result session reuse!
	var toolUseID string
	for _, line := range strings.Split(sse, "\n") {
		if strings.HasPrefix(line, "data: ") && strings.Contains(line, `"type":"tool_use"`) {
			var ev struct {
				ContentBlock struct {
					ID string `json:"id"`
				} `json:"content_block"`
			}
			_ = json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev)
			toolUseID = ev.ContentBlock.ID
		}
	}
	if toolUseID == "" {
		t.Fatalf("failed extracting toolUseID from SSE:\n%s", sse)
	}

	prevReuseHits := p.Stats().ConversationHits

	// 3. Turn 2: send assistant tool_use + user tool_result -> verify session reuse!
	resp2, raw2 := doJSON(t, "POST", ts.URL+"/v1/messages", "test-secret-key", map[string]any{
		"model":      "acp-fast-v1",
		"max_tokens": 1024,
		"tools": []map[string]any{
			{
				"name":         "get_weather",
				"description":  "Query weather",
				"input_schema": map[string]any{"type": "object"},
			},
		},
		"messages": []map[string]any{
			{"role": "user", "content": "CALL_WEATHER_TOOL"},
			{
				"role": "assistant",
				"content": []map[string]any{
					{"type": "tool_use", "id": toolUseID, "name": "get_weather", "input": map[string]any{"city": "Beijing"}},
				},
			},
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "tool_result", "tool_use_id": toolUseID, "content": "Sunny 24C"},
				},
			},
		},
	})
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("Anthropic Turn 2 failed: %d %s", resp2.StatusCode, string(raw2))
	}
	if resp2.Header.Get("X-ACP-Session-Reused") != "true" {
		t.Fatalf("expected Anthropic tool_result Turn 2 to reuse existing ACP session! body=%s", string(raw2))
	}
	if p.Stats().ConversationHits <= prevReuseHits {
		t.Fatalf("expected ConversationHits to increment")
	}
	if !strings.Contains(string(raw2), "Turn=2") || !strings.Contains(string(raw2), "Sunny 24C") {
		t.Fatalf("expected Turn=2 with Sunny 24C in response: %s", string(raw2))
	}
}

func TestHostReverseCallbacksAndWorkerCrashRecovery(t *testing.T) {
	var workDir string
	ts, p := newTestGateway(t, func(c *config.Config) {
		workDir = c.ACP.Cwd
	})

	// 1. Verify permission callback + fs/write_text_file + fs/read_text_file + terminal/* reverse calls!
	resp, raw := doJSON(t, "POST", ts.URL+"/v1/responses", "test-secret-key", map[string]any{
		"model": "acp-fast-v1",
		"input": "TEST_HOST_CALLBACKS",
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/responses failed: %d %s", resp.StatusCode, string(raw))
	}
	bodyStr := string(raw)
	if !strings.Contains(bodyStr, "PERM=opt_allow;FS=line2_secret;TERM=TERM_OK") {
		t.Fatalf("unexpected host callback bridge output: %s", bodyStr)
	}
	if _, err := os.Stat(filepath.Join(workDir, "hello_callback.txt")); err != nil {
		t.Fatalf("expected hello_callback.txt written inside Cwd: %v", err)
	}

	// 2. Verify unauthorized requests are rejected properly.
	unauthResp, _ := doJSON(t, "POST", ts.URL+"/v1/messages", "wrong-key", map[string]any{
		"model": "acp-fast-v1", "max_tokens": 10, "messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if unauthResp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401 unauthorized, got %d", unauthResp.StatusCode)
	}

	// 3. Trigger simulated worker crash and verify automatic worker respawn & idle session recovery!
	crashResp, _ := doJSON(t, "POST", ts.URL+"/v1/chat/completions", "test-secret-key", map[string]any{
		"model":    "acp-fast-v1",
		"messages": []map[string]any{{"role": "user", "content": "TRIGGER_CRASH"}},
	})
	if crashResp.StatusCode == http.StatusOK {
		t.Fatalf("expected crash turn to return non-200 error")
	}

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		st := p.Stats()
		if st.WorkerRestarts >= 1 && st.HealthyWorkers == 1 && st.ReadyIdleSessions == 3 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	stRecovered := p.Stats()
	if stRecovered.WorkerRestarts < 1 || stRecovered.HealthyWorkers != 1 || stRecovered.ReadyIdleSessions != 3 {
		t.Fatalf("expected worker to auto-restart and replenish 3 idle sessions, got %+v", stRecovered)
	}

	// Confirm new requests succeed right after crash recovery!
	afterResp, afterRaw := doJSON(t, "POST", ts.URL+"/v1/chat/completions", "test-secret-key", map[string]any{
		"model":    "acp-fast-v1",
		"messages": []map[string]any{{"role": "user", "content": "Alive again!"}},
	})
	if afterResp.StatusCode != http.StatusOK || !strings.Contains(string(afterRaw), "Alive again!") {
		t.Fatalf("expected request after crash recovery to succeed, status=%d body=%s", afterResp.StatusCode, string(afterRaw))
	}
}

func TestConcurrentBurstAndConversationForking(t *testing.T) {
	ts, p := newTestGateway(t, func(c *config.Config) {
		c.Pool.ProcessCount = 2
		c.Pool.IdlePoolSize = 3
	})

	// 1. Burst 12 concurrent requests across 2 worker processes (exceeding IdlePoolSize=3).
	const concurrency = 12
	var wg sync.WaitGroup
	errs := make(chan error, concurrency)

	for i := 0; i < concurrency; i++ {
		idx := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			marker := fmt.Sprintf("BURST_ITEM_%d", idx)
			resp, raw := doJSON(t, "POST", ts.URL+"/v1/chat/completions", "test-secret-key", map[string]any{
				"model":    "acp-fast-v1",
				"messages": []map[string]any{{"role": "user", "content": marker}},
			})
			if resp.StatusCode != http.StatusOK {
				errs <- fmt.Errorf("req %d status=%d body=%s", idx, resp.StatusCode, string(raw))
				return
			}
			if !strings.Contains(string(raw), marker) {
				errs <- fmt.Errorf("req %d cross-talk or missing marker %q: %s", idx, marker, string(raw))
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}

	// 2. Test conversation branching (two sequential branches from the exact same Turn 1 history):
	// First branch claims the cached session (Reused=true), second branch cleanly falls back to a fresh session (Reused=false)!
	r1, raw1 := doJSON(t, "POST", ts.URL+"/v1/messages", "test-secret-key", map[string]any{
		"model":      "acp-fast-v1",
		"max_tokens": 256,
		"messages":   []map[string]any{{"role": "user", "content": "ROOT_TURN"}},
	})
	if r1.StatusCode != http.StatusOK {
		t.Fatalf("root turn failed: %d", r1.StatusCode)
	}
	var m1 struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	_ = json.Unmarshal(raw1, &m1)
	var rootAnswer string
	for _, b := range m1.Content {
		if b.Type == "text" {
			rootAnswer = b.Text
		}
	}

	branchA, _ := doJSON(t, "POST", ts.URL+"/v1/messages", "test-secret-key", map[string]any{
		"model":      "acp-fast-v1",
		"max_tokens": 256,
		"messages": []map[string]any{
			{"role": "user", "content": "ROOT_TURN"},
			{"role": "assistant", "content": rootAnswer},
			{"role": "user", "content": "BRANCH_A"},
		},
	})
	branchB, rawB := doJSON(t, "POST", ts.URL+"/v1/messages", "test-secret-key", map[string]any{
		"model":      "acp-fast-v1",
		"max_tokens": 256,
		"messages": []map[string]any{
			{"role": "user", "content": "ROOT_TURN"},
			{"role": "assistant", "content": rootAnswer},
			{"role": "user", "content": "BRANCH_B"},
		},
	})
	if branchA.Header.Get("X-ACP-Session-Reused") != "true" {
		t.Fatalf("expected Branch A to reuse cached session")
	}
	if branchB.Header.Get("X-ACP-Session-Reused") != "false" {
		t.Fatalf("expected forked Branch B to cleanly acquire fresh session from idle pool")
	}
	if !strings.Contains(string(rawB), "ROOT_TURN") || !strings.Contains(string(rawB), "BRANCH_B") {
		t.Fatalf("expected Branch B to replay root history and answer BRANCH_B: %s", string(rawB))
	}

	// Verify /healthz and /v1/pool/stats endpoints.
	healthResp, _ := doJSON(t, "GET", ts.URL+"/healthz", "", nil)
	if healthResp.StatusCode != http.StatusOK {
		t.Fatalf("/healthz returned %d", healthResp.StatusCode)
	}
	statsResp, _ := doJSON(t, "GET", ts.URL+"/v1/pool/stats", "test-secret-key", nil)
	if statsResp.StatusCode != http.StatusOK || p.Stats().TotalRequests < 15 {
		t.Fatalf("unexpected pool stats: %+v", p.Stats())
	}
}

func TestThinkingFallbackInlineThinkExtractionAndRichMedia(t *testing.T) {
	// Mock server has NO thought_level configOption -> verifies fallback_meta + inline <think> extraction!
	ts, _ := newTestGateway(t, nil)

	// 1. OpenAI request with model:high suffix + input_audio + file + INLINE_THINK_TEST
	resp1, raw1 := doJSON(t, "POST", ts.URL+"/v1/chat/completions", "test-secret-key", map[string]any{
		"model": "acp-fast-v1:high",
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": "INLINE_THINK_TEST"},
					{"type": "input_audio", "input_audio": map[string]any{"data": "UklGRg==", "format": "mp3"}},
					{"type": "file", "file": map[string]any{"filename": "doc.txt", "file_data": "data:text/plain;base64,SGVsbG8="}},
				},
			},
		},
	})
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("INLINE_THINK_TEST failed: %d %s", resp1.StatusCode, string(raw1))
	}
	if gotStrat := resp1.Header.Get("X-ACP-Thinking-Strategy"); gotStrat != "fallback_meta" {
		t.Fatalf("expected X-ACP-Thinking-Strategy=fallback_meta, got %q", gotStrat)
	}
	var chat1 struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	_ = json.Unmarshal(raw1, &chat1)
	if chat1.Choices[0].Message.ReasoningContent != "inline reasoning step 2" || chat1.Choices[0].Message.Content != "Clean visible answer" {
		t.Fatalf("unexpected inline think split: %+v", chat1.Choices[0].Message)
	}

	// 2. Anthropic request with thinking={type:"disabled"} -> suppresses thinking output even if agent emits thoughts!
	respDisabled, rawDisabled := doJSON(t, "POST", ts.URL+"/v1/messages", "test-secret-key", map[string]any{
		"model":      "acp-fast-v1",
		"max_tokens": 256,
		"thinking":   map[string]any{"type": "disabled"},
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": "Hello with doc"},
					{"type": "document", "title": "readme.md", "source": map[string]any{"type": "text", "media_type": "text/markdown", "data": "# Doc"}},
				},
			},
		},
	})
	if respDisabled.StatusCode != http.StatusOK || strings.Contains(string(rawDisabled), `"type":"thinking"`) {
		t.Fatalf("expected thinking block suppressed when thinking.type=disabled: %d %s", respDisabled.StatusCode, string(rawDisabled))
	}

	// 3. StrictError thinking fallback policy returns HTTP 400 when ACP agent has no thinking level selector!
	tsStrict, _ := newTestGateway(t, func(c *config.Config) {
		c.ACP.Thinking.FallbackStrategy = config.ThinkingFallbackStrictError
	})
	respStrict, rawStrict := doJSON(t, "POST", tsStrict.URL+"/v1/messages", "test-secret-key", map[string]any{
		"model":         "acp-fast-v1",
		"max_tokens":    256,
		"thinking":      map[string]any{"type": "enabled", "budget_tokens": 16000},
		"output_config": map[string]any{"effort": "high"},
		"messages":      []map[string]any{{"role": "user", "content": "hi"}},
	})
	if respStrict.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 Bad Request under strict_error policy, got %d %s", respStrict.StatusCode, string(rawStrict))
	}
}
