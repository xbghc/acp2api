package pool

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
	"sync/atomic"
	"testing"
	"time"

	"acp2api/internal/acp"
	"acp2api/internal/bridge"
	"acp2api/internal/config"
)

// TestHelperPoolMockACP runs an ACP stdio subprocess tailored for testing Pool mechanics.
func TestHelperPoolMockACP(t *testing.T) {
	mode := os.Getenv("GO_POOL_TEST_ACP")
	if mode == "" {
		return
	}
	if mode == "broken_boot" {
		fmt.Fprintln(os.Stderr, "boot failure")
		os.Exit(1)
	}

	var (
		writeMu sync.Mutex
		seq     atomic.Int64
	)
	writeLine := func(v any) {
		b, _ := json.Marshal(v)
		b = append(b, '\n')
		writeMu.Lock()
		_, _ = os.Stdout.Write(b)
		writeMu.Unlock()
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		line, err := reader.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			var env struct {
				ID     json.RawMessage `json:"id"`
				Method string          `json:"method"`
				Params json.RawMessage `json:"params"`
			}
			if json.Unmarshal(line, &env) == nil && env.Method != "" {
				go func(id json.RawMessage, method string, params json.RawMessage) {
					switch method {
					case "initialize":
						writeLine(map[string]any{
							"jsonrpc": "2.0",
							"id":      id,
							"result": map[string]any{
								"protocolVersion": 1,
								"agentInfo":       map[string]any{"name": "pool-mock", "version": "1.0"},
							},
						})
					case "session/new":
						sid := fmt.Sprintf("psess_%d_%d", os.Getpid(), seq.Add(1))
						writeLine(map[string]any{
							"jsonrpc": "2.0",
							"id":      id,
							"result": map[string]any{
								"sessionId": sid,
								"configOptions": []map[string]any{
									{
										"id":           "model",
										"category":     "model",
										"currentValue": "m-base",
										"options": []map[string]any{
											{"value": "m-base", "name": "Base"},
											{"value": "m-ultra", "name": "Ultra"},
										},
									},
								},
							},
						})
					case "session/set_config_option":
						writeLine(map[string]any{"jsonrpc": "2.0", "id": id, "result": map[string]any{}})
					case "session/prompt":
						var p acp.SessionPromptParams
						_ = json.Unmarshal(params, &p)
						txt := ""
						if len(p.Prompt) > 0 {
							txt = p.Prompt[0].Text
						}

						if strings.Contains(txt, "SLOW_HOLD") {
							time.Sleep(180 * time.Millisecond)
						}

						chunks := []string{"Hello ", "world ", "STOP_HERE ", "extra trailing"}
						if !strings.Contains(txt, "STREAM_STOP") {
							chunks = []string{"Reply:" + strings.TrimSpace(txt)}
						}

						for _, ch := range chunks {
							writeLine(map[string]any{
								"jsonrpc": "2.0",
								"method":  "session/update",
								"params": map[string]any{
									"sessionId": p.SessionID,
									"update": map[string]any{
										"sessionUpdate": "agent_message_chunk",
										"content":       map[string]any{"type": "text", "text": ch},
									},
								},
							})
							time.Sleep(5 * time.Millisecond)
						}

						// Emit usage_update to test exact agent token accounting!
						writeLine(map[string]any{
							"jsonrpc": "2.0",
							"method":  "session/update",
							"params": map[string]any{
								"sessionId": p.SessionID,
								"update": map[string]any{
									"sessionUpdate": "usage_update",
									"usage":         map[string]any{"promptTokens": 42, "outputTokens": 17},
								},
							},
						})

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
			os.Exit(0)
		}
	}
}

func newPoolTestConfig(t *testing.T) config.Config {
	t.Helper()
	cfg := config.Default()
	cfg.ACP.Command = os.Args[0]
	cfg.ACP.Args = []string{"-test.run=TestHelperPoolMockACP", "--"}
	cfg.ACP.Env = map[string]string{"GO_POOL_TEST_ACP": "ok"}
	cfg.ACP.Cwd = t.TempDir()
	cfg.ACP.ModelAliases = map[string]string{"alias-ultra": "m-ultra"}
	cfg.Pool.ProcessCount = 1
	cfg.Pool.IdlePoolSize = 2
	if err := cfg.NormalizeAndValidate(); err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestPoolStartFailureWhenWorkersCannotBoot(t *testing.T) {
	cfg := newPoolTestConfig(t)
	cfg.ACP.Env = map[string]string{"GO_POOL_TEST_ACP": "broken_boot"}
	p := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err := p.Start(context.Background()); err == nil {
		_ = p.Close()
		t.Fatalf("expected Start error when all workers fail")
	}
}

func TestPoolStopSequencesUsageLRUEvictionAndTTL(t *testing.T) {
	cfg := newPoolTestConfig(t)
	cfg.Pool.MaxCachedSessions = 2
	cfg.Pool.IdleSessionTTLSeconds = 1
	cfg.Pool.CachedSessionTTLSeconds = 1

	p := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err := p.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer p.Close()

	// Verify AvailableModels includes default, alias, and ConfigOptions discovered choices.
	models := p.AvailableModels()
	joined := strings.Join(models, ",")
	if !strings.Contains(joined, "alias-ultra") || !strings.Contains(joined, "m-ultra") {
		t.Fatalf("expected discovered & alias models, got: %v", models)
	}

	// 1. Test StopSequences truncation + agent UsageInfo capture.
	stopResp, err := p.ExecuteTurn(context.Background(), &bridge.UnifiedRequest{
		Model:         "alias-ultra",
		StopSequences: []string{"STOP_HERE"},
		Messages: []bridge.UnifiedMessage{
			{Role: "user", Parts: []bridge.ContentPart{{Kind: bridge.PartText, Text: "STREAM_STOP"}}},
		},
	}, nil)
	if err != nil {
		t.Fatalf("ExecuteTurn stop_sequence failed: %v", err)
	}
	if stopResp.StopReason != "stop_sequence" || stopResp.StopSequence != "STOP_HERE" || stopResp.Text != "Hello world " {
		t.Fatalf("unexpected stop_sequence response: %+v", stopResp)
	}

	// 2. Test MaxCachedSessions=2 LRU eviction + SessionHint preference + exact UsageInfo.
	for i := 0; i < 3; i++ {
		r, err := p.ExecuteTurn(context.Background(), &bridge.UnifiedRequest{
			Model:       "m-base",
			SessionHint: fmt.Sprintf("hint-%d", i),
			Messages: []bridge.UnifiedMessage{
				{Role: "user", Parts: []bridge.ContentPart{{Kind: bridge.PartText, Text: fmt.Sprintf("msg-%d", i)}}},
			},
		}, nil)
		if err != nil {
			t.Fatalf("turn %d failed: %v", i, err)
		}
		if r.InputTokens != 42 || r.OutputTokens != 17 {
			t.Fatalf("expected agent-reported tokens (42, 17), got (%d, %d)", r.InputTokens, r.OutputTokens)
		}
	}
	if got := p.Stats().CachedConversations; got != 2 {
		t.Fatalf("expected MaxCachedSessions=2 enforced via LRU eviction, got %d", got)
	}

	// 3. Wait past 1s TTL and invoke evictExpiredSessions -> cached sessions & stale idle sessions expire cleanly!
	time.Sleep(1100 * time.Millisecond)
	p.evictExpiredSessions()
	if got := p.Stats().CachedConversations; got != 0 {
		t.Fatalf("expected expired cached sessions evicted, got %d", got)
	}
}

func TestPoolMaxSessionsPerProcessRotationAndSemaphore(t *testing.T) {
	cfg := newPoolTestConfig(t)
	cfg.Pool.IdlePoolSize = 1
	cfg.Pool.ReuseConversations = false // Stateless mode
	cfg.Pool.MaxSessionsPerProcess = 3  // Rotate worker every 3 created sessions
	cfg.Pool.MaxActiveSessions = 1      // Semaphore = 1 concurrent turn

	p := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err := p.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer p.Close()

	// 1. Test MaxActiveSessions=1 semaphore timeout when a slow turn holds the single slot!
	errCh := make(chan error, 1)
	go func() {
		_, err := p.ExecuteTurn(context.Background(), &bridge.UnifiedRequest{
			Messages: []bridge.UnifiedMessage{{Role: "user", Parts: []bridge.ContentPart{{Kind: bridge.PartText, Text: "SLOW_HOLD"}}}},
		}, nil)
		errCh <- err
	}()
	time.Sleep(30 * time.Millisecond)

	shortCtx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	_, semErr := p.ExecuteTurn(shortCtx, &bridge.UnifiedRequest{
		Messages: []bridge.UnifiedMessage{{Role: "user", Parts: []bridge.ContentPart{{Kind: bridge.PartText, Text: "blocked"}}}},
	}, nil)
	if semErr == nil {
		t.Fatalf("expected context deadline exceeded when MaxActiveSessions=1 is full")
	}
	if firstErr := <-errCh; firstErr != nil {
		t.Fatalf("first turn failed: %v", firstErr)
	}

	// 2. Run 5 sequential stateless turns to trigger MaxSessionsPerProcess=3 worker rotation seamlessly!
	for i := 0; i < 5; i++ {
		res, err := p.ExecuteTurn(context.Background(), &bridge.UnifiedRequest{
			Messages: []bridge.UnifiedMessage{{Role: "user", Parts: []bridge.ContentPart{{Kind: bridge.PartText, Text: fmt.Sprintf("turn-%d", i)}}}},
		}, nil)
		if err != nil || !strings.Contains(res.Text, fmt.Sprintf("turn-%d", i)) {
			t.Fatalf("stateless turn %d failed during worker rotation: res=%+v err=%v", i, res, err)
		}
	}
}
