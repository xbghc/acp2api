package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"acp2api/internal/config"
	"acp2api/internal/pool"
	"acp2api/internal/server"
)

func TestOpenAIExtendedEndpointsMultimodalResponsesStreamAndErrors(t *testing.T) {
	ts, _ := newTestGateway(t, nil)

	// 1. Retrieve single model (/v1/models/{model})
	mResp, mRaw := doJSON(t, "GET", ts.URL+"/v1/models/acp-pro-v1", "test-secret-key", nil)
	if mResp.StatusCode != http.StatusOK || !strings.Contains(string(mRaw), `"id":"acp-pro-v1"`) {
		t.Fatalf("GET /v1/models/acp-pro-v1 failed: %d %s", mResp.StatusCode, string(mRaw))
	}

	// 2. CORS OPTIONS preflight (no auth required)
	optReq, _ := http.NewRequest("OPTIONS", ts.URL+"/v1/chat/completions", nil)
	optResp, err := http.DefaultClient.Do(optReq)
	if err != nil {
		t.Fatalf("OPTIONS failed: %v", err)
	}
	optResp.Body.Close()
	if optResp.StatusCode != http.StatusNoContent || optResp.Header.Get("Access-Control-Allow-Origin") != "*" {
		t.Fatalf("unexpected OPTIONS response: %d", optResp.StatusCode)
	}

	// 3. OpenAI Chat Completions: multimodal content array (data URI + external URL), developer role,
	// non-streaming tool_calls + Turn 2 role:"tool" reuse, forced function tool_choice, and stop array!
	tcResp1, tcRaw1 := doJSON(t, "POST", ts.URL+"/chat/completions", "test-secret-key", map[string]any{
		"model": "acp-fast-v1",
		"tools": []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "get_weather",
					"description": "Weather lookup",
					"parameters":  map[string]any{"type": "object"},
				},
			},
		},
		"tool_choice": map[string]any{
			"type":     "function",
			"function": map[string]any{"name": "get_weather"},
		},
		"stop": []string{"NEVER_STOP"},
		"messages": []map[string]any{
			{"role": "developer", "content": "Follow schema strictly."},
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": "CALL_WEATHER_TOOL"},
					{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,iVBORw0KGgo="}},
					{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/photo.jpg"}},
				},
			},
		},
	})
	if tcResp1.StatusCode != http.StatusOK {
		t.Fatalf("non-streaming tool_calls failed: %d %s", tcResp1.StatusCode, string(tcRaw1))
	}
	var parsedTC1 struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content          *string `json:"content"`
				ReasoningContent string  `json:"reasoning_content"`
				ToolCalls        []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(tcRaw1, &parsedTC1); err != nil {
		t.Fatalf("unmarshal tcRaw1: %v", err)
	}
	if parsedTC1.Choices[0].FinishReason != "tool_calls" || len(parsedTC1.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected finish_reason=tool_calls with 1 call, got: %s", string(tcRaw1))
	}
	callID := parsedTC1.Choices[0].Message.ToolCalls[0].ID

	// Turn 2 with role:"tool" and string stop -> verify session reuse!
	tcResp2, tcRaw2 := doJSON(t, "POST", ts.URL+"/v1/v1/chat/completions", "test-secret-key", map[string]any{
		"model": "acp-fast-v1",
		"tools": []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "get_weather",
					"description": "Weather lookup",
					"parameters":  map[string]any{"type": "object"},
				},
			},
		},
		"stop": "END_MARK",
		"messages": []map[string]any{
			{"role": "developer", "content": "Follow schema strictly."},
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": "CALL_WEATHER_TOOL"},
					{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,iVBORw0KGgo="}},
					{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/photo.jpg"}},
				},
			},
			{
				"role":      "assistant",
				"content":   nil,
				"tool_calls": parsedTC1.Choices[0].Message.ToolCalls,
			},
			{
				"role":         "tool",
				"tool_call_id": callID,
				"name":         "get_weather",
				"content":      "Rainy 18C",
			},
		},
	})
	if tcResp2.StatusCode != http.StatusOK || tcResp2.Header.Get("X-ACP-Session-Reused") != "true" {
		t.Fatalf("expected OpenAI tool Turn 2 to reuse session! status=%d reused=%s body=%s",
			tcResp2.StatusCode, tcResp2.Header.Get("X-ACP-Session-Reused"), string(tcRaw2))
	}

	// 4. OpenAI /v1/responses SSE streaming with structured array input (message + function_call + function_call_output + tools)!
	respStream, rawRespStream := doJSON(t, "POST", ts.URL+"/v1/responses", "test-secret-key", map[string]any{
		"model":        "acp-fast-v1",
		"instructions": "Be helpful.",
		"stream":       true,
		"tool_choice":  "auto",
		"tools": []map[string]any{
			{"type": "function", "name": "get_weather", "description": "Weather"},
		},
		"input": []map[string]any{
			{"role": "system", "content": "Extra instruction"},
			{"role": "user", "content": "CALL_WEATHER_TOOL"},
		},
	})
	if respStream.StatusCode != http.StatusOK {
		t.Fatalf("/v1/responses stream failed: %d %s", respStream.StatusCode, string(rawRespStream))
	}
	rSSE := string(rawRespStream)
	for _, want := range []string{
		"event: response.created",
		"event: response.reasoning_summary_text.delta",
		"event: response.function_call_arguments.done",
		"event: response.completed",
	} {
		if !strings.Contains(rSSE, want) {
			t.Fatalf("missing %q in /v1/responses SSE:\n%s", want, rSSE)
		}
	}

	// Follow-up /v1/responses non-streaming with function_call + function_call_output items
	respSync2, rawSync2 := doJSON(t, "POST", ts.URL+"/responses", "test-secret-key", map[string]any{
		"model": "acp-fast-v1",
		"input": []map[string]any{
			{"role": "user", "content": "Hi"},
			{"type": "function_call", "call_id": "c_9", "name": "ping", "arguments": "{}"},
			{"type": "function_call_output", "call_id": "c_9", "output": "pong_result"},
		},
	})
	if respSync2.StatusCode != http.StatusOK || !strings.Contains(string(rawSync2), "pong_result") {
		t.Fatalf("unexpected /responses array input output: %d %s", respSync2.StatusCode, string(rawSync2))
	}

	// 5. Validation & 404 error responses
	badJSONReq, _ := http.NewRequest("POST", ts.URL+"/v1/chat/completions", strings.NewReader("{bad"))
	badJSONReq.Header.Set("Authorization", "Bearer test-secret-key")
	badJSONResp, _ := http.DefaultClient.Do(badJSONReq)
	badJSONResp.Body.Close()
	if badJSONResp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 on bad JSON, got %d", badJSONResp.StatusCode)
	}

	emptyMsgs, _ := doJSON(t, "POST", ts.URL+"/v1/chat/completions", "test-secret-key", map[string]any{"messages": []any{}})
	if emptyMsgs.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 on empty messages, got %d", emptyMsgs.StatusCode)
	}

	badRole, _ := doJSON(t, "POST", ts.URL+"/v1/chat/completions", "test-secret-key", map[string]any{
		"messages": []map[string]any{{"role": "unknown_role", "content": "x"}},
	})
	if badRole.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 on invalid role, got %d", badRole.StatusCode)
	}

	notFoundOpenAI, _ := doJSON(t, "GET", ts.URL+"/v1/nonexistent", "test-secret-key", nil)
	if notFoundOpenAI.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 on unknown OpenAI route, got %d", notFoundOpenAI.StatusCode)
	}
}

func TestAnthropicExtendedFormatsStopSequencesAndServerLifecycle(t *testing.T) {
	ts, _ := newTestGateway(t, nil)

	// 1. Anthropic /v1/messages with string system, base64 + URL images, nested tool_result array blocks,
	// tool_choice={"type":"none"}, stop_sequences, and X-Session-ID header!
	reqPayload := map[string]any{
		"model":          "acp-fast-v1",
		"max_tokens":     512,
		"system":         "Plain string system prompt",
		"stop_sequences": []string{"Echo="},
		"tool_choice":    map[string]any{"type": "none"},
		"metadata":       map[string]any{"user_id": "user_42"},
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": "Inspect these:"},
					{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
					{"type": "image", "source": map[string]any{"type": "url", "url": "https://example.com/pic.png"}},
				},
			},
			{
				"role": "assistant",
				"content": []map[string]any{
					{"type": "thinking", "thinking": "prior thought"},
					{"type": "tool_use", "id": "tu_1", "name": "inspect", "input": map[string]any{"x": 1}},
				},
			},
			{
				"role": "user",
				"content": []map[string]any{
					{
						"type":        "tool_result",
						"tool_use_id": "tu_1",
						"is_error":    true,
						"content": []map[string]any{
							{"type": "text", "text": "nested error output"},
							{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
						},
					},
				},
			},
		},
	}

	b, _ := json.Marshal(reqPayload)
	httpReq, _ := http.NewRequest("POST", ts.URL+"/v1/v1/messages", bytes.NewReader(b))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "test-secret-key") // raw Authorization without Bearer prefix also supported
	httpReq.Header.Set("X-Session-ID", "explicit-sess-99")
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		t.Fatalf("anthropic request failed: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("anthropic status=%d body=%s", resp.StatusCode, string(raw))
	}
	var anthResp struct {
		StopReason   string `json:"stop_reason"`
		StopSequence string `json:"stop_sequence"`
	}
	_ = json.Unmarshal(raw, &anthResp)
	if anthResp.StopReason != "stop_sequence" || anthResp.StopSequence != "Echo=" {
		t.Fatalf("expected stop_sequence='Echo=', got: %s", string(raw))
	}

	// 2. Anthropic validation errors & 404
	badAnthRole, _ := doJSON(t, "POST", ts.URL+"/v1/messages", "test-secret-key", map[string]any{
		"model": "acp-fast-v1", "messages": []map[string]any{{"role": "system", "content": "not allowed in anthropic messages"}},
	})
	if badAnthRole.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 on invalid anthropic role, got %d", badAnthRole.StatusCode)
	}

	notFoundAnth, _ := doJSON(t, "POST", ts.URL+"/v1/messages/unknown_subroute", "test-secret-key", map[string]any{})
	if notFoundAnth.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 on unknown anthropic subroute, got %d", notFoundAnth.StatusCode)
	}

	// 3. Test degraded /healthz (503) on unstarted pool + real TCP Server Shutdown lifecycle!
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	freeAddr := ln.Addr().String()
	_ = ln.Close()

	cfg := config.Default()
	cfg.Server.ListenAddr = freeAddr
	cfg.ACP.Command = "echo"
	_ = cfg.NormalizeAndValidate()

	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	unstartedPool := pool.New(cfg, logger)
	defer unstartedPool.Close()

	realSrv := server.New(cfg, unstartedPool, logger)
	srvDone := make(chan error, 1)
	go func() {
		srvDone <- realSrv.ListenAndServe()
	}()

	// Poll /healthz until TCP server is listening (returns 503 because unstartedPool has 0 healthy workers)
	var healthCode int
	for i := 0; i < 30; i++ {
		r, err := http.Get("http://" + freeAddr + "/healthz")
		if err == nil {
			healthCode = r.StatusCode
			r.Body.Close()
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if healthCode != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 on degraded /healthz, got %d", healthCode)
	}

	shutCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := realSrv.Shutdown(shutCtx); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}
	if err := <-srvDone; err != nil && err != http.ErrServerClosed {
		t.Fatalf("unexpected ListenAndServe exit error: %v", err)
	}
}
