package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"acp2api/internal/acp"
	"acp2api/internal/bridge"
)

type anthropicMessagesRequest struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	System        json.RawMessage    `json:"system,omitempty"`
	Messages      []anthropicMessage `json:"messages"`
	Stream        bool               `json:"stream"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	ToolChoice    json.RawMessage    `json:"tool_choice,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Thinking      *struct {
		Type         string `json:"type"`                    // "enabled", "adaptive", "disabled"
		BudgetTokens int    `json:"budget_tokens,omitempty"` // optional token budget
	} `json:"thinking,omitempty"`
	OutputConfig *struct {
		Effort string `json:"effort,omitempty"` // "low", "medium", "high", "max"
	} `json:"output_config,omitempty"`
	Metadata *struct {
		UserID string `json:"user_id,omitempty"`
	} `json:"metadata,omitempty"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type anthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	Title     string          `json:"title,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
	Source    *struct {
		Type      string `json:"type"` // "base64", "url", or "text"
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
		URL       string `json:"url"`
	} `json:"source,omitempty"`
}

func classifyTurnErrorAnthropic(err error) (int, string) {
	switch {
	case errors.Is(err, acp.ErrUnsupportedThinkingLevel):
		return http.StatusBadRequest, "invalid_request_error"
	case errors.Is(err, context.DeadlineExceeded):
		return http.StatusGatewayTimeout, "api_error"
	default:
		return http.StatusBadGateway, "api_error"
	}
}

func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	var rawReq anthropicMessagesRequest
	if err := json.NewDecoder(r.Body).Decode(&rawReq); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("invalid JSON body: %v", err))
		return
	}
	if len(rawReq.Messages) == 0 {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "messages must be a non-empty array")
		return
	}

	uReq, err := convertAnthropicRequest(&rawReq, r)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout())
	defer cancel()

	if uReq.Stream {
		s.serveAnthropicStream(ctx, w, uReq)
		return
	}

	resp, err := s.pool.ExecuteTurn(ctx, uReq, nil)
	if err != nil {
		status, errType := classifyTurnErrorAnthropic(err)
		writeAnthropicError(w, status, errType, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-ACP-Session-ID", resp.SessionID)
	w.Header().Set("X-ACP-Session-Reused", strconv.FormatBool(resp.SessionReused))
	if resp.ThinkingStrategy != "" && resp.ThinkingStrategy != "none" {
		w.Header().Set("X-ACP-Thinking-Strategy", resp.ThinkingStrategy)
	}

	var contentBlocks []map[string]any
	if resp.Thinking != "" {
		contentBlocks = append(contentBlocks, map[string]any{
			"type":      "thinking",
			"thinking":  resp.Thinking,
			"signature": "acp-thought",
		})
	}
	if resp.Text != "" || (len(resp.ToolCalls) == 0 && resp.Thinking == "") {
		contentBlocks = append(contentBlocks, map[string]any{
			"type": "text",
			"text": resp.Text,
		})
	}
	for _, tc := range resp.ToolCalls {
		var inputObj any = map[string]any{}
		if tc.Arguments != "" {
			_ = json.Unmarshal([]byte(tc.Arguments), &inputObj)
		}
		contentBlocks = append(contentBlocks, map[string]any{
			"type":  "tool_use",
			"id":    tc.ID,
			"name":  tc.Name,
			"input": inputObj,
		})
	}

	var stopSeq any
	if resp.StopSequence != "" {
		stopSeq = resp.StopSequence
	}

	payload := map[string]any{
		"id":            newShortID("msg_"),
		"type":          "message",
		"role":          "assistant",
		"model":         resp.Model,
		"content":       contentBlocks,
		"stop_reason":   mapAnthropicStopReason(resp.StopReason),
		"stop_sequence": stopSeq,
		"usage": map[string]int{
			"input_tokens":  resp.InputTokens,
			"output_tokens": resp.OutputTokens,
		},
	}
	_ = json.NewEncoder(w).Encode(payload)
}

func (s *Server) handleAnthropicCountTokens(w http.ResponseWriter, r *http.Request) {
	var rawReq anthropicMessagesRequest
	if err := json.NewDecoder(r.Body).Decode(&rawReq); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("invalid JSON body: %v", err))
		return
	}
	uReq, err := convertAnthropicRequest(&rawReq, r)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]int{
		"input_tokens": bridge.EstimateRequestTokens(uReq),
	})
}

func (s *Server) serveAnthropicStream(ctx context.Context, w http.ResponseWriter, uReq *bridge.UnifiedRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "streaming is not supported by response writer")
		return
	}

	msgID := newShortID("msg_")
	_, displayModel := s.cfg.ResolveModel(uReq.Model)
	if strings.TrimSpace(uReq.Model) != "" {
		displayModel = strings.TrimSpace(uReq.Model)
	}
	estInput := bridge.EstimateRequestTokens(uReq)

	headersSent := false
	nextBlockIdx := 0
	activeBlockIdx := -1
	activeKind := "" // "", "thinking", "text"

	writeSSE := func(eventName string, payload map[string]any) {
		b, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventName, b)
		flusher.Flush()
	}

	ensureStarted := func() {
		if headersSent {
			return
		}
		headersSent = true
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)

		writeSSE("message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id":            msgID,
				"type":          "message",
				"role":          "assistant",
				"model":         displayModel,
				"content":       []any{},
				"stop_reason":   nil,
				"stop_sequence": nil,
				"usage": map[string]int{
					"input_tokens":  estInput,
					"output_tokens": 0,
				},
			},
		})
		writeSSE("ping", map[string]any{"type": "ping"})
	}

	closeActiveBlock := func() {
		if activeKind == "" {
			return
		}
		if activeKind == "thinking" {
			writeSSE("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": activeBlockIdx,
				"delta": map[string]any{
					"type":      "signature_delta",
					"signature": "acp-thought",
				},
			})
		}
		writeSSE("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": activeBlockIdx,
		})
		activeKind = ""
		activeBlockIdx = -1
	}

	resp, err := s.pool.ExecuteTurn(ctx, uReq, func(ev bridge.StreamEvent) {
		ensureStarted()
		switch ev.Kind {
		case bridge.EventThinkingDelta:
			if activeKind != "thinking" {
				closeActiveBlock()
				activeBlockIdx = nextBlockIdx
				nextBlockIdx++
				activeKind = "thinking"
				writeSSE("content_block_start", map[string]any{
					"type":  "content_block_start",
					"index": activeBlockIdx,
					"content_block": map[string]any{
						"type":     "thinking",
						"thinking": "",
					},
				})
			}
			writeSSE("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": activeBlockIdx,
				"delta": map[string]any{
					"type":     "thinking_delta",
					"thinking": ev.Text,
				},
			})

		case bridge.EventTextDelta:
			if activeKind != "text" {
				closeActiveBlock()
				activeBlockIdx = nextBlockIdx
				nextBlockIdx++
				activeKind = "text"
				writeSSE("content_block_start", map[string]any{
					"type":  "content_block_start",
					"index": activeBlockIdx,
					"content_block": map[string]any{
						"type": "text",
						"text": "",
					},
				})
			}
			writeSSE("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": activeBlockIdx,
				"delta": map[string]any{
					"type": "text_delta",
					"text": ev.Text,
				},
			})

		case bridge.EventToolCall:
			if ev.ToolCall == nil {
				return
			}
			closeActiveBlock()
			idx := nextBlockIdx
			nextBlockIdx++
			writeSSE("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": idx,
				"content_block": map[string]any{
					"type":  "tool_use",
					"id":    ev.ToolCall.ID,
					"name":  ev.ToolCall.Name,
					"input": map[string]any{},
				},
			})
			writeSSE("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{
					"type":         "input_json_delta",
					"partial_json": ev.ToolCall.Arguments,
				},
			})
			writeSSE("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": idx,
			})
		}
	})

	if err != nil {
		if !headersSent {
			status, errType := classifyTurnErrorAnthropic(err)
			writeAnthropicError(w, status, errType, err.Error())
			return
		}
		closeActiveBlock()
		writeSSE("error", map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "api_error",
				"message": err.Error(),
			},
		})
		return
	}

	ensureStarted()
	closeActiveBlock()

	if nextBlockIdx == 0 {
		writeSSE("content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": 0,
			"content_block": map[string]any{
				"type": "text",
				"text": "",
			},
		})
		writeSSE("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": 0,
		})
	}

	var stopSeq any
	if resp.StopSequence != "" {
		stopSeq = resp.StopSequence
	}

	writeSSE("message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   mapAnthropicStopReason(resp.StopReason),
			"stop_sequence": stopSeq,
		},
		"usage": map[string]int{
			"output_tokens": resp.OutputTokens,
		},
	})
	writeSSE("message_stop", map[string]any{
		"type": "message_stop",
	})
}

func convertAnthropicRequest(raw *anthropicMessagesRequest, r *http.Request) (*bridge.UnifiedRequest, error) {
	sysText, err := parseAnthropicSystem(raw.System)
	if err != nil {
		return nil, err
	}

	var msgs []bridge.UnifiedMessage
	for _, m := range raw.Messages {
		role := strings.ToLower(strings.TrimSpace(m.Role))
		if role != "user" && role != "assistant" {
			return nil, fmt.Errorf("unsupported anthropic message role %q (expected user or assistant)", m.Role)
		}
		parts, err := parseAnthropicMessageContent(m.Content)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, bridge.UnifiedMessage{
			Role:  role,
			Parts: parts,
		})
	}

	tools := make([]bridge.ToolDefinition, 0, len(raw.Tools))
	for _, t := range raw.Tools {
		if strings.TrimSpace(t.Name) == "" {
			continue
		}
		tools = append(tools, bridge.ToolDefinition{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.InputSchema,
		})
	}

	userHint := ""
	if raw.Metadata != nil {
		userHint = raw.Metadata.UserID
	}

	return &bridge.UnifiedRequest{
		Model:           raw.Model,
		SessionHint:     resolveSessionHint(r, userHint),
		ResumeSessionID: resolveResumeSessionID(r),
		Mode:            resolveACPMode(r),
		Thinking:        parseAnthropicThinkingIntent(raw),
		System:          sysText,
		Messages:        msgs,
		Tools:           tools,
		ToolChoice:      parseAnthropicToolChoice(raw.ToolChoice),
		Stream:          raw.Stream,
		StopSequences:   raw.StopSequences,
	}, nil
}

func parseAnthropicThinkingIntent(raw *anthropicMessagesRequest) acp.DesiredThinking {
	var enabled *bool
	effort := ""
	budget := 0

	if raw.Thinking != nil {
		switch strings.ToLower(strings.TrimSpace(raw.Thinking.Type)) {
		case "disabled":
			f := false
			enabled = &f
			effort = "none"
		case "enabled", "adaptive":
			t := true
			enabled = &t
			budget = raw.Thinking.BudgetTokens
		}
	}
	if raw.OutputConfig != nil && strings.TrimSpace(raw.OutputConfig.Effort) != "" {
		effort = raw.OutputConfig.Effort
	}
	return acp.NormalizeThinkingIntent(enabled, effort, budget)
}

func parseAnthropicSystem(raw json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return "", nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return "", fmt.Errorf("invalid system string: %w", err)
		}
		return s, nil
	}
	if trimmed[0] == '[' {
		var blocks []anthropicContentBlock
		if err := json.Unmarshal(trimmed, &blocks); err != nil {
			return "", fmt.Errorf("invalid system blocks: %w", err)
		}
		var parts []string
		for _, b := range blocks {
			if (b.Type == "" || b.Type == "text") && b.Text != "" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n\n"), nil
	}
	return "", errors.New("system must be a string or array of content blocks")
}

func parseAnthropicMessageContent(raw json.RawMessage) ([]bridge.ContentPart, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return nil, nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return nil, fmt.Errorf("invalid message content string: %w", err)
		}
		return []bridge.ContentPart{{Kind: bridge.PartText, Text: s}}, nil
	}
	if trimmed[0] != '[' {
		return nil, errors.New("message content must be a string or array of blocks")
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(trimmed, &blocks); err != nil {
		return nil, fmt.Errorf("invalid message content blocks: %w", err)
	}

	var out []bridge.ContentPart
	for _, b := range blocks {
		switch b.Type {
		case "text":
			out = append(out, bridge.ContentPart{Kind: bridge.PartText, Text: b.Text})

		case "thinking":
			if b.Thinking != "" {
				out = append(out, bridge.ContentPart{Kind: bridge.PartThinking, Thinking: b.Thinking})
			}

		case "image":
			if b.Source != nil {
				img := bridge.ImagePayload{
					MimeType: b.Source.MediaType,
				}
				if b.Source.Type == "url" {
					img.URI = b.Source.URL
				} else {
					img.Data = b.Source.Data
				}
				out = append(out, bridge.ContentPart{Kind: bridge.PartImage, Image: &img})
			}

		case "document":
			if b.Source != nil {
				uri := b.Source.URL
				if uri == "" {
					if b.Title != "" {
						uri = "file:///" + b.Title
					} else {
						uri = "file:///document"
					}
				}
				res := bridge.ResourcePayload{
					URI:      uri,
					Name:     b.Title,
					MimeType: b.Source.MediaType,
				}
				switch b.Source.Type {
				case "text":
					res.Text = b.Source.Data
				case "base64":
					res.Blob = b.Source.Data
				}
				out = append(out, bridge.ContentPart{Kind: bridge.PartResource, Resource: &res})
			}

		case "tool_use":
			argsStr := strings.TrimSpace(string(b.Input))
			if argsStr == "" || argsStr == "null" {
				argsStr = "{}"
			}
			tc := bridge.ToolCallValue{
				ID:        b.ID,
				Name:      b.Name,
				Arguments: argsStr,
			}
			out = append(out, bridge.ContentPart{Kind: bridge.PartToolCall, ToolCall: &tc})

		case "tool_result":
			txt, imgs := parseAnthropicToolResultContent(b.Content)
			tr := bridge.ToolResultVal{
				ToolCallID: b.ToolUseID,
				Content:    txt,
				IsError:    b.IsError,
				Images:     imgs,
			}
			out = append(out, bridge.ContentPart{Kind: bridge.PartToolResult, ToolResult: &tr})
		}
	}
	return out, nil
}

func parseAnthropicToolResultContent(raw json.RawMessage) (string, []bridge.ImagePayload) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return "", nil
	}
	if trimmed[0] == '"' {
		var s string
		if json.Unmarshal(trimmed, &s) == nil {
			return s, nil
		}
	}
	if trimmed[0] == '[' {
		var blocks []anthropicContentBlock
		if json.Unmarshal(trimmed, &blocks) == nil {
			var sb strings.Builder
			var imgs []bridge.ImagePayload
			for _, b := range blocks {
				switch b.Type {
				case "text":
					if sb.Len() > 0 {
						sb.WriteByte('\n')
					}
					sb.WriteString(b.Text)
				case "image":
					if b.Source != nil {
						img := bridge.ImagePayload{MimeType: b.Source.MediaType}
						if b.Source.Type == "url" {
							img.URI = b.Source.URL
						} else {
							img.Data = b.Source.Data
						}
						imgs = append(imgs, img)
					}
				}
			}
			return sb.String(), imgs
		}
	}
	return string(trimmed), nil
}

func parseAnthropicToolChoice(raw json.RawMessage) bridge.ToolChoiceSpec {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return bridge.ToolChoiceSpec{Mode: "auto"}
	}
	var obj struct {
		Type string `json:"type"` // "auto", "any", "none", "tool"
		Name string `json:"name"`
	}
	if json.Unmarshal(trimmed, &obj) == nil {
		switch obj.Type {
		case "none":
			return bridge.ToolChoiceSpec{Mode: "none"}
		case "any":
			return bridge.ToolChoiceSpec{Mode: "required"}
		case "tool":
			return bridge.ToolChoiceSpec{Mode: "tool", ToolName: obj.Name}
		}
	}
	return bridge.ToolChoiceSpec{Mode: "auto"}
}

func mapAnthropicStopReason(stopReason string) string {
	switch stopReason {
	case "tool_calls":
		return "tool_use"
	case "max_tokens":
		return "max_tokens"
	case "stop_sequence":
		return "stop_sequence"
	default:
		return "end_turn"
	}
}
