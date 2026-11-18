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
	"time"

	"acp2api/internal/acp"
	"acp2api/internal/bridge"
)

// --- OpenAI Models Endpoints ---

func (s *Server) handleOpenAIListModels(w http.ResponseWriter, _ *http.Request) {
	models := s.pool.AvailableModels()
	now := time.Now().Unix()
	data := make([]map[string]any, 0, len(models))
	for _, id := range models {
		data = append(data, map[string]any{
			"id":       id,
			"object":   "model",
			"created":  now,
			"owned_by": "acp2api",
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   data,
	})
}

func (s *Server) handleOpenAIRetrieveModel(w http.ResponseWriter, r *http.Request) {
	modelID := strings.TrimSpace(r.PathValue("model"))
	if modelID == "" {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "model id is required")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":       modelID,
		"object":   "model",
		"created":  time.Now().Unix(),
		"owned_by": "acp2api",
	})
}

// --- OpenAI Chat Completions (/v1/chat/completions) ---

type openAIChatRequest struct {
	Model           string            `json:"model"`
	Messages        []openAIMessage   `json:"messages"`
	Stream          bool              `json:"stream"`
	StreamOptions   *openAIStreamOpts `json:"stream_options,omitempty"`
	Tools           []openAITool      `json:"tools,omitempty"`
	ToolChoice      json.RawMessage   `json:"tool_choice,omitempty"`
	Stop            json.RawMessage   `json:"stop,omitempty"`
	ReasoningEffort string            `json:"reasoning_effort,omitempty"`
	User            string            `json:"user,omitempty"`
}

type openAIStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

type openAIMessage struct {
	Role             string           `json:"role"`
	Content          json.RawMessage  `json:"content"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	Name             string           `json:"name,omitempty"`
	ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
}

type openAITool struct {
	Type     string            `json:"type"`
	Function openAIToolFuncDef `json:"function"`
}

type openAIToolFuncDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type openAIToolCall struct {
	Index    *int               `json:"index,omitempty"`
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openAIToolCallFunc `json:"function"`
}

type openAIToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func (s *Server) handleOpenAIChatCompletions(w http.ResponseWriter, r *http.Request) {
	var rawReq openAIChatRequest
	if err := json.NewDecoder(r.Body).Decode(&rawReq); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("invalid JSON body: %v", err))
		return
	}
	if len(rawReq.Messages) == 0 {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "messages must be a non-empty array")
		return
	}

	uReq, err := convertOpenAIChatRequest(&rawReq, r)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout())
	defer cancel()

	if uReq.Stream {
		s.serveOpenAIChatStream(ctx, w, uReq, rawReq.StreamOptions != nil && rawReq.StreamOptions.IncludeUsage)
		return
	}

	resp, err := s.pool.ExecuteTurn(ctx, uReq, nil)
	if err != nil {
		status, errType := classifyTurnErrorOpenAI(err)
		writeOpenAIError(w, status, errType, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-ACP-Session-ID", resp.SessionID)
	w.Header().Set("X-ACP-Session-Reused", strconv.FormatBool(resp.SessionReused))
	if resp.ThinkingStrategy != "" && resp.ThinkingStrategy != "none" {
		w.Header().Set("X-ACP-Thinking-Strategy", resp.ThinkingStrategy)
	}

	msgObj := map[string]any{
		"role": "assistant",
	}
	if len(resp.ToolCalls) > 0 && resp.Text == "" {
		msgObj["content"] = nil
	} else {
		msgObj["content"] = resp.Text
	}
	if resp.Thinking != "" {
		msgObj["reasoning_content"] = resp.Thinking
	}
	if len(resp.ToolCalls) > 0 {
		tcs := make([]openAIToolCall, 0, len(resp.ToolCalls))
		for _, tc := range resp.ToolCalls {
			tcs = append(tcs, openAIToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: openAIToolCallFunc{
					Name:      tc.Name,
					Arguments: tc.Arguments,
				},
			})
		}
		msgObj["tool_calls"] = tcs
	}

	payload := map[string]any{
		"id":      newShortID("chatcmpl-"),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   resp.Model,
		"choices": []map[string]any{
			{
				"index":         0,
				"message":       msgObj,
				"finish_reason": mapOpenAIFinishReason(resp.StopReason),
			},
		},
		"usage": map[string]int{
			"prompt_tokens":     resp.InputTokens,
			"completion_tokens": resp.OutputTokens,
			"total_tokens":      resp.InputTokens + resp.OutputTokens,
		},
	}
	_ = json.NewEncoder(w).Encode(payload)
}

func classifyTurnErrorOpenAI(err error) (int, string) {
	switch {
	case errors.Is(err, acp.ErrUnsupportedThinkingLevel):
		return http.StatusBadRequest, "invalid_request_error"
	case errors.Is(err, context.Canceled):
		return 499, "invalid_request_error"
	case errors.Is(err, context.DeadlineExceeded):
		return http.StatusGatewayTimeout, "server_error"
	default:
		return http.StatusBadGateway, "server_error"
	}
}

func (s *Server) serveOpenAIChatStream(
	ctx context.Context,
	w http.ResponseWriter,
	uReq *bridge.UnifiedRequest,
	includeUsage bool,
) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOpenAIError(w, http.StatusInternalServerError, "server_error", "streaming is not supported by response writer")
		return
	}

	completionID := newShortID("chatcmpl-")
	created := time.Now().Unix()
	_, displayModel := s.cfg.ResolveModel(uReq.Model)
	if strings.TrimSpace(uReq.Model) != "" {
		displayModel = strings.TrimSpace(uReq.Model)
	}

	headersSent := false
	toolIdx := 0

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

		writeOpenAISSEChunk(w, flusher, completionID, created, displayModel, map[string]any{
			"role":    "assistant",
			"content": "",
		}, nil, nil)
	}

	resp, err := s.pool.ExecuteTurn(ctx, uReq, func(ev bridge.StreamEvent) {
		ensureStarted()
		switch ev.Kind {
		case bridge.EventThinkingDelta:
			writeOpenAISSEChunk(w, flusher, completionID, created, displayModel, map[string]any{
				"reasoning_content": ev.Text,
			}, nil, nil)
		case bridge.EventTextDelta:
			writeOpenAISSEChunk(w, flusher, completionID, created, displayModel, map[string]any{
				"content": ev.Text,
			}, nil, nil)
		case bridge.EventToolCall:
			if ev.ToolCall == nil {
				return
			}
			idx := toolIdx
			toolIdx++
			writeOpenAISSEChunk(w, flusher, completionID, created, displayModel, map[string]any{
				"tool_calls": []map[string]any{
					{
						"index": idx,
						"id":    ev.ToolCall.ID,
						"type":  "function",
						"function": map[string]any{
							"name":      ev.ToolCall.Name,
							"arguments": "",
						},
					},
				},
			}, nil, nil)
			writeOpenAISSEChunk(w, flusher, completionID, created, displayModel, map[string]any{
				"tool_calls": []map[string]any{
					{
						"index": idx,
						"function": map[string]any{
							"arguments": ev.ToolCall.Arguments,
						},
					},
				},
			}, nil, nil)
		}
	})

	if err != nil {
		if !headersSent {
			status, errType := classifyTurnErrorOpenAI(err)
			writeOpenAIError(w, status, errType, err.Error())
			return
		}
		errPayload, _ := json.Marshal(map[string]any{
			"error": map[string]any{
				"message": err.Error(),
				"type":    "server_error",
			},
		})
		_, _ = fmt.Fprintf(w, "data: %s\n\n", errPayload)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	ensureStarted()
	finish := mapOpenAIFinishReason(resp.StopReason)
	writeOpenAISSEChunk(w, flusher, completionID, created, resp.Model, map[string]any{}, &finish, nil)

	if includeUsage {
		usageMap := map[string]int{
			"prompt_tokens":     resp.InputTokens,
			"completion_tokens": resp.OutputTokens,
			"total_tokens":      resp.InputTokens + resp.OutputTokens,
		}
		chunk := map[string]any{
			"id":      completionID,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   resp.Model,
			"choices": []any{},
			"usage":   usageMap,
		}
		b, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeOpenAISSEChunk(
	w http.ResponseWriter,
	flusher http.Flusher,
	id string,
	created int64,
	model string,
	delta map[string]any,
	finishReason *string,
	usage map[string]int,
) {
	choice := map[string]any{
		"index":         0,
		"delta":         delta,
		"finish_reason": finishReason,
	}
	chunk := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []any{choice},
	}
	if usage != nil {
		chunk["usage"] = usage
	}
	b, _ := json.Marshal(chunk)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
	flusher.Flush()
}

func convertOpenAIChatRequest(raw *openAIChatRequest, r *http.Request) (*bridge.UnifiedRequest, error) {
	var sysParts []string
	var messages []bridge.UnifiedMessage

	for _, m := range raw.Messages {
		role := strings.ToLower(strings.TrimSpace(m.Role))
		switch role {
		case "system", "developer":
			txt, _, err := parseOpenAIContent(m.Content)
			if err != nil {
				return nil, err
			}
			if strings.TrimSpace(txt) != "" {
				sysParts = append(sysParts, txt)
			}

		case "user":
			_, parts, err := parseOpenAIContent(m.Content)
			if err != nil {
				return nil, err
			}
			messages = append(messages, bridge.UnifiedMessage{
				Role:  "user",
				Parts: parts,
			})

		case "assistant":
			var parts []bridge.ContentPart
			if strings.TrimSpace(m.ReasoningContent) != "" {
				parts = append(parts, bridge.ContentPart{
					Kind:     bridge.PartThinking,
					Thinking: m.ReasoningContent,
				})
			}
			_, contentParts, err := parseOpenAIContent(m.Content)
			if err != nil {
				return nil, err
			}
			parts = append(parts, contentParts...)
			for _, tc := range m.ToolCalls {
				tcVal := bridge.ToolCallValue{
					ID:        tc.ID,
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				}
				parts = append(parts, bridge.ContentPart{
					Kind:     bridge.PartToolCall,
					ToolCall: &tcVal,
				})
			}
			messages = append(messages, bridge.UnifiedMessage{
				Role:  "assistant",
				Parts: parts,
			})

		case "tool":
			txt, _, err := parseOpenAIContent(m.Content)
			if err != nil {
				return nil, err
			}
			tr := bridge.ToolResultVal{
				ToolCallID: m.ToolCallID,
				Name:       m.Name,
				Content:    txt,
			}
			messages = append(messages, bridge.UnifiedMessage{
				Role: "tool",
				Parts: []bridge.ContentPart{
					{Kind: bridge.PartToolResult, ToolResult: &tr},
				},
			})

		default:
			return nil, fmt.Errorf("unsupported message role %q", m.Role)
		}
	}

	tools := make([]bridge.ToolDefinition, 0, len(raw.Tools))
	for _, t := range raw.Tools {
		if t.Function.Name == "" {
			continue
		}
		tools = append(tools, bridge.ToolDefinition{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}

	return &bridge.UnifiedRequest{
		Model:           raw.Model,
		SessionHint:     resolveSessionHint(r, raw.User),
		ResumeSessionID: resolveResumeSessionID(r),
		Mode:            resolveACPMode(r),
		Thinking:        acp.NormalizeThinkingIntent(nil, raw.ReasoningEffort, 0),
		System:          strings.Join(sysParts, "\n\n"),
		Messages:        messages,
		Tools:           tools,
		ToolChoice:      parseOpenAIToolChoice(raw.ToolChoice),
		Stream:          raw.Stream,
		StopSequences:   parseStopSequences(raw.Stop),
	}, nil
}

func parseOpenAIContent(raw json.RawMessage) (string, []bridge.ContentPart, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return "", nil, nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return "", nil, fmt.Errorf("invalid string content: %w", err)
		}
		if s == "" {
			return "", nil, nil
		}
		return s, []bridge.ContentPart{{Kind: bridge.PartText, Text: s}}, nil
	}
	if trimmed[0] == '[' {
		var items []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			ImageURL *struct {
				URL string `json:"url"`
			} `json:"image_url"`
			InputAudio *struct {
				Data   string `json:"data"`
				Format string `json:"format"`
			} `json:"input_audio"`
			File *struct {
				Filename string `json:"filename"`
				FileData string `json:"file_data"`
			} `json:"file"`
		}
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return "", nil, fmt.Errorf("invalid content array: %w", err)
		}
		var sb strings.Builder
		var parts []bridge.ContentPart
		for _, it := range items {
			switch it.Type {
			case "text", "input_text", "output_text":
				sb.WriteString(it.Text)
				parts = append(parts, bridge.ContentPart{Kind: bridge.PartText, Text: it.Text})
			case "image_url", "input_image":
				if it.ImageURL != nil && it.ImageURL.URL != "" {
					img := parseImageURL(it.ImageURL.URL)
					parts = append(parts, bridge.ContentPart{Kind: bridge.PartImage, Image: &img})
				}
			case "input_audio":
				if it.InputAudio != nil && it.InputAudio.Data != "" {
					mime := "audio/wav"
					if f := strings.ToLower(strings.TrimSpace(it.InputAudio.Format)); f != "" {
						if f == "mp3" {
							mime = "audio/mpeg"
						} else if !strings.Contains(f, "/") {
							mime = "audio/" + f
						} else {
							mime = f
						}
					}
					aud := bridge.AudioPayload{MimeType: mime, Data: it.InputAudio.Data}
					parts = append(parts, bridge.ContentPart{Kind: bridge.PartAudio, Audio: &aud})
				}
			case "file", "input_file":
				if it.File != nil {
					res := parseOpenAIFileResource(it.File.Filename, it.File.FileData)
					parts = append(parts, bridge.ContentPart{Kind: bridge.PartResource, Resource: &res})
				}
			}
		}
		return sb.String(), parts, nil
	}
	return "", nil, errors.New("message content must be a string, array, or null")
}

func parseOpenAIFileResource(filename, fileData string) bridge.ResourcePayload {
	uri := "file:///attachment"
	if filename != "" {
		uri = "file:///" + filename
	}
	mime := "application/octet-stream"
	blob := fileData
	if strings.HasPrefix(fileData, "data:") {
		rest := fileData[len("data:"):]
		if semi := strings.Index(rest, ";base64,"); semi >= 0 {
			if rest[:semi] != "" {
				mime = rest[:semi]
			}
			blob = rest[semi+len(";base64,"):]
		}
	}
	return bridge.ResourcePayload{
		URI:      uri,
		Name:     filename,
		MimeType: mime,
		Blob:     blob,
	}
}

func parseImageURL(u string) bridge.ImagePayload {
	if strings.HasPrefix(u, "data:") {
		rest := u[len("data:"):]
		if semi := strings.Index(rest, ";base64,"); semi >= 0 {
			mime := rest[:semi]
			if mime == "" {
				mime = "image/png"
			}
			return bridge.ImagePayload{
				MimeType: mime,
				Data:     rest[semi+len(";base64,"):],
			}
		}
	}
	return bridge.ImagePayload{
		URI: u,
	}
}

func parseOpenAIToolChoice(raw json.RawMessage) bridge.ToolChoiceSpec {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return bridge.ToolChoiceSpec{Mode: "auto"}
	}
	if trimmed[0] == '"' {
		var s string
		if json.Unmarshal(trimmed, &s) == nil {
			switch s {
			case "none", "auto", "required":
				return bridge.ToolChoiceSpec{Mode: s}
			}
		}
		return bridge.ToolChoiceSpec{Mode: "auto"}
	}
	var obj struct {
		Type     string `json:"type"`
		Name     string `json:"name"`
		Function *struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(trimmed, &obj) == nil && obj.Type == "function" {
		fnName := obj.Name
		if obj.Function != nil && obj.Function.Name != "" {
			fnName = obj.Function.Name
		}
		if fnName != "" {
			return bridge.ToolChoiceSpec{Mode: "tool", ToolName: fnName}
		}
	}
	return bridge.ToolChoiceSpec{Mode: "auto"}
}

func parseStopSequences(raw json.RawMessage) []string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return nil
	}
	if trimmed[0] == '"' {
		var s string
		if json.Unmarshal(trimmed, &s) == nil && s != "" {
			return []string{s}
		}
		return nil
	}
	var arr []string
	if json.Unmarshal(trimmed, &arr) == nil {
		return arr
	}
	return nil
}

func mapOpenAIFinishReason(stopReason string) string {
	switch stopReason {
	case "tool_calls":
		return "tool_calls"
	case "max_tokens":
		return "length"
	default:
		return "stop"
	}
}

// --- OpenAI Responses API (/v1/responses) ---

type openAIResponsesRequest struct {
	Model           string          `json:"model"`
	Instructions    string          `json:"instructions,omitempty"`
	Input           json.RawMessage `json:"input"`
	Stream          bool            `json:"stream"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
	Reasoning       *struct {
		Effort string `json:"effort,omitempty"`
	} `json:"reasoning,omitempty"`
	Tools []struct {
		Type        string          `json:"type"`
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
		Function    *struct {
			Name        string          `json:"name"`
			Description string          `json:"description,omitempty"`
			Parameters  json.RawMessage `json:"parameters,omitempty"`
		} `json:"function,omitempty"`
	} `json:"tools,omitempty"`
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`
	User       string          `json:"user,omitempty"`
}

func (s *Server) handleOpenAIResponses(w http.ResponseWriter, r *http.Request) {
	var rawReq openAIResponsesRequest
	if err := json.NewDecoder(r.Body).Decode(&rawReq); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("invalid JSON body: %v", err))
		return
	}

	uReq, err := convertOpenAIResponsesRequest(&rawReq, r)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout())
	defer cancel()

	if uReq.Stream {
		s.serveOpenAIResponsesStream(ctx, w, uReq)
		return
	}

	resp, err := s.pool.ExecuteTurn(ctx, uReq, nil)
	if err != nil {
		status, errType := classifyTurnErrorOpenAI(err)
		writeOpenAIError(w, status, errType, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-ACP-Session-ID", resp.SessionID)
	w.Header().Set("X-ACP-Session-Reused", strconv.FormatBool(resp.SessionReused))
	if resp.ThinkingStrategy != "" && resp.ThinkingStrategy != "none" {
		w.Header().Set("X-ACP-Thinking-Strategy", resp.ThinkingStrategy)
	}

	payload := buildResponsesObject(newShortID("resp_"), resp)
	_ = json.NewEncoder(w).Encode(payload)
}

func convertOpenAIResponsesRequest(raw *openAIResponsesRequest, r *http.Request) (*bridge.UnifiedRequest, error) {
	var sysParts []string
	if strings.TrimSpace(raw.Instructions) != "" {
		sysParts = append(sysParts, raw.Instructions)
	}

	var msgs []bridge.UnifiedMessage
	trimmedInput := bytes.TrimSpace(raw.Input)
	if len(trimmedInput) == 0 || string(trimmedInput) == "null" {
		return nil, errors.New("input is required")
	}

	if trimmedInput[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmedInput, &s); err != nil {
			return nil, err
		}
		msgs = append(msgs, bridge.UnifiedMessage{
			Role:  "user",
			Parts: []bridge.ContentPart{{Kind: bridge.PartText, Text: s}},
		})
	} else if trimmedInput[0] == '[' {
		var items []struct {
			Type      string          `json:"type"`
			Role      string          `json:"role"`
			Content   json.RawMessage `json:"content"`
			CallID    string          `json:"call_id"`
			Name      string          `json:"name"`
			Arguments string          `json:"arguments"`
			Output    string          `json:"output"`
		}
		if err := json.Unmarshal(trimmedInput, &items); err != nil {
			return nil, fmt.Errorf("invalid input array: %w", err)
		}
		for _, it := range items {
			switch it.Type {
			case "function_call":
				tc := bridge.ToolCallValue{ID: it.CallID, Name: it.Name, Arguments: it.Arguments}
				msgs = append(msgs, bridge.UnifiedMessage{
					Role:  "assistant",
					Parts: []bridge.ContentPart{{Kind: bridge.PartToolCall, ToolCall: &tc}},
				})
			case "function_call_output":
				tr := bridge.ToolResultVal{ToolCallID: it.CallID, Content: it.Output}
				msgs = append(msgs, bridge.UnifiedMessage{
					Role:  "tool",
					Parts: []bridge.ContentPart{{Kind: bridge.PartToolResult, ToolResult: &tr}},
				})
			default:
				role := strings.ToLower(strings.TrimSpace(it.Role))
				if role == "system" || role == "developer" {
					txt, _, _ := parseOpenAIContent(it.Content)
					if txt != "" {
						sysParts = append(sysParts, txt)
					}
					continue
				}
				if role == "" {
					role = "user"
				}
				_, parts, err := parseOpenAIContent(it.Content)
				if err != nil {
					return nil, err
				}
				msgs = append(msgs, bridge.UnifiedMessage{Role: role, Parts: parts})
			}
		}
	} else {
		return nil, errors.New("input must be a string or array")
	}

	var tools []bridge.ToolDefinition
	for _, t := range raw.Tools {
		name := t.Name
		desc := t.Description
		params := t.Parameters
		if t.Function != nil && t.Function.Name != "" {
			name = t.Function.Name
			desc = t.Function.Description
			params = t.Function.Parameters
		}
		if name != "" {
			tools = append(tools, bridge.ToolDefinition{Name: name, Description: desc, Parameters: params})
		}
	}

	effort := raw.ReasoningEffort
	if effort == "" && raw.Reasoning != nil {
		effort = raw.Reasoning.Effort
	}

	return &bridge.UnifiedRequest{
		Model:           raw.Model,
		SessionHint:     resolveSessionHint(r, raw.User),
		ResumeSessionID: resolveResumeSessionID(r),
		Mode:            resolveACPMode(r),
		Thinking:        acp.NormalizeThinkingIntent(nil, effort, 0),
		System:          strings.Join(sysParts, "\n\n"),
		Messages:        msgs,
		Tools:           tools,
		ToolChoice:      parseOpenAIToolChoice(raw.ToolChoice),
		Stream:          raw.Stream,
	}, nil
}

func buildResponsesObject(respID string, resp *bridge.TurnResponse) map[string]any {
	var output []map[string]any
	if resp.Thinking != "" {
		output = append(output, map[string]any{
			"type": "reasoning",
			"id":   newShortID("rs_"),
			"summary": []map[string]any{
				{
					"type": "summary_text",
					"text": resp.Thinking,
				},
			},
		})
	}
	if resp.Text != "" || len(resp.ToolCalls) == 0 {
		output = append(output, map[string]any{
			"type":   "message",
			"id":     newShortID("msg_"),
			"status": "completed",
			"role":   "assistant",
			"content": []map[string]any{
				{
					"type":        "output_text",
					"text":        resp.Text,
					"annotations": []any{},
				},
			},
		})
	}
	for _, tc := range resp.ToolCalls {
		output = append(output, map[string]any{
			"type":      "function_call",
			"id":        newShortID("fc_"),
			"call_id":   tc.ID,
			"name":      tc.Name,
			"arguments": tc.Arguments,
			"status":    "completed",
		})
	}
	return map[string]any{
		"id":          respID,
		"object":      "response",
		"created_at":  time.Now().Unix(),
		"status":      "completed",
		"model":       resp.Model,
		"output":      output,
		"output_text": resp.Text,
		"usage": map[string]int{
			"input_tokens":  resp.InputTokens,
			"output_tokens": resp.OutputTokens,
			"total_tokens":  resp.InputTokens + resp.OutputTokens,
		},
	}
}

func (s *Server) serveOpenAIResponsesStream(ctx context.Context, w http.ResponseWriter, uReq *bridge.UnifiedRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOpenAIError(w, http.StatusInternalServerError, "server_error", "streaming unsupported")
		return
	}
	respID := newShortID("resp_")
	_, displayModel := s.cfg.ResolveModel(uReq.Model)
	if strings.TrimSpace(uReq.Model) != "" {
		displayModel = strings.TrimSpace(uReq.Model)
	}
	headersSent := false

	writeEvent := func(eventName string, payload map[string]any) {
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
		w.WriteHeader(http.StatusOK)
		writeEvent("response.created", map[string]any{
			"type": "response.created",
			"response": map[string]any{
				"id":     respID,
				"object": "response",
				"status": "in_progress",
				"model":  displayModel,
			},
		})
	}

	resp, err := s.pool.ExecuteTurn(ctx, uReq, func(ev bridge.StreamEvent) {
		ensureStarted()
		switch ev.Kind {
		case bridge.EventThinkingDelta:
			writeEvent("response.reasoning_summary_text.delta", map[string]any{
				"type":  "response.reasoning_summary_text.delta",
				"delta": ev.Text,
			})
		case bridge.EventTextDelta:
			writeEvent("response.output_text.delta", map[string]any{
				"type":  "response.output_text.delta",
				"delta": ev.Text,
			})
		case bridge.EventToolCall:
			if ev.ToolCall != nil {
				writeEvent("response.function_call_arguments.done", map[string]any{
					"type":      "response.function_call_arguments.done",
					"call_id":   ev.ToolCall.ID,
					"name":      ev.ToolCall.Name,
					"arguments": ev.ToolCall.Arguments,
				})
			}
		}
	})

	if err != nil {
		if !headersSent {
			status, errType := classifyTurnErrorOpenAI(err)
			writeOpenAIError(w, status, errType, err.Error())
			return
		}
		writeEvent("error", map[string]any{"type": "error", "message": err.Error()})
		return
	}

	ensureStarted()
	writeEvent("response.completed", map[string]any{
		"type":     "response.completed",
		"response": buildResponsesObject(respID, resp),
	})
}
