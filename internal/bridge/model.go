package bridge

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"unicode/utf8"
)

type PartKind string

const (
	PartText       PartKind = "text"
	PartImage      PartKind = "image"
	PartThinking   PartKind = "thinking"
	PartToolCall   PartKind = "tool_call"
	PartToolResult PartKind = "tool_result"
)

type ImagePayload struct {
	MimeType string `json:"mime_type,omitempty"`
	Data     string `json:"data,omitempty"` // base64-encoded bytes
	URI      string `json:"uri,omitempty"`
}

type ContentPart struct {
	Kind       PartKind       `json:"kind"`
	Text       string         `json:"text,omitempty"`
	Image      *ImagePayload  `json:"image,omitempty"`
	Thinking   string         `json:"thinking,omitempty"`
	ToolCall   *ToolCallValue `json:"tool_call,omitempty"`
	ToolResult *ToolResultVal `json:"tool_result,omitempty"`
}

type ToolCallValue struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON object string
}

type ToolResultVal struct {
	ToolCallID string         `json:"tool_call_id"`
	Name       string         `json:"name,omitempty"`
	Content    string         `json:"content"`
	IsError    bool           `json:"is_error,omitempty"`
	Images     []ImagePayload `json:"images,omitempty"`
}

type UnifiedMessage struct {
	Role  string        `json:"role"` // "user", "assistant", "tool"
	Parts []ContentPart `json:"parts"`
}

type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ToolChoiceSpec struct {
	Mode     string `json:"mode"` // "", "auto", "none", "required", "tool"
	ToolName string `json:"tool_name,omitempty"`
}

// UnifiedRequest is the normalized representation of an incoming OpenAI or Anthropic chat request.
type UnifiedRequest struct {
	Model         string
	SessionHint   string
	System        string
	Messages      []UnifiedMessage
	Tools         []ToolDefinition
	ToolChoice    ToolChoiceSpec
	Stream        bool
	StopSequences []string
}

// StreamEventKind identifies normalized events emitted during a turn execution.
type StreamEventKind string

const (
	EventThinkingDelta StreamEventKind = "thinking_delta"
	EventTextDelta     StreamEventKind = "text_delta"
	EventToolCall      StreamEventKind = "tool_call"
)

// StreamEvent is emitted in real time as the ACP session streams updates.
type StreamEvent struct {
	Kind     StreamEventKind
	Text     string
	ToolCall *ToolCallValue
}

// TurnResponse holds the completed assistant turn output and token usage.
type TurnResponse struct {
	Model         string
	SessionID     string
	SessionReused bool
	Thinking      string
	Text          string
	ToolCalls     []ToolCallValue
	StopReason    string // "end_turn", "tool_calls", "max_tokens", "stop_sequence", "cancelled"
	StopSequence  string
	InputTokens   int
	OutputTokens  int
}

// AssistantMessage converts a completed TurnResponse into a UnifiedMessage for conversation history caching.
func (r *TurnResponse) AssistantMessage() UnifiedMessage {
	var parts []ContentPart
	if r.Thinking != "" {
		parts = append(parts, ContentPart{
			Kind:     PartThinking,
			Thinking: r.Thinking,
		})
	}
	if r.Text != "" {
		parts = append(parts, ContentPart{
			Kind: PartText,
			Text: r.Text,
		})
	}
	for i := range r.ToolCalls {
		tc := r.ToolCalls[i]
		parts = append(parts, ContentPart{
			Kind:     PartToolCall,
			ToolCall: &tc,
		})
	}
	return UnifiedMessage{
		Role:  "assistant",
		Parts: parts,
	}
}

// ComputeConversationHash computes a deterministic SHA-256 fingerprint of (acpModel, system, tools, messages),
// normalizing ephemeral fields (such as thinking blocks and trailing whitespace) so subsequent stateless HTTP
// turns reliably match existing stateful ACP sessions.
func ComputeConversationHash(acpModel string, system string, tools []ToolDefinition, messages []UnifiedMessage) string {
	h := sha256.New()
	writeField := func(tag, val string) {
		h.Write([]byte(tag))
		h.Write([]byte{':'})
		h.Write([]byte(val))
		h.Write([]byte{0})
	}

	writeField("model", strings.TrimSpace(acpModel))
	writeField("system", strings.TrimSpace(system))

	for _, t := range tools {
		writeField("tool_name", t.Name)
		writeField("tool_desc", strings.TrimSpace(t.Description))
		writeField("tool_schema", compactJSONString(t.Parameters))
	}

	for _, m := range normalizeMessagesForHash(messages) {
		writeField("msg_role", m.Role)
		for _, p := range m.Parts {
			switch p.Kind {
			case PartThinking:
				// Intentionally skipped: clients may omit or echo thinking blocks across turns.
			case PartText:
				trimmed := strings.TrimSpace(p.Text)
				if trimmed != "" {
					writeField("part_text", trimmed)
				}
			case PartImage:
				if p.Image != nil {
					imgHash := sha256.Sum256([]byte(p.Image.MimeType + "|" + p.Image.URI + "|" + p.Image.Data))
					writeField("part_img", hex.EncodeToString(imgHash[:]))
				}
			case PartToolCall:
				if p.ToolCall != nil {
					writeField("part_tc", p.ToolCall.ID+"|"+p.ToolCall.Name+"|"+compactJSONString([]byte(p.ToolCall.Arguments)))
				}
			case PartToolResult:
				if p.ToolResult != nil {
					errFlag := "0"
					if p.ToolResult.IsError {
						errFlag = "1"
					}
					writeField("part_tr", p.ToolResult.ToolCallID+"|"+errFlag+"|"+strings.TrimSpace(p.ToolResult.Content))
				}
			}
		}
	}

	return hex.EncodeToString(h.Sum(nil))
}

// normalizeMessagesForHash merges adjacent tool-result messages into canonical order so OpenAI ("role":"tool" per call)
// and Anthropic ("role":"user" with multiple tool_result blocks) hash consistently within their own flows.
func normalizeMessagesForHash(msgs []UnifiedMessage) []UnifiedMessage {
	out := make([]UnifiedMessage, 0, len(msgs))
	for _, m := range msgs {
		role := m.Role
		if role == "developer" {
			role = "system"
		}
		if role == "tool" {
			// Normalize tool results into canonical "tool" role entries, coalescing adjacent tool messages.
			if len(out) > 0 && out[len(out)-1].Role == "tool" {
				out[len(out)-1].Parts = append(out[len(out)-1].Parts, m.Parts...)
				continue
			}
		}
		// Check if a "user" message contains exclusively PartToolResult (Anthropic style) -> normalize role to "tool".
		if role == "user" && len(m.Parts) > 0 && allToolResults(m.Parts) {
			if len(out) > 0 && out[len(out)-1].Role == "tool" {
				out[len(out)-1].Parts = append(out[len(out)-1].Parts, m.Parts...)
				continue
			}
			cp := UnifiedMessage{Role: "tool", Parts: append([]ContentPart(nil), m.Parts...)}
			out = append(out, cp)
			continue
		}
		cp := UnifiedMessage{Role: role, Parts: append([]ContentPart(nil), m.Parts...)}
		out = append(out, cp)
	}
	return out
}

func allToolResults(parts []ContentPart) bool {
	for _, p := range parts {
		if p.Kind != PartToolResult {
			return false
		}
	}
	return true
}

func compactJSONString(raw []byte) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return "{}"
	}
	var v any
	if err := json.Unmarshal([]byte(trimmed), &v); err != nil {
		return trimmed
	}
	norm, err := json.Marshal(v)
	if err != nil {
		return trimmed
	}
	return string(norm)
}

// EstimateTextTokens estimates LLM token count for UTF-8 text without external tokenizer dependencies.
func EstimateTextTokens(s string) int {
	if s == "" {
		return 0
	}
	asciiChars := 0
	nonASCIIRunes := 0
	for _, r := range s {
		if r < utf8.RuneSelf {
			asciiChars++
		} else {
			nonASCIIRunes++
		}
	}
	// Rough empirical approximation: ~4 ASCII chars/token, ~1.5 non-ASCII (CJK) runes/token.
	est := (asciiChars+3)/4 + (nonASCIIRunes*2+2)/3
	if est < 1 {
		est = 1
	}
	return est
}

// EstimateRequestTokens estimates input token usage for a UnifiedRequest.
func EstimateRequestTokens(req *UnifiedRequest) int {
	total := EstimateTextTokens(req.System)
	for _, t := range req.Tools {
		total += EstimateTextTokens(t.Name) + EstimateTextTokens(t.Description) + EstimateTextTokens(string(t.Parameters)) + 8
	}
	for _, m := range req.Messages {
		total += 4 // per-message framing overhead
		for _, p := range m.Parts {
			switch p.Kind {
			case PartText:
				total += EstimateTextTokens(p.Text)
			case PartThinking:
				total += EstimateTextTokens(p.Thinking)
			case PartImage:
				total += 256
			case PartToolCall:
				if p.ToolCall != nil {
					total += EstimateTextTokens(p.ToolCall.Name) + EstimateTextTokens(p.ToolCall.Arguments) + 8
				}
			case PartToolResult:
				if p.ToolResult != nil {
					total += EstimateTextTokens(p.ToolResult.Content) + 6
				}
			}
		}
	}
	if total < 1 {
		total = 1
	}
	return total
}
