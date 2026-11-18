package bridge

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
)

const (
	openTag  = "<tool_call>"
	closeTag = "</tool_call>"
)

var toolSeq atomic.Uint64

// ToolStreamParser incrementally separates assistant text chunks from <tool_call>...</tool_call> blocks
// with minimal latency (at most len("<tool_call>")-1 bytes held back on potential tag boundaries).
type ToolStreamParser struct {
	enabled       bool
	inToolCall    bool
	trimLeadingNL bool
	buf           string
	emit          func(StreamEvent)
}

func NewToolStreamParser(enabled bool, emit func(StreamEvent)) *ToolStreamParser {
	return &ToolStreamParser{
		enabled: enabled,
		emit:    emit,
	}
}

// Feed processes an incoming assistant text delta.
func (p *ToolStreamParser) Feed(chunk string) {
	if chunk == "" {
		return
	}
	if !p.enabled {
		p.emit(StreamEvent{Kind: EventTextDelta, Text: chunk})
		return
	}

	if p.trimLeadingNL && p.buf == "" {
		if strings.HasPrefix(chunk, "\r\n") {
			chunk = chunk[2:]
			p.trimLeadingNL = false
		} else if strings.HasPrefix(chunk, "\r") {
			// Wait one more chunk in case \n follows \r
			if len(chunk) == 1 {
				return
			}
			chunk = chunk[1:]
			p.trimLeadingNL = false
		} else if strings.HasPrefix(chunk, "\n") {
			chunk = chunk[1:]
			p.trimLeadingNL = false
		} else {
			p.trimLeadingNL = false
		}
		if chunk == "" {
			return
		}
	}

	p.buf += chunk
	for len(p.buf) > 0 {
		if !p.inToolCall {
			idx := strings.Index(p.buf, openTag)
			if idx >= 0 {
				before := p.buf[:idx]
				if before != "" {
					p.emit(StreamEvent{Kind: EventTextDelta, Text: before})
				}
				p.buf = p.buf[idx+len(openTag):]
				p.inToolCall = true
				continue
			}

			keep := longestSuffixPrefix(p.buf, openTag)
			flushLen := len(p.buf) - keep
			if flushLen > 0 {
				p.emit(StreamEvent{Kind: EventTextDelta, Text: p.buf[:flushLen]})
				p.buf = p.buf[flushLen:]
			}
			return
		}

		// Inside <tool_call>...
		endIdx := strings.Index(p.buf, closeTag)
		if endIdx < 0 {
			return
		}
		inner := p.buf[:endIdx]
		rest := p.buf[endIdx+len(closeTag):]
		p.inToolCall = false

		if tc, ok := parseToolCallPayload(inner); ok {
			// Trim a single leading newline after valid </tool_call>, even across chunk boundaries.
			if rest == "" {
				p.trimLeadingNL = true
			} else if strings.HasPrefix(rest, "\r\n") {
				rest = rest[2:]
			} else if strings.HasPrefix(rest, "\n") {
				rest = rest[1:]
			}
			p.buf = rest
			p.emit(StreamEvent{Kind: EventToolCall, ToolCall: &tc})
		} else {
			// Malformed JSON inside <tool_call>: emit raw text rather than dropping data.
			p.buf = rest
			p.emit(StreamEvent{Kind: EventTextDelta, Text: openTag + inner + closeTag})
		}
	}
}

// Finish flushes any buffered partial content at the end of a turn.
func (p *ToolStreamParser) Finish() {
	if !p.enabled || p.buf == "" {
		return
	}
	if p.inToolCall {
		if tc, ok := parseToolCallPayload(p.buf); ok {
			p.emit(StreamEvent{Kind: EventToolCall, ToolCall: &tc})
		} else {
			p.emit(StreamEvent{Kind: EventTextDelta, Text: openTag + p.buf})
		}
	} else {
		p.emit(StreamEvent{Kind: EventTextDelta, Text: p.buf})
	}
	p.buf = ""
	p.inToolCall = false
}

func longestSuffixPrefix(s, tag string) int {
	maxCheck := len(tag) - 1
	if len(s) < maxCheck {
		maxCheck = len(s)
	}
	for k := maxCheck; k >= 1; k-- {
		if strings.HasSuffix(s, tag[:k]) {
			return k
		}
	}
	return 0
}

type rawParsedToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	Input     json.RawMessage `json:"input"` // fallback if model uses Anthropic-style key
}

func parseToolCallPayload(raw string) (ToolCallValue, bool) {
	cleaned := strings.TrimSpace(raw)
	// Strip optional ```json ... ``` markdown fence if model wrapped it.
	if strings.HasPrefix(cleaned, "```") {
		if nl := strings.IndexByte(cleaned, '\n'); nl >= 0 {
			cleaned = cleaned[nl+1:]
		}
		if endFence := strings.LastIndex(cleaned, "```"); endFence >= 0 {
			cleaned = cleaned[:endFence]
		}
		cleaned = strings.TrimSpace(cleaned)
	}

	var parsed rawParsedToolCall
	if err := json.Unmarshal([]byte(cleaned), &parsed); err != nil {
		return ToolCallValue{}, false
	}
	name := strings.TrimSpace(parsed.Name)
	if name == "" {
		return ToolCallValue{}, false
	}

	argRaw := bytes.TrimSpace(parsed.Arguments)
	if len(argRaw) == 0 || string(argRaw) == "null" {
		argRaw = bytes.TrimSpace(parsed.Input)
	}

	argsJSON := "{}"
	if len(argRaw) > 0 && string(argRaw) != "null" {
		if argRaw[0] == '"' {
			// Model encoded arguments as a JSON string; unwrap if valid JSON inside.
			var innerStr string
			if err := json.Unmarshal(argRaw, &innerStr); err == nil {
				trimmedInner := strings.TrimSpace(innerStr)
				var check any
				if json.Unmarshal([]byte(trimmedInner), &check) == nil {
					argsJSON = trimmedInner
				} else {
					b, _ := json.Marshal(map[string]string{"value": innerStr})
					argsJSON = string(b)
				}
			}
		} else {
			var check any
			if json.Unmarshal(argRaw, &check) == nil {
				compacted, err := json.Marshal(check)
				if err == nil {
					argsJSON = string(compacted)
				} else {
					argsJSON = string(argRaw)
				}
			}
		}
	}

	id := strings.TrimSpace(parsed.ID)
	if id == "" {
		id = newToolCallID()
	}

	return ToolCallValue{
		ID:        id,
		Name:      name,
		Arguments: argsJSON,
	}, true
}

func newToolCallID() string {
	var b [6]byte
	if _, err := rand.Read(b[:]); err == nil {
		return "call_" + hex.EncodeToString(b[:])
	}
	return fmt.Sprintf("call_%d", toolSeq.Add(1))
}
