package bridge

import "strings"

type thinkParserState int

const (
	thinkStateDetectStart thinkParserState = iota
	thinkStateInside
	thinkStateTrimAfterClose
	thinkStatePassThrough
)

// InlineThinkParser extracts leading <think>...</think> or <thinking>...</thinking> blocks
// from agent_message_chunk streams when an ACP server does not use native agent_thought_chunk notifications.
type InlineThinkParser struct {
	enabled    bool
	state      thinkParserState
	trimOpenNL bool
	closingTag string
	buf        string
	onThink    func(string)
	onText     func(string)
}

func NewInlineThinkParser(enabled bool, onThink func(string), onText func(string)) *InlineThinkParser {
	st := thinkStateDetectStart
	if !enabled {
		st = thinkStatePassThrough
	}
	return &InlineThinkParser{
		enabled: enabled,
		state:   st,
		onThink: onThink,
		onText:  onText,
	}
}

// MarkNativeThoughtSeen disables inline <think> tag interception because the ACP server
// is already emitting native agent_thought_chunk notifications.
func (p *InlineThinkParser) MarkNativeThoughtSeen() {
	if p.state == thinkStateDetectStart {
		if p.buf != "" {
			b := p.buf
			p.buf = ""
			p.state = thinkStatePassThrough
			p.onText(b)
			return
		}
		p.state = thinkStatePassThrough
	}
}

// Feed processes an incoming agent_message_chunk text delta.
func (p *InlineThinkParser) Feed(chunk string) {
	if chunk == "" {
		return
	}
	if p.state == thinkStatePassThrough {
		p.onText(chunk)
		return
	}

	if p.trimOpenNL && p.buf == "" {
		if strings.HasPrefix(chunk, "\r\n") {
			chunk = chunk[2:]
			p.trimOpenNL = false
		} else if chunk == "\r" {
			return
		} else if strings.HasPrefix(chunk, "\n") {
			chunk = chunk[1:]
			p.trimOpenNL = false
		} else {
			p.trimOpenNL = false
		}
		if chunk == "" {
			return
		}
	}

	p.buf += chunk
	for len(p.buf) > 0 {
		switch p.state {
		case thinkStateDetectStart:
			trimmed := strings.TrimLeft(p.buf, " \t\r\n")
			if trimmed == "" {
				if len(p.buf) > 32 {
					b := p.buf
					p.buf = ""
					p.state = thinkStatePassThrough
					p.onText(b)
				}
				return
			}
			if strings.HasPrefix(trimmed, "<thinking>") {
				rest := trimmed[len("<thinking>"):]
				if rest == "" {
					p.trimOpenNL = true
				} else {
					rest = trimSingleLeadingNewline(rest)
				}
				p.buf = rest
				p.closingTag = "</thinking>"
				p.state = thinkStateInside
				continue
			}
			if strings.HasPrefix(trimmed, "<think>") {
				rest := trimmed[len("<think>"):]
				if rest == "" {
					p.trimOpenNL = true
				} else {
					rest = trimSingleLeadingNewline(rest)
				}
				p.buf = rest
				p.closingTag = "</think>"
				p.state = thinkStateInside
				continue
			}
			if strings.HasPrefix("<thinking>", trimmed) || strings.HasPrefix("<think>", trimmed) {
				return
			}
			b := p.buf
			p.buf = ""
			p.state = thinkStatePassThrough
			p.onText(b)
			return

		case thinkStateInside:
			endIdx := strings.Index(p.buf, p.closingTag)
			if endIdx >= 0 {
				thought := p.buf[:endIdx]
				if thought != "" {
					p.onThink(thought)
				}
				p.buf = p.buf[endIdx+len(p.closingTag):]
				p.state = thinkStateTrimAfterClose
				continue
			}
			keep := longestSuffixPrefix(p.buf, p.closingTag)
			flushLen := len(p.buf) - keep
			if flushLen > 0 {
				p.onThink(p.buf[:flushLen])
				p.buf = p.buf[flushLen:]
			}
			return

		case thinkStateTrimAfterClose:
			trimmed := strings.TrimLeft(p.buf, "\r\n")
			if trimmed == "" {
				p.buf = ""
				return
			}
			p.buf = ""
			p.state = thinkStatePassThrough
			p.onText(trimmed)
			return

		case thinkStatePassThrough:
			b := p.buf
			p.buf = ""
			p.onText(b)
			return
		}
	}
}

func (p *InlineThinkParser) Finish() {
	if p.buf == "" {
		return
	}
	b := p.buf
	p.buf = ""
	if p.state == thinkStateInside {
		p.onThink(b)
	} else if p.state != thinkStateTrimAfterClose {
		p.onText(b)
	}
	p.state = thinkStatePassThrough
}

func trimSingleLeadingNewline(s string) string {
	if strings.HasPrefix(s, "\r\n") {
		return s[2:]
	}
	if strings.HasPrefix(s, "\n") {
		return s[1:]
	}
	return s
}
