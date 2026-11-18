package bridge

import (
	"encoding/json"
	"fmt"
	"strings"

	"acp2api/internal/acp"
)

// BuildPromptBlocks constructs ACP ContentBlocks assuming full multimodal prompt capabilities.
func BuildPromptBlocks(req *UnifiedRequest, startIdx int) []acp.ContentBlock {
	return BuildPromptBlocksWithCaps(req, startIdx, acp.PromptCapabilities{
		Image:           true,
		Audio:           true,
		EmbeddedContext: true,
	}, "")
}

// BuildPromptBlocksWithCaps constructs the ACP ContentBlock slice for session/prompt while respecting
// the agent's negotiated PromptCapabilities (gracefully degrading unsupported rich media types).
func BuildPromptBlocksWithCaps(
	req *UnifiedRequest,
	startIdx int,
	caps acp.PromptCapabilities,
	thinkingPromptHint string,
) []acp.ContentBlock {
	var blocks []acp.ContentBlock
	var preamble strings.Builder

	if startIdx <= 0 {
		startIdx = 0
		if sys := strings.TrimSpace(req.System); sys != "" {
			preamble.WriteString("<system_instructions>\n")
			preamble.WriteString(sys)
			preamble.WriteString("\n</system_instructions>\n\n")
		}
		if len(req.Tools) > 0 && req.ToolChoice.Mode != "none" {
			preamble.WriteString(buildToolsPromptSection(req.Tools, req.ToolChoice))
			preamble.WriteString("\n\n")
		}
	} else if len(req.Tools) > 0 {
		if directive := buildTurnToolChoiceDirective(req.ToolChoice); directive != "" {
			preamble.WriteString(directive)
			preamble.WriteString("\n\n")
		}
	}

	if strings.TrimSpace(thinkingPromptHint) != "" {
		preamble.WriteString(strings.TrimSpace(thinkingPromptHint))
		preamble.WriteString("\n\n")
	}

	msgs := req.Messages[startIdx:]
	if len(msgs) == 0 {
		text := strings.TrimSpace(preamble.String())
		if text == "" {
			text = "Hello"
		}
		return []acp.ContentBlock{{Type: "text", Text: text}}
	}

	if len(msgs) > 1 {
		preamble.WriteString("<conversation_history>\n")
		for i := 0; i < len(msgs)-1; i++ {
			preamble.WriteString(formatMessageAsHistory(msgs[i]))
			preamble.WriteString("\n")
		}
		preamble.WriteString("</conversation_history>\n\n")
	}

	if preamble.Len() > 0 {
		blocks = append(blocks, acp.ContentBlock{
			Type: "text",
			Text: preamble.String(),
		})
	}

	lastMsg := msgs[len(msgs)-1]
	lastBlocks := formatActiveMessageBlocks(lastMsg, caps)
	blocks = append(blocks, lastBlocks...)

	if len(req.StopSequences) > 0 {
		blocks = append(blocks, acp.ContentBlock{
			Type: "text",
			Text: fmt.Sprintf("\n[System note: Stop generating immediately if you would output any of these sequences: %q]", req.StopSequences),
		})
	}

	return coalesceTextBlocks(blocks)
}

func buildToolsPromptSection(tools []ToolDefinition, choice ToolChoiceSpec) string {
	var sb strings.Builder
	sb.WriteString("<available_tools>\n")
	sb.WriteString("You have access to the following client-provided tools. ")
	sb.WriteString("When you need to invoke a tool, output ONLY the XML block(s) in the exact format below (you may provide brief explanation text before `<tool_call>` if appropriate, unless forced otherwise):\n")
	sb.WriteString("<tool_call>\n{\"name\": \"<tool_name>\", \"arguments\": {<json_arguments_object>}}\n</tool_call>\n")
	sb.WriteString("Wait for the user/system to return `<tool_response>` in the next turn before fabricating tool results.\n\n")
	sb.WriteString("Tool Definitions:\n")
	for _, t := range tools {
		schema := strings.TrimSpace(string(t.Parameters))
		if schema == "" {
			schema = `{"type":"object","properties":{}}`
		}
		entry := map[string]any{
			"name":        t.Name,
			"description": t.Description,
			"parameters":  json.RawMessage(schema),
		}
		b, err := json.Marshal(entry)
		if err != nil {
			sb.WriteString(fmt.Sprintf("- %s: %s\n", t.Name, t.Description))
			continue
		}
		sb.WriteString(string(b))
		sb.WriteString("\n")
	}
	if dir := buildTurnToolChoiceDirective(choice); dir != "" {
		sb.WriteString("\n")
		sb.WriteString(dir)
		sb.WriteString("\n")
	}
	sb.WriteString("</available_tools>")
	return sb.String()
}

func buildTurnToolChoiceDirective(choice ToolChoiceSpec) string {
	switch choice.Mode {
	case "none":
		return "[Tool Directive: Do NOT call any `<tool_call>` in this turn; respond with plain text only.]"
	case "required":
		return "[Tool Directive: You MUST invoke at least one `<tool_call>` in this response.]"
	case "tool":
		if choice.ToolName != "" {
			return fmt.Sprintf("[Tool Directive: You MUST invoke the tool %q using `<tool_call>` in this response.]", choice.ToolName)
		}
	}
	return ""
}

func formatMessageAsHistory(m UnifiedMessage) string {
	var sb strings.Builder
	roleLabel := "User"
	switch m.Role {
	case "assistant":
		roleLabel = "Assistant"
	case "tool":
		roleLabel = "Tool"
	case "system", "developer":
		roleLabel = "System"
	}
	sb.WriteString("[" + roleLabel + "]\n")
	for _, p := range m.Parts {
		switch p.Kind {
		case PartText:
			sb.WriteString(p.Text)
			sb.WriteString("\n")
		case PartImage:
			sb.WriteString("[Image attached]\n")
		case PartAudio:
			sb.WriteString("[Audio attached]\n")
		case PartResource:
			if p.Resource != nil && p.Resource.Text != "" {
				sb.WriteString(fmt.Sprintf("<resource uri=%q>\n%s\n</resource>\n", p.Resource.URI, p.Resource.Text))
			} else {
				sb.WriteString("[Resource attached]\n")
			}
		case PartToolCall:
			if p.ToolCall != nil {
				sb.WriteString(fmt.Sprintf("<tool_call>\n{\"id\": %q, \"name\": %q, \"arguments\": %s}\n</tool_call>\n",
					p.ToolCall.ID, p.ToolCall.Name, ensureJSONObjectString(p.ToolCall.Arguments)))
			}
		case PartToolResult:
			if p.ToolResult != nil {
				sb.WriteString(formatToolResultXML(p.ToolResult))
				sb.WriteString("\n")
			}
		}
	}
	return strings.TrimRight(sb.String(), "\n")
}

func formatActiveMessageBlocks(m UnifiedMessage, caps acp.PromptCapabilities) []acp.ContentBlock {
	var blocks []acp.ContentBlock
	for _, p := range m.Parts {
		switch p.Kind {
		case PartText:
			if p.Text != "" {
				blocks = append(blocks, acp.ContentBlock{
					Type: "text",
					Text: p.Text,
				})
			}
		case PartImage:
			if p.Image != nil {
				blocks = append(blocks, imagePayloadToBlock(*p.Image, caps.Image))
			}
		case PartAudio:
			if p.Audio != nil {
				blocks = append(blocks, audioPayloadToBlock(*p.Audio, caps.Audio))
			}
		case PartResource:
			if p.Resource != nil {
				blocks = append(blocks, resourcePayloadToBlock(*p.Resource, caps.EmbeddedContext))
			}
		case PartToolCall:
			if p.ToolCall != nil {
				blocks = append(blocks, acp.ContentBlock{
					Type: "text",
					Text: fmt.Sprintf("<tool_call>\n{\"id\": %q, \"name\": %q, \"arguments\": %s}\n</tool_call>\n",
						p.ToolCall.ID, p.ToolCall.Name, ensureJSONObjectString(p.ToolCall.Arguments)),
				})
			}
		case PartToolResult:
			if p.ToolResult != nil {
				blocks = append(blocks, acp.ContentBlock{
					Type: "text",
					Text: formatToolResultXML(p.ToolResult) + "\n",
				})
				for _, img := range p.ToolResult.Images {
					blocks = append(blocks, imagePayloadToBlock(img, caps.Image))
				}
			}
		}
	}
	if len(blocks) == 0 {
		blocks = append(blocks, acp.ContentBlock{Type: "text", Text: ""})
	}
	return blocks
}

func formatToolResultXML(tr *ToolResultVal) string {
	status := "success"
	if tr.IsError {
		status = "error"
	}
	if tr.Name != "" {
		return fmt.Sprintf("<tool_response tool_call_id=%q name=%q status=%q>\n%s\n</tool_response>",
			tr.ToolCallID, tr.Name, status, tr.Content)
	}
	return fmt.Sprintf("<tool_response tool_call_id=%q status=%q>\n%s\n</tool_response>",
		tr.ToolCallID, status, tr.Content)
}

func imagePayloadToBlock(img ImagePayload, imageSupported bool) acp.ContentBlock {
	mime := strings.TrimSpace(img.MimeType)
	if mime == "" {
		mime = "image/png"
	}
	if img.Data != "" && imageSupported {
		return acp.ContentBlock{
			Type:     "image",
			MimeType: mime,
			Data:     img.Data,
			URI:      img.URI,
		}
	}
	if img.URI != "" {
		return acp.ContentBlock{
			Type:     "resource_link",
			URI:      img.URI,
			Name:     "image",
			MimeType: mime,
		}
	}
	return acp.ContentBlock{
		Type: "text",
		Text: fmt.Sprintf("[Image attachment (%s) omitted: agent does not advertise image prompt capability]", mime),
	}
}

func audioPayloadToBlock(aud AudioPayload, audioSupported bool) acp.ContentBlock {
	mime := strings.TrimSpace(aud.MimeType)
	if mime == "" {
		mime = "audio/wav"
	}
	if aud.Data != "" && audioSupported {
		return acp.ContentBlock{
			Type:     "audio",
			MimeType: mime,
			Data:     aud.Data,
		}
	}
	return acp.ContentBlock{
		Type: "text",
		Text: fmt.Sprintf("[Audio attachment (%s) omitted: agent does not advertise audio prompt capability]", mime),
	}
}

func resourcePayloadToBlock(res ResourcePayload, embeddedSupported bool) acp.ContentBlock {
	uri := strings.TrimSpace(res.URI)
	if uri == "" {
		if res.Name != "" {
			uri = "file:///" + res.Name
		} else {
			uri = "file:///document"
		}
	}
	if embeddedSupported && (res.Text != "" || res.Blob != "") {
		return acp.ContentBlock{
			Type: "resource",
			Resource: &acp.EmbeddedResource{
				URI:      uri,
				MimeType: res.MimeType,
				Text:     res.Text,
				Blob:     res.Blob,
			},
		}
	}
	if res.Text != "" {
		return acp.ContentBlock{
			Type: "text",
			Text: fmt.Sprintf("<attached_resource uri=%q>\n%s\n</attached_resource>", uri, res.Text),
		}
	}
	name := res.Name
	if name == "" {
		name = uri
	}
	return acp.ContentBlock{
		Type:     "resource_link",
		URI:      uri,
		Name:     name,
		MimeType: res.MimeType,
	}
}

func coalesceTextBlocks(in []acp.ContentBlock) []acp.ContentBlock {
	if len(in) <= 1 {
		return in
	}
	out := make([]acp.ContentBlock, 0, len(in))
	for _, b := range in {
		if b.Type == "text" && len(out) > 0 && out[len(out)-1].Type == "text" {
			out[len(out)-1].Text += b.Text
			continue
		}
		out = append(out, b)
	}
	return out
}

func ensureJSONObjectString(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return "{}"
	}
	var js json.RawMessage
	if json.Unmarshal([]byte(trimmed), &js) == nil {
		return trimmed
	}
	b, _ := json.Marshal(map[string]string{"raw": raw})
	return string(b)
}
