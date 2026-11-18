package bridge

import (
	"encoding/json"
	"fmt"
	"strings"

	"acp2api/internal/acp"
)

// BuildPromptBlocks constructs the ACP ContentBlock slice for session/prompt.
// When startIdx == 0, the target session is brand-new (pre-warmed idle session), so system instructions,
// tool definitions, and any prior history are included.
// When startIdx > 0, the target session already holds req.Messages[:startIdx], so only delta messages
// req.Messages[startIdx:] are sent.
func BuildPromptBlocks(req *UnifiedRequest, startIdx int) []acp.ContentBlock {
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
		// On incremental turns, only add a lightweight directive if tool_choice forces specific behavior.
		if directive := buildTurnToolChoiceDirective(req.ToolChoice); directive != "" {
			preamble.WriteString(directive)
			preamble.WriteString("\n\n")
		}
	}

	msgs := req.Messages[startIdx:]
	if len(msgs) == 0 {
		text := strings.TrimSpace(preamble.String())
		if text == "" {
			text = "Hello"
		}
		return []acp.ContentBlock{{Type: "text", Text: text}}
	}

	// If there are multiple messages to send (e.g. replaying prior history on a fresh session, or multiple delta messages),
	// format messages[:len-1] as structured conversation context and messages[len-1] as the active turn.
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
	lastBlocks := formatActiveMessageBlocks(lastMsg)
	blocks = append(blocks, lastBlocks...)

	if len(req.StopSequences) > 0 {
		blocks = append(blocks, acp.ContentBlock{
			Type: "text",
			Text: fmt.Sprintf("\n[System note: Stop generating immediately if you would output any of these sequences: %q]", req.StopSequences),
		})
	}

	// Merge adjacent text blocks so ACP servers receive clean, compact prompt arrays.
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

func formatActiveMessageBlocks(m UnifiedMessage) []acp.ContentBlock {
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
				blocks = append(blocks, imagePayloadToBlock(*p.Image))
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
					blocks = append(blocks, imagePayloadToBlock(img))
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

func imagePayloadToBlock(img ImagePayload) acp.ContentBlock {
	mime := strings.TrimSpace(img.MimeType)
	if mime == "" {
		mime = "image/png"
	}
	if img.Data != "" {
		return acp.ContentBlock{
			Type:     "image",
			MimeType: mime,
			Data:     img.Data,
			URI:      img.URI,
		}
	}
	return acp.ContentBlock{
		Type: "text",
		Text: fmt.Sprintf("[Image URL: %s]", img.URI),
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
