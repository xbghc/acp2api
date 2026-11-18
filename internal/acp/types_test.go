package acp

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRPCErrorFormatting(t *testing.T) {
	e1 := &RPCError{Code: -32601, Message: "Method not found"}
	if got := e1.Error(); got != "acp rpc error -32601: Method not found" {
		t.Fatalf("unexpected error string: %q", got)
	}

	e2 := &RPCError{Code: -32602, Message: "Invalid params", Data: json.RawMessage(`{"field":"cwd"}`)}
	if got := e2.Error(); !strings.Contains(got, `{"field":"cwd"}`) {
		t.Fatalf("expected data in error string: %q", got)
	}
}

func TestExtractChoiceValuesFlatAndGrouped(t *testing.T) {
	flatOpt := SessionConfigOption{
		ID:      "model",
		Options: json.RawMessage(`[{"value":"m1","name":"Model 1"},{"value":"m2","name":"Model 2"}]`),
	}
	choices := flatOpt.ExtractChoiceValues()
	if len(choices) != 2 || choices[0].Value != "m1" || choices[1].Value != "m2" {
		t.Fatalf("unexpected flat choices: %+v", choices)
	}

	groupedOpt := SessionConfigOption{
		ID: "model",
		Options: json.RawMessage(`[
			{"group":"Fast","options":[{"value":"fast-1","name":"Fast 1"}]},
			{"group":"Pro","options":[{"value":"pro-1","name":"Pro 1"},{"value":"pro-2","name":"Pro 2"}]}
		]`),
	}
	gChoices := groupedOpt.ExtractChoiceValues()
	if len(gChoices) != 3 || gChoices[2].Value != "pro-2" {
		t.Fatalf("unexpected grouped choices: %+v", gChoices)
	}

	emptyOpt := SessionConfigOption{ID: "mode"}
	if emptyOpt.ExtractChoiceValues() != nil {
		t.Fatalf("expected nil for empty options")
	}

	badOpt := SessionConfigOption{ID: "mode", Options: json.RawMessage(`"invalid"`)}
	if badOpt.ExtractChoiceValues() != nil {
		t.Fatalf("expected nil for non-array options")
	}
}

func TestParseSessionUpdateVariants(t *testing.T) {
	// 1. Single object content block
	u1, err := ParseSessionUpdate(json.RawMessage(`{
		"sessionUpdate": "agent_message_chunk",
		"content": {"type": "text", "text": "hello "}
	}`))
	if err != nil || u1.Kind != "agent_message_chunk" || u1.Text != "hello " {
		t.Fatalf("unexpected u1: %+v err=%v", u1, err)
	}

	// 2. Array of content blocks
	u2, err := ParseSessionUpdate(json.RawMessage(`{
		"sessionUpdate": "agent_thought_chunk",
		"content": [{"type": "text", "text": "part1 "}, {"type": "image", "data": "abc"}, {"type": "text", "text": "part2"}]
	}`))
	if err != nil || u2.Kind != "agent_thought_chunk" || u2.Text != "part1 part2" {
		t.Fatalf("unexpected u2: %+v err=%v", u2, err)
	}

	// 3. String content & direct text fallback
	u3, err := ParseSessionUpdate(json.RawMessage(`{
		"sessionUpdate": "agent_message_chunk",
		"content": "plain string"
	}`))
	if err != nil || u3.Text != "plain string" {
		t.Fatalf("unexpected u3: %+v err=%v", u3, err)
	}

	u4, err := ParseSessionUpdate(json.RawMessage(`{
		"sessionUpdate": "agent_message_chunk",
		"text": "direct text"
	}`))
	if err != nil || u4.Text != "direct text" {
		t.Fatalf("unexpected u4: %+v err=%v", u4, err)
	}

	// 4. tool_call update with name fallback and usage_update
	u5, err := ParseSessionUpdate(json.RawMessage(`{
		"sessionUpdate": "tool_call",
		"toolCallId": "tc_1",
		"name": "bash",
		"kind": "execute",
		"status": "in_progress"
	}`))
	if err != nil || u5.ToolCallID != "tc_1" || u5.ToolTitle != "bash" || u5.ToolKind != "execute" || u5.ToolStatus != "in_progress" {
		t.Fatalf("unexpected u5: %+v err=%v", u5, err)
	}

	u6, err := ParseSessionUpdate(json.RawMessage(`{
		"sessionUpdate": "usage_update",
		"usage": {"inputTokens": 12, "outputTokens": 34}
	}`))
	if err != nil || u6.Usage == nil || u6.Usage.InputTokens != 12 || u6.Usage.OutputTokens != 34 {
		t.Fatalf("unexpected u6: %+v err=%v", u6, err)
	}

	// 5. Invalid JSON
	if _, err := ParseSessionUpdate(json.RawMessage(`{bad`)); err == nil {
		t.Fatalf("expected error on invalid JSON")
	}
}
