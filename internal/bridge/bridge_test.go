package bridge

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestToolStreamParserChunkedFragmentation(t *testing.T) {
	fullStream := "Let me check the weather for you.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Shanghai\", \"unit\": \"celsius\"}}\r\n</tool_call>\r\n<tool_call>```json\n{\"name\":\"get_time\",\"arguments\":\"{\\\"tz\\\":\\\"Asia/Shanghai\\\"}\"}\n```</tool_call>Done!"

	var events []StreamEvent
	parser := NewToolStreamParser(true, func(ev StreamEvent) {
		events = append(events, ev)
	})

	// Feed 1 byte at a time to stress-test every single tag & CRLF boundary!
	for i := 0; i < len(fullStream); i++ {
		parser.Feed(fullStream[i : i+1])
	}
	parser.Finish()

	var combinedText strings.Builder
	var calls []ToolCallValue
	for _, ev := range events {
		switch ev.Kind {
		case EventTextDelta:
			combinedText.WriteString(ev.Text)
		case EventToolCall:
			if ev.ToolCall != nil {
				calls = append(calls, *ev.ToolCall)
			}
		}
	}

	if got := combinedText.String(); got != "Let me check the weather for you.\nDone!" {
		t.Fatalf("unexpected text output: %q", got)
	}
	if len(calls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d: %+v", len(calls), calls)
	}
	if calls[0].Name != "get_weather" || calls[0].Arguments != `{"city":"Shanghai","unit":"celsius"}` {
		t.Fatalf("unexpected first tool call: %+v", calls[0])
	}
	if calls[1].Name != "get_time" || calls[1].Arguments != `{"tz":"Asia/Shanghai"}` {
		t.Fatalf("unexpected second tool call: %+v", calls[1])
	}
}

func TestToolStreamParserDisabledMalformedAndEOFFlush(t *testing.T) {
	// 1. Disabled parser passes <tool_call> through untouched.
	var passthrough strings.Builder
	pDisabled := NewToolStreamParser(false, func(ev StreamEvent) {
		passthrough.WriteString(ev.Text)
	})
	pDisabled.Feed("")
	pDisabled.Feed("<tool_call>{\"name\":\"x\"}</tool_call>")
	pDisabled.Finish()
	if passthrough.String() != "<tool_call>{\"name\":\"x\"}</tool_call>" {
		t.Fatalf("expected disabled passthrough, got %q", passthrough.String())
	}

	// 2. Malformed JSON inside <tool_call> + trailing partial <tool_ at EOF
	var textOut strings.Builder
	pMalformed := NewToolStreamParser(true, func(ev StreamEvent) {
		if ev.Kind == EventTextDelta {
			textOut.WriteString(ev.Text)
		}
	})
	pMalformed.Feed("<tool_call>not valid json</tool_call> and <tool_")
	pMalformed.Finish()
	if textOut.String() != "<tool_call>not valid json</tool_call> and <tool_" {
		t.Fatalf("expected malformed and trailing partial tag preserved, got %q", textOut.String())
	}

	// 3. Unclosed <tool_call> at EOF with valid JSON (using Anthropic "input" key & plain string arg) recovers cleanly!
	var recovered []ToolCallValue
	pUnclosedOK := NewToolStreamParser(true, func(ev StreamEvent) {
		if ev.Kind == EventToolCall && ev.ToolCall != nil {
			recovered = append(recovered, *ev.ToolCall)
		}
	})
	pUnclosedOK.Feed(`<tool_call>{"id":"custom_1","name":"search","input":"plain_query"}`)
	pUnclosedOK.Finish()
	if len(recovered) != 1 || recovered[0].ID != "custom_1" || recovered[0].Name != "search" || recovered[0].Arguments != `{"value":"plain_query"}` {
		t.Fatalf("expected recovered unclosed tool call, got %+v", recovered)
	}

	// 4. Unclosed <tool_call> at EOF with invalid JSON flushes raw text.
	var unclosedBad strings.Builder
	pUnclosedBad := NewToolStreamParser(true, func(ev StreamEvent) {
		unclosedBad.WriteString(ev.Text)
	})
	pUnclosedBad.Feed(`<tool_call>{broken`)
	pUnclosedBad.Finish()
	if unclosedBad.String() != `<tool_call>{broken` {
		t.Fatalf("expected broken unclosed tag flushed, got %q", unclosedBad.String())
	}
}

func TestConversationHashNormalizationAndAssistantMessage(t *testing.T) {
	tr := &TurnResponse{
		Thinking: "step 1",
		Text:     "Here are two calls.",
		ToolCalls: []ToolCallValue{
			{ID: "c1", Name: "fn1", Arguments: `{"a": 1}`},
			{ID: "c2", Name: "fn2", Arguments: `{"b": 2}`},
		},
	}
	asstMsg := tr.AssistantMessage()
	if asstMsg.Role != "assistant" || len(asstMsg.Parts) != 4 {
		t.Fatalf("unexpected AssistantMessage: %+v", asstMsg)
	}

	tools := []ToolDefinition{
		{Name: "calc", Description: "add", Parameters: json.RawMessage(`{ "type" : "object" }`)},
	}

	// OpenAI style: two consecutive role:"tool" messages + assistant thinking included
	openAIHistory := []UnifiedMessage{
		{Role: "developer", Parts: []ContentPart{{Kind: PartText, Text: "extra rule"}}},
		{Role: "user", Parts: []ContentPart{
			{Kind: PartText, Text: "check img"},
			{Kind: PartImage, Image: &ImagePayload{MimeType: "image/png", Data: "AAAA"}},
		}},
		asstMsg,
		{Role: "tool", Parts: []ContentPart{{Kind: PartToolResult, ToolResult: &ToolResultVal{ToolCallID: "c1", Content: "ok1"}}}},
		{Role: "tool", Parts: []ContentPart{{Kind: PartToolResult, ToolResult: &ToolResultVal{ToolCallID: "c2", Content: "err2", IsError: true}}}},
	}

	// Anthropic style: single role:"user" message containing two PartToolResult blocks + thinking omitted
	anthropicHistory := []UnifiedMessage{
		{Role: "system", Parts: []ContentPart{{Kind: PartText, Text: "extra rule"}}},
		{Role: "user", Parts: []ContentPart{
			{Kind: PartText, Text: "check img  \n"},
			{Kind: PartImage, Image: &ImagePayload{MimeType: "image/png", Data: "AAAA"}},
		}},
		{Role: "assistant", Parts: []ContentPart{
			{Kind: PartText, Text: "Here are two calls."},
			{Kind: PartToolCall, ToolCall: &ToolCallValue{ID: "c1", Name: "fn1", Arguments: `{"a":1}`}},
			{Kind: PartToolCall, ToolCall: &ToolCallValue{ID: "c2", Name: "fn2", Arguments: `{"b":2}`}},
		}},
		{Role: "user", Parts: []ContentPart{
			{Kind: PartToolResult, ToolResult: &ToolResultVal{ToolCallID: "c1", Content: "ok1"}},
			{Kind: PartToolResult, ToolResult: &ToolResultVal{ToolCallID: "c2", Content: "err2", IsError: true}},
		}},
	}

	h1 := ComputeConversationHash("m1", "sys", tools, openAIHistory)
	h2 := ComputeConversationHash("m1", "sys", tools, anthropicHistory)
	if h1 != h2 {
		t.Fatalf("expected OpenAI and Anthropic equivalent histories to produce identical hash: %s != %s", h1, h2)
	}
}

func TestBuildPromptBlocksFullFeaturesAndTokenEstimation(t *testing.T) {
	req := &UnifiedRequest{
		System: "System rule.",
		Tools: []ToolDefinition{
			{Name: "lookup", Description: "Search docs"},
		},
		ToolChoice:    ToolChoiceSpec{Mode: "tool", ToolName: "lookup"},
		StopSequences: []string{"</END>"},
		Messages: []UnifiedMessage{
			{Role: "system", Parts: []ContentPart{{Kind: PartText, Text: "inline sys"}}},
			{Role: "user", Parts: []ContentPart{
				{Kind: PartText, Text: "你好世界 Hello"},
				{Kind: PartImage, Image: &ImagePayload{URI: "https://example.com/a.png"}},
			}},
			{Role: "assistant", Parts: []ContentPart{
				{Kind: PartThinking, Thinking: "let me call lookup"},
				{Kind: PartToolCall, ToolCall: &ToolCallValue{ID: "tc1", Name: "lookup", Arguments: "not-json"}},
			}},
			{Role: "tool", Parts: []ContentPart{
				{Kind: PartToolResult, ToolResult: &ToolResultVal{
					ToolCallID: "tc1",
					Name:       "lookup",
					IsError:    true,
					Content:    "timeout",
					Images:     []ImagePayload{{MimeType: "image/png", Data: "iVBORw0KGgo="}},
				}},
			}},
		},
	}

	est := EstimateRequestTokens(req)
	if est < 256 {
		t.Fatalf("expected token estimate to account for CJK, tools, and image (>=256), got %d", est)
	}
	if EstimateTextTokens("") != 0 {
		t.Fatalf("expected 0 tokens for empty text")
	}

	freshBlocks := BuildPromptBlocks(req, 0)
	if len(freshBlocks) != 3 {
		t.Fatalf("expected [text, image, stop_note] blocks after coalescing, got %d: %+v", len(freshBlocks), freshBlocks)
	}
	if !strings.Contains(freshBlocks[0].Text, "<available_tools>") ||
		!strings.Contains(freshBlocks[0].Text, `MUST invoke the tool "lookup"`) ||
		!strings.Contains(freshBlocks[0].Text, `status="error"`) {
		t.Fatalf("missing expected sections in fresh prompt block:\n%s", freshBlocks[0].Text)
	}
	if freshBlocks[1].Type != "image" || freshBlocks[1].Data != "iVBORw0KGgo=" {
		t.Fatalf("expected embedded base64 image block, got %+v", freshBlocks[1])
	}

	// Incremental turn with required tool_choice directive & empty messages fallback
	req.ToolChoice = ToolChoiceSpec{Mode: "required"}
	incBlocks := BuildPromptBlocks(req, 3)
	if !strings.Contains(incBlocks[0].Text, "MUST invoke at least one") {
		t.Fatalf("expected incremental required directive, got: %s", incBlocks[0].Text)
	}

	emptyBlocks := BuildPromptBlocks(&UnifiedRequest{}, 0)
	if len(emptyBlocks) != 1 || emptyBlocks[0].Text != "Hello" {
		t.Fatalf("expected fallback Hello block for empty request, got %+v", emptyBlocks)
	}
}
