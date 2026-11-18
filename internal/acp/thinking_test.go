package acp

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"acp2api/internal/config"
)

func TestInspectAndResolveThinkingAcrossStandardAndNonStandardACP(t *testing.T) {
	thinkCfg := config.Default().ACP.Thinking

	// Case 1: Standard category="thought_level" with tier clamping (desired "xhigh" -> closest "high")
	var stdMeta SessionNewResult
	if err := json.Unmarshal([]byte(`{
		"sessionId": "s1",
		"configOptions": [{
			"id": "thought_level",
			"category": "thought_level",
			"currentValue": "medium",
			"options": [
				{"value": "low", "name": "Low"},
				{"value": "medium", "name": "Medium"},
				{"value": "high", "name": "High"}
			]
		}]
	}`), &stdMeta); err != nil {
		t.Fatalf("unmarshal stdMeta: %v", err)
	}
	insp1 := InspectSessionThinking(stdMeta)
	if !insp1.Detected || insp1.Source != "standard_config_option" || insp1.CurrentLevel != "medium" {
		t.Fatalf("unexpected std inspection: %+v", insp1)
	}
	res1, err := ResolveThinkingAction(stdMeta, NormalizeThinkingIntent(nil, "xhigh", 0), "", thinkCfg)
	if err != nil || res1.Strategy != "config_option" || res1.ConfigValue != "high" {
		t.Fatalf("expected xhigh clamped to high, got %+v err=%v", res1, err)
	}

	// Case 2: Non-standard ACP server returning numeric currentValue, scalar options array, and custom id="thinking_budget"
	var budgetMeta SessionNewResult
	if err := json.Unmarshal([]byte(`{
		"sessionId": "s2",
		"configOptions": [{
			"id": "thinking_budget",
			"type": "number",
			"currentValue": 4096,
			"values": [1024, 4096, 16384, 32000]
		}]
	}`), &budgetMeta); err != nil {
		t.Fatalf("unmarshal budgetMeta: %v", err)
	}
	insp2 := InspectSessionThinking(budgetMeta)
	if !insp2.Detected || insp2.Source != "heuristic_config_option" || insp2.CurrentLevel != "4096" {
		t.Fatalf("unexpected budget inspection: %+v", insp2)
	}
	tVal := true
	res2, err := ResolveThinkingAction(budgetMeta, NormalizeThinkingIntent(&tVal, "", 15000), "", thinkCfg)
	if err != nil || res2.Strategy != "heuristic_config_option" || res2.ConfigValue != 16384 {
		t.Fatalf("expected budget 15000 matched to numeric 16384, got %+v err=%v", res2, err)
	}

	// Case 3: Non-standard boolean configOption (currentValue: true, type: "boolean")
	var boolMeta SessionNewResult
	_ = json.Unmarshal([]byte(`{
		"sessionId": "s3",
		"configOptions": [{
			"id": "thinking",
			"type": "boolean",
			"currentValue": false
		}]
	}`), &boolMeta)
	res3, err := ResolveThinkingAction(boolMeta, NormalizeThinkingIntent(&tVal, "high", 0), "", thinkCfg)
	if err != nil || res3.ConfigID != "thinking" || res3.ConfigValue != true {
		t.Fatalf("expected boolean true configValue, got %+v err=%v", res3, err)
	}

	// Case 4: Opaque config ID recognized purely by effort-shaped {id, label} choices!
	var opaqueIDMeta SessionNewResult
	_ = json.Unmarshal([]byte(`{
		"sessionId": "s4",
		"configOptions": [{
			"id": "cfg_99",
			"options": [{"id": "low", "label": "Fast"}, {"id": "medium", "label": "Balanced"}, {"id": "high", "label": "Deep"}]
		}]
	}`), &opaqueIDMeta)
	res4, err := ResolveThinkingAction(opaqueIDMeta, NormalizeThinkingIntent(nil, "minimal", 0), "", thinkCfg)
	if err != nil || res4.ConfigID != "cfg_99" || res4.ConfigValue != "low" {
		t.Fatalf("expected minimal clamped to low on opaque cfg_99, got %+v err=%v", res4, err)
	}

	// Case 5: Model Variant fallback (no configOptions, but models.availableModels has <model>-high)
	var modelVarMeta SessionNewResult
	_ = json.Unmarshal([]byte(`{
		"sessionId": "s5",
		"models": {
			"currentModelId": "sonnet",
			"availableModels": [
				{"modelId": "sonnet", "name": "Sonnet"},
				{"modelId": "sonnet-high", "name": "Sonnet High"}
			]
		}
	}`), &modelVarMeta)
	res5, err := ResolveThinkingAction(modelVarMeta, NormalizeThinkingIntent(nil, "high", 0), "sonnet", thinkCfg)
	if err != nil || res5.Strategy != "model_variant" || res5.ModelOverride != "sonnet-high" {
		t.Fatalf("expected model_variant sonnet-high, got %+v err=%v", res5, err)
	}

	// Case 6: Mode fallback + Top-level / _meta non-standard inspection
	var modeMeta SessionNewResult
	_ = json.Unmarshal([]byte(`{
		"sessionId": "s6",
		"modes": {
			"currentModeId": "chat",
			"availableModes": [{"id": "chat", "name": "Chat"}, {"id": "thinking", "name": "Deep Thinking"}]
		}
	}`), &modeMeta)
	res6, err := ResolveThinkingAction(modeMeta, NormalizeThinkingIntent(nil, "high", 0), "", thinkCfg)
	if err != nil || res6.Strategy != "mode" || res6.ModeOverride != "thinking" {
		t.Fatalf("expected mode fallback 'thinking', got %+v err=%v", res6, err)
	}

	var topLevelMeta SessionNewResult
	_ = json.Unmarshal([]byte(`{
		"sessionId": "s6b",
		"thinking": {"level": "high", "availableLevels": ["low", "high"]}
	}`), &topLevelMeta)
	if insp := InspectSessionThinking(topLevelMeta); !insp.Detected || insp.CurrentLevel != "high" || len(insp.AvailableChoices) != 2 {
		t.Fatalf("expected top-level thinking object detected, got %+v", insp)
	}

	// Case 7: Completely opaque ACP server (empty SessionNewResult) across all FallbackStrategies!
	opaqueMeta := SessionNewResult{SessionID: "s_opaque"}
	if InspectSessionThinking(opaqueMeta).Detected {
		t.Fatalf("expected Detected=false on empty session metadata")
	}

	// 7a. auto -> fallback_meta
	resAuto, err := ResolveThinkingAction(opaqueMeta, NormalizeThinkingIntent(nil, "high", 16000), "", thinkCfg)
	if err != nil || resAuto.Strategy != "fallback_meta" || resAuto.MetaHint["thoughtLevel"] != "high" {
		t.Fatalf("expected fallback_meta, got %+v err=%v", resAuto, err)
	}

	// 7b. prompt_hint -> fallback_prompt_hint with <thinking_guidance>
	promptCfg := thinkCfg
	promptCfg.FallbackStrategy = config.ThinkingFallbackPromptHint
	resPrompt, err := ResolveThinkingAction(opaqueMeta, NormalizeThinkingIntent(nil, "high", 0), "", promptCfg)
	if err != nil || resPrompt.Strategy != "fallback_prompt_hint" || !strings.Contains(resPrompt.PromptHint, "<thinking_guidance>") {
		t.Fatalf("expected fallback_prompt_hint, got %+v err=%v", resPrompt, err)
	}

	// 7c. override_config_id -> override_config_option
	overrideCfg := thinkCfg
	overrideCfg.OverrideConfigID = "custom_effort"
	resOverride, err := ResolveThinkingAction(opaqueMeta, NormalizeThinkingIntent(nil, "low", 0), "", overrideCfg)
	if err != nil || resOverride.Strategy != "override_config_option" || resOverride.ConfigID != "custom_effort" {
		t.Fatalf("expected override_config_option, got %+v err=%v", resOverride, err)
	}

	// 7d. strict_error -> ErrUnsupportedThinkingLevel
	strictCfg := thinkCfg
	strictCfg.FallbackStrategy = config.ThinkingFallbackStrictError
	if _, err := ResolveThinkingAction(opaqueMeta, NormalizeThinkingIntent(nil, "high", 0), "", strictCfg); !errors.Is(err, ErrUnsupportedThinkingLevel) {
		t.Fatalf("expected ErrUnsupportedThinkingLevel in strict_error mode, got: %v", err)
	}
}
