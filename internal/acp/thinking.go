package acp

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	"acp2api/internal/config"
)

// ErrUnsupportedThinkingLevel is returned only when FallbackStrategy == ThinkingFallbackStrictError
// and the underlying ACP agent offers no mechanism to apply the requested thinking level.
var ErrUnsupportedThinkingLevel = errors.New("requested thinking level is not supported by the underlying ACP agent")

// DesiredThinking represents normalized client thinking/reasoning intent from OpenAI or Anthropic requests.
type DesiredThinking struct {
	Specified    bool
	Enabled      bool
	Level        string // "none", "minimal", "low", "medium", "high", "xhigh"
	BudgetTokens int    // optional explicit token budget (e.g. Anthropic budget_tokens)
}

// NormalizeThinkingIntent converts raw effort strings or token budgets into a canonical DesiredThinking.
func NormalizeThinkingIntent(enabled *bool, rawEffort string, budgetTokens int) DesiredThinking {
	effort := strings.ToLower(strings.TrimSpace(rawEffort))
	if effort == "" && enabled == nil && budgetTokens == 0 {
		return DesiredThinking{}
	}

	if enabled != nil && !*enabled {
		return DesiredThinking{Specified: true, Enabled: false, Level: "none"}
	}

	switch effort {
	case "none", "off", "disabled", "false", "0":
		return DesiredThinking{Specified: true, Enabled: false, Level: "none"}
	case "minimal", "min":
		return DesiredThinking{Specified: true, Enabled: true, Level: "minimal", BudgetTokens: defaultBudgetForTier("minimal", budgetTokens)}
	case "low":
		return DesiredThinking{Specified: true, Enabled: true, Level: "low", BudgetTokens: defaultBudgetForTier("low", budgetTokens)}
	case "medium", "med", "adaptive", "auto", "thinking", "enabled", "true", "on":
		lvl := "medium"
		if budgetTokens > 0 {
			lvl = budgetToTier(budgetTokens)
		}
		return DesiredThinking{Specified: true, Enabled: true, Level: lvl, BudgetTokens: defaultBudgetForTier(lvl, budgetTokens)}
	case "high":
		return DesiredThinking{Specified: true, Enabled: true, Level: "high", BudgetTokens: defaultBudgetForTier("high", budgetTokens)}
	case "xhigh", "max", "ultra":
		return DesiredThinking{Specified: true, Enabled: true, Level: "xhigh", BudgetTokens: defaultBudgetForTier("xhigh", budgetTokens)}
	}

	if budgetTokens > 0 {
		lvl := budgetToTier(budgetTokens)
		return DesiredThinking{Specified: true, Enabled: true, Level: lvl, BudgetTokens: budgetTokens}
	}
	if enabled != nil && *enabled {
		return DesiredThinking{Specified: true, Enabled: true, Level: "medium", BudgetTokens: 8192}
	}
	if effort != "" {
		// Pass custom effort name while marking enabled.
		return DesiredThinking{Specified: true, Enabled: true, Level: effort, BudgetTokens: defaultBudgetForTier("medium", budgetTokens)}
	}
	return DesiredThinking{}
}

func budgetToTier(budget int) string {
	switch {
	case budget <= 0:
		return "none"
	case budget <= 2048:
		return "low"
	case budget <= 10240:
		return "medium"
	case budget <= 24576:
		return "high"
	default:
		return "xhigh"
	}
}

func defaultBudgetForTier(tier string, explicit int) int {
	if explicit > 0 {
		return explicit
	}
	switch tier {
	case "minimal":
		return 1024
	case "low":
		return 2048
	case "medium":
		return 8192
	case "high":
		return 16384
	case "xhigh":
		return 31999
	default:
		return 0
	}
}

// ThinkingInspection summarizes what thinking level state & capabilities could be extracted from a session.
type ThinkingInspection struct {
	Detected         bool
	Source           string // "standard_config_option", "heuristic_config_option", "top_level_field", "meta_field", "model_variants", "modes", "none"
	ConfigID         string
	CurrentLevel     string
	AvailableChoices []string
}

// InspectSessionThinking extracts current and available thinking levels from both standard ACP configOptions
// and non-standard ACP session responses (custom config IDs, top-level fields, _meta, model variants, or modes).
func InspectSessionThinking(meta SessionNewResult) ThinkingInspection {
	// 1. Standard ACP configOption (category == "thought_level")
	if opt, ok := findStandardThinkingOption(meta.ConfigOptions); ok {
		return ThinkingInspection{
			Detected:         true,
			Source:           "standard_config_option",
			ConfigID:         opt.ID,
			CurrentLevel:     opt.EffectiveValue(),
			AvailableChoices: choiceValuesSlice(opt.ExtractChoiceValues()),
		}
	}

	// 2. Non-standard configOption matched heuristically by ID, Name, or effort-shaped choices
	if opt, ok := findHeuristicThinkingOption(meta.ConfigOptions); ok {
		return ThinkingInspection{
			Detected:         true,
			Source:           "heuristic_config_option",
			ConfigID:         opt.ID,
			CurrentLevel:     opt.EffectiveValue(),
			AvailableChoices: choiceValuesSlice(opt.ExtractChoiceValues()),
		}
	}

	// 3. Non-standard top-level fields on SessionNewResult
	for _, v := range []FlexString{meta.ThoughtLevel, meta.ThinkingLevel, meta.ReasoningEffort} {
		if s := strings.TrimSpace(v.String()); s != "" {
			return ThinkingInspection{
				Detected:     true,
				Source:       "top_level_field",
				CurrentLevel: s,
			}
		}
	}
	if len(meta.Thinking) > 0 {
		if lvl, choices := parseLooseThinkingField(meta.Thinking); lvl != "" || len(choices) > 0 {
			return ThinkingInspection{
				Detected:         true,
				Source:           "top_level_field",
				CurrentLevel:     lvl,
				AvailableChoices: choices,
			}
		}
	}

	// 4. _meta extension map on SessionNewResult
	if len(meta.Meta) > 0 {
		for _, k := range []string{"thought_level", "thoughtLevel", "thinking_level", "thinkingLevel", "reasoning_effort", "reasoningEffort", "effort", "thinking"} {
			if raw, exists := meta.Meta[k]; exists && raw != nil {
				b, _ := json.Marshal(raw)
				if lvl, choices := parseLooseThinkingField(b); lvl != "" || len(choices) > 0 {
					return ThinkingInspection{
						Detected:         true,
						Source:           "meta_field",
						CurrentLevel:     lvl,
						AvailableChoices: choices,
					}
				}
			}
		}
	}

	// 5. Model variants encoding thinking tiers (e.g. model-thinking, model-high)
	if meta.Models != nil {
		var variants []string
		for _, m := range meta.Models.AvailableModels {
			lower := strings.ToLower(m.ModelID)
			if strings.Contains(lower, "thinking") || strings.HasSuffix(lower, "-high") || strings.HasSuffix(lower, "/high") || strings.HasSuffix(lower, "-xhigh") {
				variants = append(variants, m.ModelID)
			}
		}
		if len(variants) > 0 {
			return ThinkingInspection{
				Detected:         true,
				Source:           "model_variants",
				CurrentLevel:     meta.Models.CurrentModelID,
				AvailableChoices: variants,
			}
		}
	}

	// 6. Modes encoding thinking behavior
	if meta.Modes != nil {
		var modes []string
		for _, m := range meta.Modes.AvailableModes {
			lower := strings.ToLower(m.ID + " " + m.Name)
			if strings.Contains(lower, "think") || strings.Contains(lower, "reason") {
				modes = append(modes, m.ID)
			}
		}
		if len(modes) > 0 {
			return ThinkingInspection{
				Detected:         true,
				Source:           "modes",
				CurrentLevel:     meta.Modes.CurrentModeID,
				AvailableChoices: modes,
			}
		}
	}

	return ThinkingInspection{Detected: false, Source: "none"}
}

// ThinkingResolution describes the concrete protocol action(s) chosen to satisfy DesiredThinking.
type ThinkingResolution struct {
	Strategy      string         // e.g. "config_option", "heuristic_config_option", "model_variant", "mode", "override_config_option", "fallback_meta", "fallback_prompt_hint", "fallback_ignore"
	ConfigID      string         // non-empty -> call session/set_config_option
	ConfigValue   any            // typed value for session/set_config_option
	ModelOverride string         // non-empty -> switch session model to this variant
	ModeOverride  string         // non-empty -> call session/set_mode
	MetaHint      map[string]any // attached to session/prompt._meta
	PromptHint    string         // optional prompt-level thinking guidance
}

// ResolveThinkingAction executes the 6-layer progressive resolution pipeline for applying DesiredThinking
// across standard, non-standard, and completely opaque ACP agents.
func ResolveThinkingAction(
	meta SessionNewResult,
	desired DesiredThinking,
	baseModel string,
	cfg config.ThinkingConfig,
) (ThinkingResolution, error) {
	if !cfg.Enabled || !desired.Specified {
		return ThinkingResolution{Strategy: "none"}, nil
	}

	effectiveDesired := desired
	if alias, ok := cfg.LevelAliases[desired.Level]; ok && strings.TrimSpace(alias) != "" {
		effectiveDesired.Level = strings.TrimSpace(alias)
	}

	// Layer 1: Standard ACP configOption (category == "thought_level")
	if opt, ok := findStandardThinkingOption(meta.ConfigOptions); ok {
		val := matchOptionValue(opt, effectiveDesired)
		return ThinkingResolution{
			Strategy:    "config_option",
			ConfigID:    opt.ID,
			ConfigValue: val,
		}, nil
	}

	// Layer 2: Heuristic non-standard configOption detection
	if opt, ok := findHeuristicThinkingOption(meta.ConfigOptions); ok {
		val := matchOptionValue(opt, effectiveDesired)
		return ThinkingResolution{
			Strategy:    "heuristic_config_option",
			ConfigID:    opt.ID,
			ConfigValue: val,
		}, nil
	}

	// Layer 3a: Model variant matching (e.g. <model>-high, <model>/high, <model>-thinking)
	if variant := matchThinkingModelVariant(meta.Models, baseModel, effectiveDesired); variant != "" {
		return ThinkingResolution{
			Strategy:      "model_variant",
			ModelOverride: variant,
		}, nil
	}

	// Layer 3b: Session Mode matching (e.g. mode "think" / "deep-thinking")
	if modeID := matchThinkingMode(meta.Modes, effectiveDesired); modeID != "" {
		return ThinkingResolution{
			Strategy:     "mode",
			ModeOverride: modeID,
		}, nil
	}

	// Layer 4: Explicit operator-configured OverrideConfigID (optimistic set_config_option)
	if overrideID := strings.TrimSpace(cfg.OverrideConfigID); overrideID != "" {
		res := ThinkingResolution{
			Strategy:    "override_config_option",
			ConfigID:    overrideID,
			ConfigValue: effectiveDesired.Level,
		}
		if cfg.SendMetaHint {
			res.MetaHint = buildThinkingMetaHint(effectiveDesired)
		}
		return res, nil
	}

	// Layer 5: Configured fallback when ACP server provides NO thinking level mechanism
	switch cfg.FallbackStrategy {
	case config.ThinkingFallbackStrictError:
		return ThinkingResolution{Strategy: "strict_error"}, fmt.Errorf("%w: level=%q", ErrUnsupportedThinkingLevel, desired.Level)

	case config.ThinkingFallbackIgnore:
		return ThinkingResolution{Strategy: "fallback_ignore"}, nil

	case config.ThinkingFallbackPromptHint:
		res := ThinkingResolution{
			Strategy:   "fallback_prompt_hint",
			PromptHint: buildPromptThinkingGuidance(effectiveDesired),
		}
		if cfg.SendMetaHint {
			res.MetaHint = buildThinkingMetaHint(effectiveDesired)
		}
		return res, nil

	default: // ThinkingFallbackAuto
		if cfg.SendMetaHint {
			return ThinkingResolution{
				Strategy: "fallback_meta",
				MetaHint: buildThinkingMetaHint(effectiveDesired),
			}, nil
		}
		return ThinkingResolution{Strategy: "fallback_ignore"}, nil
	}
}

func findStandardThinkingOption(opts []SessionConfigOption) (SessionConfigOption, bool) {
	for _, opt := range opts {
		cat := strings.ToLower(strings.TrimSpace(opt.Category))
		if cat == "thought_level" || cat == "thinking_level" || cat == "reasoning_effort" {
			return opt, true
		}
	}
	return SessionConfigOption{}, false
}

func findHeuristicThinkingOption(opts []SessionConfigOption) (SessionConfigOption, bool) {
	knownIDs := map[string]bool{
		"thought_level":          true,
		"thoughtlevel":           true,
		"thinking_level":         true,
		"thinkinglevel":          true,
		"reasoning_effort":       true,
		"reasoningeffort":        true,
		"model_reasoning_effort": true,
		"effort":                 true,
		"thinking":               true,
		"reasoning":              true,
		"thinking_budget":        true,
		"budget_tokens":          true,
		"think_mode":             true,
	}

	for _, opt := range opts {
		cat := strings.ToLower(strings.TrimSpace(opt.Category))
		if cat == "model" || cat == "mode" {
			continue
		}
		idLower := strings.ToLower(strings.TrimSpace(opt.ID))
		nameLower := strings.ToLower(strings.TrimSpace(opt.Name))
		if knownIDs[idLower] ||
			strings.Contains(idLower, "thought") ||
			strings.Contains(idLower, "think") ||
			strings.Contains(idLower, "reason") ||
			strings.Contains(idLower, "effort") ||
			strings.Contains(nameLower, "thinking") ||
			strings.Contains(nameLower, "reasoning") ||
			strings.Contains(nameLower, "thought") ||
			strings.Contains(nameLower, "思考") {
			return opt, true
		}

		// Inspect choice values: if choices look like ["low", "medium", "high"], recognize it even with opaque ID!
		choices := opt.ExtractChoiceValues()
		if len(choices) >= 2 && choicesLookLikeEffortTiers(choices) {
			return opt, true
		}
	}
	return SessionConfigOption{}, false
}

func choicesLookLikeEffortTiers(choices []ConfigOptionChoice) bool {
	tierMatches := 0
	for _, c := range choices {
		switch strings.ToLower(strings.TrimSpace(c.Value)) {
		case "minimal", "low", "medium", "high", "xhigh", "max":
			tierMatches++
		}
	}
	return tierMatches >= 2 && tierMatches*2 >= len(choices)
}

func matchOptionValue(opt SessionConfigOption, desired DesiredThinking) any {
	optType := strings.ToLower(strings.TrimSpace(opt.Type))
	if optType == "boolean" || optType == "bool" {
		return desired.Enabled
	}

	choices := opt.ExtractChoiceValues()
	if len(choices) == 0 {
		if optType == "number" || optType == "integer" {
			if !desired.Enabled {
				return 0
			}
			return defaultBudgetForTier(desired.Level, desired.BudgetTokens)
		}
		return desired.Level
	}

	// 1. Exact case-insensitive match
	for _, c := range choices {
		if strings.EqualFold(c.Value, desired.Level) {
			return coerceChoiceType(optType, c.Value)
		}
	}

	// 2. Boolean-like string choices ("true"/"false", "on"/"off", "enabled"/"disabled")
	if matched, ok := matchBooleanLikeChoice(choices, desired.Enabled); ok {
		return coerceChoiceType(optType, matched)
	}

	// 3. Numeric token budget choices ("1024", "4096", "16384", ...)
	if matched, ok := matchNumericBudgetChoice(choices, desired); ok {
		return coerceChoiceType(optType, matched)
	}

	// 4. Nearest tier rank clamping (e.g. desired "xhigh" -> highest available "high"; desired "minimal" -> "low")
	targetRank := tierRank(desired.Level, desired.Enabled)
	bestVal := choices[0].Value
	bestDist := math.MaxInt32
	for _, c := range choices {
		r, known := choiceTierRank(c.Value)
		if !known {
			continue
		}
		dist := absInt(r - targetRank)
		if dist < bestDist {
			bestDist = dist
			bestVal = c.Value
		}
	}
	if bestDist < math.MaxInt32 {
		return coerceChoiceType(optType, bestVal)
	}

	return coerceChoiceType(optType, desired.Level)
}

func coerceChoiceType(optType, val string) any {
	if optType == "number" || optType == "integer" {
		if n, ok := ParseNumericOption(val); ok {
			return n
		}
	}
	if optType == "boolean" || optType == "bool" {
		return strings.EqualFold(val, "true") || strings.EqualFold(val, "1") || strings.EqualFold(val, "on")
	}
	return val
}

func matchBooleanLikeChoice(choices []ConfigOptionChoice, enabled bool) (string, bool) {
	if len(choices) > 3 {
		return "", false
	}
	var trueVal, falseVal string
	for _, c := range choices {
		switch strings.ToLower(strings.TrimSpace(c.Value)) {
		case "true", "on", "enabled", "yes", "1", "adaptive", "thinking":
			trueVal = c.Value
		case "false", "off", "disabled", "no", "0", "none":
			falseVal = c.Value
		}
	}
	if trueVal != "" && falseVal != "" {
		if enabled {
			return trueVal, true
		}
		return falseVal, true
	}
	return "", false
}

func matchNumericBudgetChoice(choices []ConfigOptionChoice, desired DesiredThinking) (string, bool) {
	type numChoice struct {
		raw string
		val int
	}
	var nums []numChoice
	for _, c := range choices {
		if n, ok := ParseNumericOption(c.Value); ok {
			nums = append(nums, numChoice{raw: c.Value, val: n})
		}
	}
	if len(nums) == 0 {
		return "", false
	}
	target := 0
	if desired.Enabled {
		target = defaultBudgetForTier(desired.Level, desired.BudgetTokens)
	}
	best := nums[0]
	bestDiff := absInt(nums[0].val - target)
	for _, nc := range nums[1:] {
		if d := absInt(nc.val - target); d < bestDiff {
			best = nc
			bestDiff = d
		}
	}
	return best.raw, true
}

func tierRank(level string, enabled bool) int {
	if !enabled {
		return 0
	}
	if r, ok := choiceTierRank(level); ok {
		return r
	}
	return 3 // default medium
}

func choiceTierRank(val string) (int, bool) {
	switch strings.ToLower(strings.TrimSpace(val)) {
	case "none", "off", "disabled", "false", "0":
		return 0, true
	case "minimal", "min":
		return 1, true
	case "low":
		return 2, true
	case "medium", "med", "default", "auto", "adaptive":
		return 3, true
	case "high", "thinking", "enabled":
		return 4, true
	case "xhigh", "max", "ultra":
		return 5, true
	default:
		return 0, false
	}
}

func matchThinkingModelVariant(models *SessionModelState, baseModel string, desired DesiredThinking) string {
	if models == nil || len(models.AvailableModels) == 0 || !desired.Enabled {
		return ""
	}
	base := strings.TrimSpace(baseModel)
	if base == "" {
		base = models.CurrentModelID
	}
	if base == "" {
		return ""
	}
	candidates := []string{
		base + "-" + desired.Level,
		base + "/" + desired.Level,
		base + ":" + desired.Level,
		base + "-thinking",
	}
	for _, cand := range candidates {
		for _, m := range models.AvailableModels {
			if strings.EqualFold(m.ModelID, cand) {
				return m.ModelID
			}
		}
	}
	return ""
}

func matchThinkingMode(modes *SessionModeState, desired DesiredThinking) string {
	if modes == nil || len(modes.AvailableModes) == 0 || !desired.Enabled {
		return ""
	}
	for _, m := range modes.AvailableModes {
		if strings.EqualFold(m.ID, desired.Level) || strings.EqualFold(m.ID, "think") || strings.EqualFold(m.ID, "thinking") {
			return m.ID
		}
	}
	return ""
}

func buildThinkingMetaHint(desired DesiredThinking) map[string]any {
	m := map[string]any{
		"thoughtLevel":    desired.Level,
		"reasoningEffort": desired.Level,
		"thinkingEnabled": desired.Enabled,
	}
	if desired.BudgetTokens > 0 {
		m["thinkingBudgetTokens"] = desired.BudgetTokens
	}
	return m
}

func buildPromptThinkingGuidance(desired DesiredThinking) string {
	if !desired.Enabled {
		return "<thinking_guidance>Respond concisely and directly without extended deliberation.</thinking_guidance>"
	}
	switch desired.Level {
	case "minimal", "low":
		return "<thinking_guidance>Keep internal reasoning brief and focused before answering.</thinking_guidance>"
	case "high", "xhigh":
		return "<thinking_guidance>Think thoroughly and step-by-step through edge cases before answering.</thinking_guidance>"
	default:
		return "<thinking_guidance>Reason carefully before providing your final answer.</thinking_guidance>"
	}
}

func choiceValuesSlice(choices []ConfigOptionChoice) []string {
	if len(choices) == 0 {
		return nil
	}
	out := make([]string, 0, len(choices))
	for _, c := range choices {
		out = append(out, c.Value)
	}
	return out
}

func parseLooseThinkingField(raw []byte) (string, []string) {
	var fs FlexString
	if err := json.Unmarshal(raw, &fs); err == nil && strings.TrimSpace(fs.String()) != "" && raw[0] != '{' && raw[0] != '[' {
		return strings.TrimSpace(fs.String()), nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err == nil {
		lvl := extractFlexField(obj, "level", "current", "currentValue", "effort", "mode", "type")
		var choices []string
		for _, k := range []string{"availableLevels", "levels", "options", "values"} {
			if arrRaw, ok := obj[k]; ok {
				var sc []FlexString
				if json.Unmarshal(arrRaw, &sc) == nil {
					for _, s := range sc {
						if v := strings.TrimSpace(s.String()); v != "" {
							choices = append(choices, v)
						}
					}
				}
			}
		}
		return lvl, choices
	}
	return "", nil
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
