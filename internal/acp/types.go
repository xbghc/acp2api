package acp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"acp2api/internal/config"
)

const ProtocolVersion = 1

// FlexString tolerantly unmarshals JSON strings, booleans, or numbers into a Go string.
// Many non-standard ACP servers emit boolean or numeric config values (e.g., currentValue: true or 16000).
type FlexString string

func (fs *FlexString) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		*fs = ""
		return nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return err
		}
		*fs = FlexString(s)
		return nil
	}
	// Boolean or number literal
	*fs = FlexString(string(trimmed))
	return nil
}

func (fs FlexString) String() string {
	return string(fs)
}

// RPCError represents a JSON-RPC 2.0 error object.
type RPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *RPCError) Error() string {
	if len(e.Data) > 0 && string(e.Data) != "null" {
		return fmt.Sprintf("acp rpc error %d: %s (%s)", e.Code, e.Message, string(e.Data))
	}
	return fmt.Sprintf("acp rpc error %d: %s", e.Code, e.Message)
}

type rpcEnvelope struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

// --- Handshake & Authentication (initialize, authenticate) ---

type ImplementationInfo struct {
	Name    string `json:"name"`
	Title   string `json:"title,omitempty"`
	Version string `json:"version"`
}

type FSCapabilities struct {
	ReadTextFile  bool `json:"readTextFile"`
	WriteTextFile bool `json:"writeTextFile"`
}

type ClientCapabilities struct {
	FS       FSCapabilities `json:"fs"`
	Terminal bool           `json:"terminal"`
	Meta     map[string]any `json:"_meta,omitempty"`
}

type InitializeParams struct {
	ProtocolVersion    int                `json:"protocolVersion"`
	ClientInfo         ImplementationInfo `json:"clientInfo"`
	ClientCapabilities ClientCapabilities `json:"clientCapabilities"`
	Meta               map[string]any     `json:"_meta,omitempty"`
}

type MCPCapabilities struct {
	HTTP bool `json:"http"`
	SSE  bool `json:"sse"`
}

type PromptCapabilities struct {
	Image           bool `json:"image"`
	Audio           bool `json:"audio"`
	EmbeddedContext bool `json:"embeddedContext"`
}

type SessionCapabilities struct {
	Fork   json.RawMessage `json:"fork,omitempty"`
	Resume json.RawMessage `json:"resume,omitempty"`
	List   json.RawMessage `json:"list,omitempty"`
	Close  json.RawMessage `json:"close,omitempty"`
}

func hasRawCapability(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" || string(trimmed) == "false" {
		return false
	}
	return true
}

func (sc SessionCapabilities) HasFork() bool   { return hasRawCapability(sc.Fork) }
func (sc SessionCapabilities) HasResume() bool { return hasRawCapability(sc.Resume) }
func (sc SessionCapabilities) HasList() bool   { return hasRawCapability(sc.List) }
func (sc SessionCapabilities) HasClose() bool  { return hasRawCapability(sc.Close) }

type AgentCapabilities struct {
	LoadSession         bool                `json:"loadSession"`
	MCPCapabilities     MCPCapabilities     `json:"mcpCapabilities"`
	PromptCapabilities  PromptCapabilities  `json:"promptCapabilities"`
	SessionCapabilities SessionCapabilities `json:"sessionCapabilities"`
	Meta                map[string]any      `json:"_meta,omitempty"`
}

type AuthMethod struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

type InitializeResult struct {
	ProtocolVersion   int                `json:"protocolVersion"`
	AgentInfo         ImplementationInfo `json:"agentInfo"`
	AgentCapabilities AgentCapabilities  `json:"agentCapabilities"`
	AuthMethods       []AuthMethod       `json:"authMethods,omitempty"`
	Meta              map[string]any     `json:"_meta,omitempty"`
}

type AuthenticateParams struct {
	MethodID string         `json:"methodId"`
	Meta     map[string]any `json:"_meta,omitempty"`
}

// --- Session Creation, Lifecycle, Modes & Configuration ---

type SessionNewParams struct {
	Cwd        string                   `json:"cwd"`
	MCPServers []config.MCPServerConfig `json:"mcpServers"`
	Meta       map[string]any           `json:"_meta,omitempty"`
}

type SessionLoadParams struct {
	SessionID  string                   `json:"sessionId"`
	Cwd        string                   `json:"cwd"`
	MCPServers []config.MCPServerConfig `json:"mcpServers"`
	Meta       map[string]any           `json:"_meta,omitempty"`
}

type SessionResumeParams struct {
	SessionID  string                   `json:"sessionId"`
	Cwd        string                   `json:"cwd"`
	MCPServers []config.MCPServerConfig `json:"mcpServers,omitempty"`
	Meta       map[string]any           `json:"_meta,omitempty"`
}

type SessionForkParams struct {
	SessionID  string                   `json:"sessionId"`
	Cwd        string                   `json:"cwd"`
	MCPServers []config.MCPServerConfig `json:"mcpServers,omitempty"`
	Meta       map[string]any           `json:"_meta,omitempty"`
}

type SessionListParams struct {
	Cwd    string         `json:"cwd,omitempty"`
	Cursor string         `json:"cursor,omitempty"`
	Meta   map[string]any `json:"_meta,omitempty"`
}

type SessionInfo struct {
	SessionID string         `json:"sessionId"`
	Cwd       string         `json:"cwd,omitempty"`
	Title     string         `json:"title,omitempty"`
	UpdatedAt string         `json:"updatedAt,omitempty"`
	Meta      map[string]any `json:"_meta,omitempty"`
}

type SessionListResult struct {
	Sessions   []SessionInfo `json:"sessions"`
	NextCursor string        `json:"nextCursor,omitempty"`
}

type SessionCloseParams struct {
	SessionID string         `json:"sessionId"`
	Meta      map[string]any `json:"_meta,omitempty"`
}

type ModelInfo struct {
	ModelID     string `json:"modelId"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

type SessionModelState struct {
	CurrentModelID  string      `json:"currentModelId"`
	AvailableModels []ModelInfo `json:"availableModels"`
}

type SessionMode struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

type SessionModeState struct {
	CurrentModeID  string        `json:"currentModeId"`
	AvailableModes []SessionMode `json:"availableModes"`
}

type ConfigOptionChoice struct {
	Value       string `json:"value"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

type SessionConfigOption struct {
	ID           string          `json:"id"`
	Name         string          `json:"name"`
	Description  string          `json:"description,omitempty"`
	Category     string          `json:"category,omitempty"`
	Type         string          `json:"type,omitempty"`
	CurrentValue FlexString      `json:"currentValue,omitempty"`
	DefaultValue FlexString      `json:"defaultValue,omitempty"` // non-standard fallback alias
	Options      json.RawMessage `json:"options,omitempty"`
	Values       json.RawMessage `json:"values,omitempty"` // non-standard fallback alias
}

// EffectiveValue returns CurrentValue or DefaultValue as a plain string.
func (o SessionConfigOption) EffectiveValue() string {
	if s := strings.TrimSpace(o.CurrentValue.String()); s != "" {
		return s
	}
	return strings.TrimSpace(o.DefaultValue.String())
}

// ExtractChoiceValues tolerantly flattens standard ACP select options, grouped options,
// non-standard {id,label} objects, scalar arrays (["low","medium","high"] or [1024,4096]),
// or boolean toggles.
func (o SessionConfigOption) ExtractChoiceValues() []ConfigOptionChoice {
	raw := o.Options
	if len(bytes.TrimSpace(raw)) == 0 || string(bytes.TrimSpace(raw)) == "null" {
		raw = o.Values
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		if strings.EqualFold(o.Type, "boolean") || strings.EqualFold(o.Type, "bool") {
			return []ConfigOptionChoice{
				{Value: "true", Name: "Enabled"},
				{Value: "false", Name: "Disabled"},
			}
		}
		return nil
	}

	if trimmed[0] != '[' {
		return nil
	}

	// 1. Standard [{value, name, description}] or non-standard [{id, label}]
	var rawObjs []map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &rawObjs); err == nil && len(rawObjs) > 0 {
		// Check if grouped: [{"group": "...", "options": [...]}]
		if _, hasGroup := rawObjs[0]["options"]; hasGroup {
			var grouped []struct {
				Group   string               `json:"group"`
				Options []ConfigOptionChoice `json:"options"`
			}
			if err := json.Unmarshal(trimmed, &grouped); err == nil {
				var out []ConfigOptionChoice
				for _, g := range grouped {
					out = append(out, g.Options...)
				}
				if len(out) > 0 {
					return out
				}
			}
		}

		var out []ConfigOptionChoice
		for _, item := range rawObjs {
			val := extractFlexField(item, "value", "id", "key", "level")
			if val == "" {
				continue
			}
			name := extractFlexField(item, "name", "label", "title")
			if name == "" {
				name = val
			}
			desc := extractFlexField(item, "description")
			out = append(out, ConfigOptionChoice{
				Value:       val,
				Name:        name,
				Description: desc,
			})
		}
		if len(out) > 0 {
			return out
		}
	}

	// 2. Non-standard array of scalars: ["low", "medium", "high"] or [1024, 4096, 16000]
	var scalars []FlexString
	if err := json.Unmarshal(trimmed, &scalars); err == nil && len(scalars) > 0 {
		out := make([]ConfigOptionChoice, 0, len(scalars))
		for _, sc := range scalars {
			v := strings.TrimSpace(sc.String())
			if v != "" {
				out = append(out, ConfigOptionChoice{Value: v, Name: v})
			}
		}
		if len(out) > 0 {
			return out
		}
	}

	return nil
}

func extractFlexField(m map[string]json.RawMessage, keys ...string) string {
	for _, k := range keys {
		if raw, ok := m[k]; ok {
			var fs FlexString
			if json.Unmarshal(raw, &fs) == nil && strings.TrimSpace(fs.String()) != "" {
				return strings.TrimSpace(fs.String())
			}
		}
	}
	return ""
}

// SessionNewResult captures both standard ACP session state and non-standard thinking metadata fields.
type SessionNewResult struct {
	SessionID       string                `json:"sessionId"`
	Models          *SessionModelState    `json:"models,omitempty"`
	Modes           *SessionModeState     `json:"modes,omitempty"`
	ConfigOptions   []SessionConfigOption `json:"configOptions,omitempty"`
	ThoughtLevel    FlexString            `json:"thoughtLevel,omitempty"`    // non-standard top-level field
	ThinkingLevel   FlexString            `json:"thinkingLevel,omitempty"`   // non-standard top-level field
	ReasoningEffort FlexString            `json:"reasoningEffort,omitempty"` // non-standard top-level field
	Thinking        json.RawMessage       `json:"thinking,omitempty"`        // non-standard top-level object/string
	Meta            map[string]any        `json:"_meta,omitempty"`
}

type SetConfigOptionParams struct {
	SessionID string `json:"sessionId"`
	ConfigID  string `json:"configId"`
	Value     any    `json:"value"`
}

type SetConfigOptionResult struct {
	ConfigOptions []SessionConfigOption `json:"configOptions,omitempty"`
}

type SetModelParams struct {
	SessionID string `json:"sessionId"`
	ModelID   string `json:"modelId"`
}

type SetSessionModeParams struct {
	SessionID string `json:"sessionId"`
	ModeID    string `json:"modeId"`
}

// --- Content Blocks, Prompts & Streaming Updates ---

type EmbeddedResource struct {
	URI      string         `json:"uri"`
	MimeType string         `json:"mimeType,omitempty"`
	Text     string         `json:"text,omitempty"`
	Blob     string         `json:"blob,omitempty"` // base64-encoded binary resource
	Meta     map[string]any `json:"_meta,omitempty"`
}

type Annotations struct {
	Audience     []string `json:"audience,omitempty"`
	Priority     *float64 `json:"priority,omitempty"`
	LastModified string   `json:"lastModified,omitempty"`
}

type ContentBlock struct {
	Type        string            `json:"type"`                  // "text", "image", "audio", "resource", "resource_link"
	Text        string            `json:"text,omitempty"`        // for "text"
	MimeType    string            `json:"mimeType,omitempty"`    // for "image", "audio", "resource_link"
	Data        string            `json:"data,omitempty"`        // base64 for "image", "audio"
	URI         string            `json:"uri,omitempty"`         // for "image", "resource_link"
	Name        string            `json:"name,omitempty"`        // for "resource_link"
	Title       string            `json:"title,omitempty"`       // for "resource_link"
	Description string            `json:"description,omitempty"` // for "resource_link"
	Size        *int64            `json:"size,omitempty"`        // for "resource_link"
	Resource    *EmbeddedResource `json:"resource,omitempty"`    // for "resource"
	Annotations *Annotations      `json:"annotations,omitempty"`
	Meta        map[string]any    `json:"_meta,omitempty"`
}

type SessionPromptParams struct {
	SessionID string         `json:"sessionId"`
	Prompt    []ContentBlock `json:"prompt"`
	Meta      map[string]any `json:"_meta,omitempty"`
}

type UsageInfo struct {
	InputTokens       int `json:"inputTokens,omitempty"`
	OutputTokens      int `json:"outputTokens,omitempty"`
	PromptTokens      int `json:"promptTokens,omitempty"`
	ThoughtTokens     int `json:"thoughtTokens,omitempty"`
	CachedReadTokens  int `json:"cachedReadTokens,omitempty"`
	CachedWriteTokens int `json:"cachedWriteTokens,omitempty"`
	TotalTokens       int `json:"totalTokens,omitempty"`
}

type SessionPromptResult struct {
	StopReason string         `json:"stopReason"`
	Usage      *UsageInfo     `json:"usage,omitempty"`
	Meta       map[string]any `json:"_meta,omitempty"`
}

type SessionCancelParams struct {
	SessionID string `json:"sessionId"`
}

type CancelRequestParams struct {
	ID json.RawMessage `json:"id"`
}

type SessionUpdateParams struct {
	SessionID string          `json:"sessionId"`
	Update    json.RawMessage `json:"update"`
}

type ToolCallLocation struct {
	Path string `json:"path"`
	Line *int   `json:"line,omitempty"`
}

type PlanEntry struct {
	Content  string `json:"content"`
	Priority string `json:"priority,omitempty"`
	Status   string `json:"status,omitempty"`
}

type AvailableCommand struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// ParsedUpdate is a normalized representation of all standard ACP session/update notifications.
type ParsedUpdate struct {
	Kind              string // "agent_message_chunk", "agent_thought_chunk", "user_message_chunk", "tool_call", "tool_call_update", "plan", "config_option_update", "current_mode_update", "available_commands_update", "session_info_update", "usage_update"
	Text              string
	ToolCallID        string
	ToolTitle         string
	ToolKind          string
	ToolStatus        string
	Locations         []ToolCallLocation
	PlanEntries       []PlanEntry
	ConfigOptions     []SessionConfigOption
	CurrentModeID     string
	AvailableCommands []AvailableCommand
	SessionTitle      string
	UpdatedAt         string
	Usage             *UsageInfo
}

type rawSessionUpdate struct {
	SessionUpdate     string                `json:"sessionUpdate"`
	Content           json.RawMessage       `json:"content,omitempty"`
	Text              string                `json:"text,omitempty"`
	ToolCallID        string                `json:"toolCallId,omitempty"`
	Title             string                `json:"title,omitempty"`
	Name              string                `json:"name,omitempty"`
	Kind              string                `json:"kind,omitempty"`
	Status            string                `json:"status,omitempty"`
	Locations         []ToolCallLocation    `json:"locations,omitempty"`
	Entries           []PlanEntry           `json:"entries,omitempty"`
	ConfigOptions     []SessionConfigOption `json:"configOptions,omitempty"`
	CurrentModeID     string                `json:"currentModeId,omitempty"`
	AvailableCommands []AvailableCommand    `json:"availableCommands,omitempty"`
	UpdatedAt         string                `json:"updatedAt,omitempty"`
	Usage             *UsageInfo            `json:"usage,omitempty"`
}

// ParseSessionUpdate decodes any ACP session/update.update payload tolerantly.
func ParseSessionUpdate(raw json.RawMessage) (ParsedUpdate, error) {
	var u rawSessionUpdate
	if err := json.Unmarshal(raw, &u); err != nil {
		return ParsedUpdate{}, err
	}
	title := u.Title
	if title == "" {
		title = u.Name
	}
	parsed := ParsedUpdate{
		Kind:              u.SessionUpdate,
		ToolCallID:        u.ToolCallID,
		ToolTitle:         title,
		ToolKind:          u.Kind,
		ToolStatus:        u.Status,
		Locations:         u.Locations,
		PlanEntries:       u.Entries,
		ConfigOptions:     u.ConfigOptions,
		CurrentModeID:     u.CurrentModeID,
		AvailableCommands: u.AvailableCommands,
		SessionTitle:      u.Title,
		UpdatedAt:         u.UpdatedAt,
		Usage:             u.Usage,
	}
	switch u.SessionUpdate {
	case "agent_message_chunk", "agent_thought_chunk", "user_message_chunk":
		if len(u.Content) > 0 {
			parsed.Text = extractTextFromContent(u.Content)
		} else {
			parsed.Text = u.Text
		}
	}
	return parsed, nil
}

func extractTextFromContent(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return ""
	}
	switch trimmed[0] {
	case '"':
		var s string
		if err := json.Unmarshal(trimmed, &s); err == nil {
			return s
		}
	case '{':
		var b ContentBlock
		if err := json.Unmarshal(trimmed, &b); err == nil {
			switch b.Type {
			case "", "text":
				return b.Text
			case "resource":
				if b.Resource != nil && b.Resource.Text != "" {
					return b.Resource.Text
				}
			}
		}
	case '[':
		var blocks []ContentBlock
		if err := json.Unmarshal(trimmed, &blocks); err == nil {
			var sb strings.Builder
			for _, b := range blocks {
				switch b.Type {
				case "", "text":
					sb.WriteString(b.Text)
				case "resource":
					if b.Resource != nil && b.Resource.Text != "" {
						sb.WriteString(b.Resource.Text)
					}
				}
			}
			return sb.String()
		}
	}
	return ""
}

// ParseNumericOption attempts to parse a string option value as an integer budget.
func ParseNumericOption(s string) (int, bool) {
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0, false
	}
	return v, true
}

// --- Reverse Requests (Agent -> Client) ---

type PermissionOption struct {
	OptionID string `json:"optionId"`
	Name     string `json:"name"`
	Kind     string `json:"kind"` // "allow_once", "allow_always", "reject_once", "reject_always"
}

type PermissionToolCall struct {
	ToolCallID string `json:"toolCallId,omitempty"`
	Title      string `json:"title,omitempty"`
	Kind       string `json:"kind,omitempty"`
	Status     string `json:"status,omitempty"`
}

type RequestPermissionParams struct {
	SessionID string              `json:"sessionId"`
	Title     string              `json:"title,omitempty"`
	ToolCall  *PermissionToolCall `json:"toolCall,omitempty"`
	Options   []PermissionOption  `json:"options"`
}

type PermissionOutcome struct {
	Outcome  string `json:"outcome"`            // "selected" or "cancelled"
	OptionID string `json:"optionId,omitempty"` // required when outcome == "selected"
}

type RequestPermissionResult struct {
	Outcome PermissionOutcome `json:"outcome"`
}

type FSReadTextFileParams struct {
	SessionID string `json:"sessionId"`
	Path      string `json:"path"`
	Line      int    `json:"line,omitempty"`
	Limit     int    `json:"limit,omitempty"`
	LineStart int    `json:"lineStart,omitempty"`
	LineLimit int    `json:"lineLimit,omitempty"`
}

type FSReadTextFileResult struct {
	Content string `json:"content"`
}

type FSWriteTextFileParams struct {
	SessionID string `json:"sessionId"`
	Path      string `json:"path"`
	Content   string `json:"content"`
}

type TerminalCreateParams struct {
	SessionID       string          `json:"sessionId"`
	Command         string          `json:"command"`
	Args            []string        `json:"args,omitempty"`
	Env             json.RawMessage `json:"env,omitempty"`
	Cwd             string          `json:"cwd,omitempty"`
	OutputByteLimit int             `json:"outputByteLimit,omitempty"`
}

type TerminalCreateResult struct {
	TerminalID string `json:"terminalId"`
}

type TerminalIDParams struct {
	SessionID  string `json:"sessionId,omitempty"`
	TerminalID string `json:"terminalId"`
}

type TerminalExitStatus struct {
	ExitCode *int   `json:"exitCode,omitempty"`
	Signal   string `json:"signal,omitempty"`
}

type TerminalOutputResult struct {
	Output     string              `json:"output"`
	Truncated  bool                `json:"truncated"`
	ExitStatus *TerminalExitStatus `json:"exitStatus,omitempty"`
}
