package acp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"acp2api/internal/config"
)

const ProtocolVersion = 1

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

// rpcEnvelope is used to decode arbitrary incoming JSON-RPC 2.0 messages on stdout.
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

// --- Handshake (initialize) ---

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
}

type InitializeParams struct {
	ProtocolVersion    int                `json:"protocolVersion"`
	ClientInfo         ImplementationInfo `json:"clientInfo"`
	ClientCapabilities ClientCapabilities `json:"clientCapabilities"`
}

type PromptCapabilities struct {
	Image           bool `json:"image"`
	Audio           bool `json:"audio"`
	EmbeddedContext bool `json:"embeddedContext"`
}

type AgentCapabilities struct {
	LoadSession        bool               `json:"loadSession"`
	PromptCapabilities PromptCapabilities `json:"promptCapabilities"`
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
}

// --- Session Creation & Configuration ---

type SessionNewParams struct {
	Cwd        string                   `json:"cwd"`
	MCPServers []config.MCPServerConfig `json:"mcpServers"`
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

type ConfigOptionChoice struct {
	Value       string `json:"value"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

type SessionConfigOption struct {
	ID           string               `json:"id"`
	Name         string               `json:"name"`
	Description  string               `json:"description,omitempty"`
	Category     string               `json:"category,omitempty"`
	Type         string               `json:"type,omitempty"`
	CurrentValue string               `json:"currentValue,omitempty"`
	Options      json.RawMessage      `json:"options,omitempty"`
}

// ExtractChoiceValues flattens flat or grouped SessionConfigOption choices.
func (o SessionConfigOption) ExtractChoiceValues() []ConfigOptionChoice {
	if len(o.Options) == 0 {
		return nil
	}
	var flat []ConfigOptionChoice
	if err := json.Unmarshal(o.Options, &flat); err == nil && len(flat) > 0 && flat[0].Value != "" {
		return flat
	}
	var grouped []struct {
		Group   string               `json:"group"`
		Options []ConfigOptionChoice `json:"options"`
	}
	if err := json.Unmarshal(o.Options, &grouped); err == nil {
		var out []ConfigOptionChoice
		for _, g := range grouped {
			out = append(out, g.Options...)
		}
		return out
	}
	return nil
}

type SessionNewResult struct {
	SessionID     string                `json:"sessionId"`
	Models        *SessionModelState    `json:"models,omitempty"`
	ConfigOptions []SessionConfigOption `json:"configOptions,omitempty"`
}

type SetConfigOptionParams struct {
	SessionID string `json:"sessionId"`
	ConfigID  string `json:"configId"`
	Value     string `json:"value"`
}

type SetModelParams struct {
	SessionID string `json:"sessionId"`
	ModelID   string `json:"modelId"`
}

// --- Prompt & Streaming Updates ---

type ContentBlock struct {
	Type     string `json:"type"`               // "text", "image", "resource_link"
	Text     string `json:"text,omitempty"`     // for type == "text"
	MimeType string `json:"mimeType,omitempty"` // for type == "image"
	Data     string `json:"data,omitempty"`     // base64 data for type == "image"
	URI      string `json:"uri,omitempty"`      // optional URI for image/resource_link
	Name     string `json:"name,omitempty"`     // for resource_link
}

type SessionPromptParams struct {
	SessionID string         `json:"sessionId"`
	Prompt    []ContentBlock `json:"prompt"`
}

type UsageInfo struct {
	InputTokens  int `json:"inputTokens,omitempty"`
	OutputTokens int `json:"outputTokens,omitempty"`
	PromptTokens int `json:"promptTokens,omitempty"`
	TotalTokens  int `json:"totalTokens,omitempty"`
}

type SessionPromptResult struct {
	StopReason string     `json:"stopReason"`
	Usage      *UsageInfo `json:"usage,omitempty"`
}

type SessionCancelParams struct {
	SessionID string `json:"sessionId"`
}

type SessionUpdateParams struct {
	SessionID string          `json:"sessionId"`
	Update    json.RawMessage `json:"update"`
}

// ParsedUpdate is a normalized representation of a session/update notification.
type ParsedUpdate struct {
	Kind       string // "agent_message_chunk", "agent_thought_chunk", "tool_call", "tool_call_update", "usage_update", etc.
	Text       string
	ToolCallID string
	ToolTitle  string
	ToolKind   string
	ToolStatus string
	Usage      *UsageInfo
}

type rawSessionUpdate struct {
	SessionUpdate string          `json:"sessionUpdate"`
	Content       json.RawMessage `json:"content,omitempty"`
	Text          string          `json:"text,omitempty"`
	ToolCallID    string          `json:"toolCallId,omitempty"`
	Title         string          `json:"title,omitempty"`
	Name          string          `json:"name,omitempty"`
	Kind          string          `json:"kind,omitempty"`
	Status        string          `json:"status,omitempty"`
	Usage         *UsageInfo      `json:"usage,omitempty"`
}

// ParseSessionUpdate decodes a session/update.update payload tolerantly across ACP v1 implementations.
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
		Kind:       u.SessionUpdate,
		ToolCallID: u.ToolCallID,
		ToolTitle:  title,
		ToolKind:   u.Kind,
		ToolStatus: u.Status,
		Usage:      u.Usage,
	}
	switch u.SessionUpdate {
	case "agent_message_chunk", "agent_thought_chunk":
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
			if b.Type == "" || b.Type == "text" {
				return b.Text
			}
		}
	case '[':
		var blocks []ContentBlock
		if err := json.Unmarshal(trimmed, &blocks); err == nil {
			var sb strings.Builder
			for _, b := range blocks {
				if b.Type == "" || b.Type == "text" {
					sb.WriteString(b.Text)
				}
			}
			return sb.String()
		}
	}
	return ""
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
	Kind       string `json:"kind,omitempty"` // "read", "edit", "delete", "move", "search", "execute", "think", "fetch", "other"
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
	Line      int    `json:"line,omitempty"`      // ACP v1 1-based start line
	Limit     int    `json:"limit,omitempty"`     // ACP v1 max line count
	LineStart int    `json:"lineStart,omitempty"` // fallback alias
	LineLimit int    `json:"lineLimit,omitempty"` // fallback alias
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
	Env             json.RawMessage `json:"env,omitempty"` // []EnvPair or map[string]string
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
