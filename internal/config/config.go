package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// PermissionPolicy controls how ACP session/request_permission callbacks are answered.
type PermissionPolicy string

const (
	PermissionAllowAll      PermissionPolicy = "allow_all"
	PermissionAllowReadOnly PermissionPolicy = "allow_read_only"
	PermissionDenyAll       PermissionPolicy = "deny_all"
)

// ThinkingFallbackStrategy defines what acp2api does when an ACP server neither advertises
// standard thought_level configOptions nor matches heuristic config/model/mode selectors.
type ThinkingFallbackStrategy string

const (
	ThinkingFallbackAuto        ThinkingFallbackStrategy = "auto"         // send _meta hint (if enabled) and degrade gracefully without failing
	ThinkingFallbackPromptHint  ThinkingFallbackStrategy = "prompt_hint"  // also inject lightweight <thinking_guidance> in prompt
	ThinkingFallbackIgnore      ThinkingFallbackStrategy = "ignore"       // silently ignore unresolvable thinking level
	ThinkingFallbackStrictError ThinkingFallbackStrategy = "strict_error" // return HTTP 400 error if requested thinking level cannot be applied
)

// Config holds the full application configuration.
type Config struct {
	Server ServerConfig `json:"server"`
	ACP    ACPConfig    `json:"acp"`
	Pool   PoolConfig   `json:"pool"`
	Log    LogConfig    `json:"log"`
}

// ServerConfig configures the HTTP API server.
type ServerConfig struct {
	ListenAddr            string   `json:"listen_addr"`
	APIKeys               []string `json:"api_keys"`
	ReadHeaderTimeoutSecs int      `json:"read_header_timeout_seconds"`
	RequestTimeoutSecs    int      `json:"request_timeout_seconds"`
}

// EnvPair represents a name-value environment variable for ACP MCP server definitions.
type EnvPair struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// HeaderPair represents an HTTP header for HTTP/SSE MCP server definitions in ACP.
type HeaderPair struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// MCPServerConfig defines an MCP server passed to ACP session/new or session/load.
// Supports ACP stdio (default), http, and sse MCP transport schemas.
type MCPServerConfig struct {
	Type    string       `json:"type,omitempty"`    // "" / "stdio", "http", "sse"
	Name    string       `json:"name"`              // required for all transports
	Command string       `json:"command,omitempty"` // for stdio
	Args    []string     `json:"args,omitempty"`    // for stdio
	Env     []EnvPair    `json:"env,omitempty"`     // for stdio
	URL     string       `json:"url,omitempty"`     // for http / sse
	Headers []HeaderPair `json:"headers,omitempty"` // for http / sse
}

// MarshalJSON ensures ACP-strict non-null slices for the selected MCP transport type.
func (m MCPServerConfig) MarshalJSON() ([]byte, error) {
	t := strings.ToLower(strings.TrimSpace(m.Type))
	switch t {
	case "http", "sse":
		headers := m.Headers
		if headers == nil {
			headers = []HeaderPair{}
		}
		return json.Marshal(struct {
			Type    string       `json:"type"`
			Name    string       `json:"name"`
			URL     string       `json:"url"`
			Headers []HeaderPair `json:"headers"`
		}{
			Type:    t,
			Name:    m.Name,
			URL:     m.URL,
			Headers: headers,
		})
	default:
		args := m.Args
		if args == nil {
			args = []string{}
		}
		env := m.Env
		if env == nil {
			env = []EnvPair{}
		}
		if t == "stdio" {
			return json.Marshal(struct {
				Type    string    `json:"type"`
				Name    string    `json:"name"`
				Command string    `json:"command"`
				Args    []string  `json:"args"`
				Env     []EnvPair `json:"env"`
			}{
				Type:    "stdio",
				Name:    m.Name,
				Command: m.Command,
				Args:    args,
				Env:     env,
			})
		}
		return json.Marshal(struct {
			Name    string    `json:"name"`
			Command string    `json:"command"`
			Args    []string  `json:"args"`
			Env     []EnvPair `json:"env"`
		}{
			Name:    m.Name,
			Command: m.Command,
			Args:    args,
			Env:     env,
		})
	}
}

// ClientCapabilitiesConfig controls which local host capabilities acp2api exposes to the ACP agent.
type ClientCapabilitiesConfig struct {
	EnableFSRead   bool `json:"enable_fs_read"`
	EnableFSWrite  bool `json:"enable_fs_write"`
	EnableTerminal bool `json:"enable_terminal"`
	RestrictToCwd  bool `json:"restrict_to_cwd"`
}

// ThinkingConfig controls how OpenAI reasoning_effort / Anthropic thinking parameters are mapped
// onto standard and non-standard ACP agents.
type ThinkingConfig struct {
	Enabled                bool                     `json:"enabled"`
	OverrideConfigID       string                   `json:"override_config_id,omitempty"`
	LevelAliases           map[string]string        `json:"level_aliases,omitempty"`
	FallbackStrategy       ThinkingFallbackStrategy `json:"fallback_strategy"`
	SendMetaHint           bool                     `json:"send_meta_hint"`
	ExtractInlineThinkTags bool                     `json:"extract_inline_think_tags"`
	ParseModelSuffix       bool                     `json:"parse_model_suffix"`
}

// ACPConfig configures how the ACP stdio subprocess is launched and managed.
type ACPConfig struct {
	Command               string                   `json:"command"`
	Args                  []string                 `json:"args"`
	Env                   map[string]string        `json:"env"`
	Cwd                   string                   `json:"cwd"`
	AuthMethodID          string                   `json:"auth_method_id,omitempty"`
	DefaultMode           string                   `json:"default_mode,omitempty"`
	InitializeTimeoutSecs int                      `json:"initialize_timeout_seconds"`
	SessionNewTimeoutSecs int                      `json:"session_new_timeout_seconds"`
	PermissionPolicy      PermissionPolicy         `json:"permission_policy"`
	Capabilities          ClientCapabilitiesConfig `json:"capabilities"`
	MCPServers            []MCPServerConfig        `json:"mcp_servers"`
	DefaultModel          string                   `json:"default_model"`
	ModelAliases          map[string]string        `json:"model_aliases"`
	PassThroughModels     bool                     `json:"pass_through_models"`
	ExposeThoughts        bool                     `json:"expose_thoughts"`
	Thinking              ThinkingConfig           `json:"thinking"`
}

// PoolConfig configures the ACP process pool, pre-warmed idle session pool, and conversation cache.
type PoolConfig struct {
	ProcessCount            int  `json:"process_count"`
	IdlePoolSize            int  `json:"idle_pool_size"`
	MaxActiveSessions       int  `json:"max_active_sessions"`
	ReuseConversations      bool `json:"reuse_conversations"`
	EnableNativeSessionFork bool `json:"enable_native_session_fork"`
	MaxCachedSessions       int  `json:"max_cached_sessions"`
	IdleSessionTTLSeconds   int  `json:"idle_session_ttl_seconds"`
	CachedSessionTTLSeconds int  `json:"cached_session_ttl_seconds"`
	MaxSessionsPerProcess   int  `json:"max_sessions_per_process"`
}

// LogConfig configures structured logging.
type LogConfig struct {
	Level  string `json:"level"`  // debug, info, warn, error
	Format string `json:"format"` // text, json
}

// Default returns a Config populated with production-ready defaults.
func Default() Config {
	cwd, err := os.Getwd()
	if err != nil || cwd == "" {
		cwd = "."
	}
	return Config{
		Server: ServerConfig{
			ListenAddr:            ":8080",
			APIKeys:               nil,
			ReadHeaderTimeoutSecs: 30,
			RequestTimeoutSecs:    600,
		},
		ACP: ACPConfig{
			Command:               "",
			Args:                  nil,
			Env:                   map[string]string{},
			Cwd:                   cwd,
			InitializeTimeoutSecs: 30,
			SessionNewTimeoutSecs: 30,
			PermissionPolicy:      PermissionAllowAll,
			Capabilities: ClientCapabilitiesConfig{
				EnableFSRead:   false,
				EnableFSWrite:  false,
				EnableTerminal: false,
				RestrictToCwd:  true,
			},
			MCPServers:        []MCPServerConfig{},
			DefaultModel:      "acp-default",
			ModelAliases:      map[string]string{},
			PassThroughModels: true,
			ExposeThoughts:    true,
			Thinking: ThinkingConfig{
				Enabled:                true,
				LevelAliases:           map[string]string{},
				FallbackStrategy:       ThinkingFallbackAuto,
				SendMetaHint:           true,
				ExtractInlineThinkTags: true,
				ParseModelSuffix:       true,
			},
		},
		Pool: PoolConfig{
			ProcessCount:            1,
			IdlePoolSize:            3,
			MaxActiveSessions:       32,
			ReuseConversations:      true,
			EnableNativeSessionFork: true,
			MaxCachedSessions:       64,
			IdleSessionTTLSeconds:   1800,
			CachedSessionTTLSeconds: 1800,
			MaxSessionsPerProcess:   0,
		},
		Log: LogConfig{
			Level:  "info",
			Format: "text",
		},
	}
}

// LoadFile reads a JSON or JSONC configuration file on top of Default().
func LoadFile(path string) (Config, error) {
	cfg := Default()
	raw, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("read config file %q: %w", path, err)
	}
	clean := stripJSONC(raw)
	dec := json.NewDecoder(bytes.NewReader(clean))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return cfg, fmt.Errorf("parse config file %q: %w", path, err)
	}
	return cfg, nil
}

// ApplyEnv overrides configuration fields from environment variables when set.
func (c *Config) ApplyEnv() {
	if v := strings.TrimSpace(os.Getenv("ACP2API_LISTEN")); v != "" {
		c.Server.ListenAddr = v
	}
	if v := strings.TrimSpace(os.Getenv("ACP2API_API_KEY")); v != "" {
		parts := strings.Split(v, ",")
		keys := make([]string, 0, len(parts))
		for _, p := range parts {
			if trimmed := strings.TrimSpace(p); trimmed != "" {
				keys = append(keys, trimmed)
			}
		}
		if len(keys) > 0 {
			c.Server.APIKeys = keys
		}
	}
	if v := strings.TrimSpace(os.Getenv("ACP2API_COMMAND")); v != "" {
		c.ACP.Command = v
	}
	if v := strings.TrimSpace(os.Getenv("ACP2API_ARGS")); v != "" {
		c.ACP.Args = strings.Fields(v)
	}
	if v := strings.TrimSpace(os.Getenv("ACP2API_CWD")); v != "" {
		c.ACP.Cwd = v
	}
	if v := strings.TrimSpace(os.Getenv("ACP2API_PERMISSION_POLICY")); v != "" {
		c.ACP.PermissionPolicy = PermissionPolicy(v)
	}
	if v := strings.TrimSpace(os.Getenv("ACP2API_THINKING_FALLBACK")); v != "" {
		c.ACP.Thinking.FallbackStrategy = ThinkingFallbackStrategy(v)
	}
	if v := strings.TrimSpace(os.Getenv("ACP2API_LOG_LEVEL")); v != "" {
		c.Log.Level = v
	}
}

// NormalizeAndValidate normalizes paths/defaults and checks required settings.
func (c *Config) NormalizeAndValidate() error {
	if strings.TrimSpace(c.Server.ListenAddr) == "" {
		c.Server.ListenAddr = ":8080"
	}
	if strings.TrimSpace(c.ACP.Command) == "" {
		return errors.New("acp.command is required (set via -acp-cmd, ACP2API_COMMAND, config file, or after '--')")
	}
	if strings.TrimSpace(c.ACP.Cwd) == "" {
		cwd, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("resolve working directory: %w", err)
		}
		c.ACP.Cwd = cwd
	}
	absCwd, err := filepath.Abs(c.ACP.Cwd)
	if err != nil {
		return fmt.Errorf("invalid acp.cwd %q: %w", c.ACP.Cwd, err)
	}
	c.ACP.Cwd = absCwd

	switch c.ACP.PermissionPolicy {
	case "", PermissionAllowAll:
		c.ACP.PermissionPolicy = PermissionAllowAll
	case PermissionAllowReadOnly, PermissionDenyAll:
	default:
		return fmt.Errorf("invalid acp.permission_policy %q (expected allow_all, allow_read_only, or deny_all)", c.ACP.PermissionPolicy)
	}

	switch c.ACP.Thinking.FallbackStrategy {
	case "", ThinkingFallbackAuto:
		c.ACP.Thinking.FallbackStrategy = ThinkingFallbackAuto
	case ThinkingFallbackPromptHint, ThinkingFallbackIgnore, ThinkingFallbackStrictError:
	default:
		return fmt.Errorf("invalid acp.thinking.fallback_strategy %q (expected auto, prompt_hint, ignore, or strict_error)", c.ACP.Thinking.FallbackStrategy)
	}
	if c.ACP.Thinking.LevelAliases == nil {
		c.ACP.Thinking.LevelAliases = map[string]string{}
	}

	if c.ACP.InitializeTimeoutSecs <= 0 {
		c.ACP.InitializeTimeoutSecs = 30
	}
	if c.ACP.SessionNewTimeoutSecs <= 0 {
		c.ACP.SessionNewTimeoutSecs = 30
	}
	if c.Server.ReadHeaderTimeoutSecs <= 0 {
		c.Server.ReadHeaderTimeoutSecs = 30
	}
	if c.Server.RequestTimeoutSecs <= 0 {
		c.Server.RequestTimeoutSecs = 600
	}
	if c.Pool.ProcessCount <= 0 {
		c.Pool.ProcessCount = 1
	}
	if c.Pool.IdlePoolSize < 0 {
		c.Pool.IdlePoolSize = 0
	}
	if c.Pool.MaxCachedSessions < 0 {
		c.Pool.MaxCachedSessions = 0
	}
	if c.Pool.IdleSessionTTLSeconds <= 0 {
		c.Pool.IdleSessionTTLSeconds = 1800
	}
	if c.Pool.CachedSessionTTLSeconds <= 0 {
		c.Pool.CachedSessionTTLSeconds = 1800
	}
	if c.ACP.MCPServers == nil {
		c.ACP.MCPServers = []MCPServerConfig{}
	}
	for i := range c.ACP.MCPServers {
		t := strings.ToLower(strings.TrimSpace(c.ACP.MCPServers[i].Type))
		if t == "http" || t == "sse" {
			if c.ACP.MCPServers[i].Headers == nil {
				c.ACP.MCPServers[i].Headers = []HeaderPair{}
			}
		} else {
			if c.ACP.MCPServers[i].Args == nil {
				c.ACP.MCPServers[i].Args = []string{}
			}
			if c.ACP.MCPServers[i].Env == nil {
				c.ACP.MCPServers[i].Env = []EnvPair{}
			}
		}
	}
	if strings.TrimSpace(c.ACP.DefaultModel) == "" {
		c.ACP.DefaultModel = "acp-default"
	}
	return nil
}

func (c *Config) InitializeTimeout() time.Duration {
	return time.Duration(c.ACP.InitializeTimeoutSecs) * time.Second
}

func (c *Config) SessionNewTimeout() time.Duration {
	return time.Duration(c.ACP.SessionNewTimeoutSecs) * time.Second
}

func (c *Config) RequestTimeout() time.Duration {
	return time.Duration(c.Server.RequestTimeoutSecs) * time.Second
}

func (c *Config) IdleSessionTTL() time.Duration {
	return time.Duration(c.Pool.IdleSessionTTLSeconds) * time.Second
}

func (c *Config) CachedSessionTTL() time.Duration {
	return time.Duration(c.Pool.CachedSessionTTLSeconds) * time.Second
}

// ResolveModel maps a client-requested model ID to a target ACP model ID (if any).
// Returns (resolvedACPModel, displayModel).
func (c *Config) ResolveModel(requested string) (string, string) {
	req := strings.TrimSpace(requested)
	if req == "" {
		req = c.ACP.DefaultModel
	}
	if target, ok := c.ACP.ModelAliases[req]; ok {
		return target, req
	}
	if req == c.ACP.DefaultModel || !c.ACP.PassThroughModels {
		return "", req
	}
	return req, req
}

// SplitModelThinkingSuffix extracts optional `:effort` or `(effort)` suffixes (e.g. `claude-sonnet-4-5:high`, `gpt-5(xhigh)`)
// unless the full requested string matches an explicit model alias.
func (c *Config) SplitModelThinkingSuffix(requested string) (cleanModel string, effortSuffix string) {
	req := strings.TrimSpace(requested)
	if req == "" || !c.ACP.Thinking.ParseModelSuffix {
		return req, ""
	}
	if _, exactAlias := c.ACP.ModelAliases[req]; exactAlias {
		return req, ""
	}
	// Check `model(effort)` pattern
	if strings.HasSuffix(req, ")") {
		if openIdx := strings.LastIndexByte(req, '('); openIdx > 0 {
			candidate := strings.ToLower(strings.TrimSpace(req[openIdx+1 : len(req)-1]))
			if isKnownEffortTier(candidate) {
				return strings.TrimSpace(req[:openIdx]), candidate
			}
		}
	}
	// Check `model:effort` pattern
	if colonIdx := strings.LastIndexByte(req, ':'); colonIdx > 0 {
		candidate := strings.ToLower(strings.TrimSpace(req[colonIdx+1:]))
		if isKnownEffortTier(candidate) {
			return strings.TrimSpace(req[:colonIdx]), candidate
		}
	}
	return req, ""
}

func isKnownEffortTier(s string) bool {
	switch s {
	case "none", "off", "disabled", "minimal", "low", "medium", "high", "xhigh", "max", "thinking":
		return true
	default:
		return false
	}
}

// stripJSONC removes // line comments, /* block comments */, and trailing commas outside string literals.
func stripJSONC(src []byte) []byte {
	out := make([]byte, 0, len(src))
	inString := false
	escaped := false

	for i := 0; i < len(src); i++ {
		ch := src[i]
		if inString {
			out = append(out, ch)
			if escaped {
				escaped = false
			} else if ch == '\\' {
				escaped = true
			} else if ch == '"' {
				inString = false
			}
			continue
		}

		if ch == '"' {
			inString = true
			out = append(out, ch)
			continue
		}

		if ch == '/' && i+1 < len(src) && src[i+1] == '/' {
			i += 2
			for i < len(src) && src[i] != '\n' {
				i++
			}
			if i < len(src) {
				out = append(out, '\n')
			}
			continue
		}

		if ch == '/' && i+1 < len(src) && src[i+1] == '*' {
			i += 2
			for i+1 < len(src) && !(src[i] == '*' && src[i+1] == '/') {
				if src[i] == '\n' {
					out = append(out, '\n')
				}
				i++
			}
			i++
			continue
		}

		out = append(out, ch)
	}

	clean := make([]byte, 0, len(out))
	inString = false
	escaped = false
	for i := 0; i < len(out); i++ {
		ch := out[i]
		if inString {
			clean = append(clean, ch)
			if escaped {
				escaped = false
			} else if ch == '\\' {
				escaped = true
			} else if ch == '"' {
				inString = false
			}
			continue
		}
		if ch == '"' {
			inString = true
			clean = append(clean, ch)
			continue
		}
		if ch == ',' {
			j := i + 1
			for j < len(out) && (out[j] == ' ' || out[j] == '\t' || out[j] == '\r' || out[j] == '\n') {
				j++
			}
			if j < len(out) && (out[j] == '}' || out[j] == ']') {
				continue
			}
		}
		clean = append(clean, ch)
	}
	return clean
}
