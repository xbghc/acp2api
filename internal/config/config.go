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

// MCPServerConfig defines an MCP server passed to ACP session/new.
type MCPServerConfig struct {
	Name    string    `json:"name"`
	Command string    `json:"command"`
	Args    []string  `json:"args"`
	Env     []EnvPair `json:"env"`
}

// ClientCapabilitiesConfig controls which local host capabilities acp2api exposes to the ACP agent.
type ClientCapabilitiesConfig struct {
	EnableFSRead   bool `json:"enable_fs_read"`
	EnableFSWrite  bool `json:"enable_fs_write"`
	EnableTerminal bool `json:"enable_terminal"`
	RestrictToCwd  bool `json:"restrict_to_cwd"`
}

// ACPConfig configures how the ACP stdio subprocess is launched and managed.
type ACPConfig struct {
	Command               string                   `json:"command"`
	Args                  []string                 `json:"args"`
	Env                   map[string]string        `json:"env"`
	Cwd                   string                   `json:"cwd"`
	InitializeTimeoutSecs int                      `json:"initialize_timeout_seconds"`
	SessionNewTimeoutSecs int                      `json:"session_new_timeout_seconds"`
	PermissionPolicy      PermissionPolicy         `json:"permission_policy"`
	Capabilities          ClientCapabilitiesConfig `json:"capabilities"`
	MCPServers            []MCPServerConfig        `json:"mcp_servers"`
	DefaultModel          string                   `json:"default_model"`
	ModelAliases          map[string]string        `json:"model_aliases"`
	PassThroughModels     bool                     `json:"pass_through_models"`
	ExposeThoughts        bool                     `json:"expose_thoughts"`
}

// PoolConfig configures the ACP process pool, pre-warmed idle session pool, and conversation cache.
type PoolConfig struct {
	ProcessCount            int  `json:"process_count"`
	IdlePoolSize            int  `json:"idle_pool_size"`
	MaxActiveSessions       int  `json:"max_active_sessions"`
	ReuseConversations      bool `json:"reuse_conversations"`
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
			APIKeys:                nil,
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
		},
		Pool: PoolConfig{
			ProcessCount:            1,
			IdlePoolSize:            3,
			MaxActiveSessions:       32,
			ReuseConversations:      true,
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
		if c.ACP.MCPServers[i].Args == nil {
			c.ACP.MCPServers[i].Args = []string{}
		}
		if c.ACP.MCPServers[i].Env == nil {
			c.ACP.MCPServers[i].Env = []EnvPair{}
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

		// Line comment
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

		// Block comment
		if ch == '/' && i+1 < len(src) && src[i+1] == '*' {
			i += 2
			for i+1 < len(src) && !(src[i] == '*' && src[i+1] == '/') {
				if src[i] == '\n' {
					out = append(out, '\n')
				}
				i++
			}
			i++ // skip '/'
			continue
		}

		out = append(out, ch)
	}

	// Second pass: remove trailing commas before '}' or ']'
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
