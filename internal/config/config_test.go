package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLoadFileJSONCAndValidation(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.jsonc")
	content := `{
		// HTTP server settings
		"server": {
			"listen_addr": "127.0.0.1:18080",
			"api_keys": ["sk-test-1", "sk-test-2",],
		},
		/* ACP stdio process settings
		   multi-line block comment */
		"acp": {
			"command": "mock-acp",
			"args": ["--stdio", "--url=https://example.com//not-a-comment", "/*also-string*/", "a,]b"],
			"permission_policy": "allow_read_only",
			"mcp_servers": [
				{"name": "fs", "command": "mcp-fs"}
			],
			"model_aliases": {
				"gpt-4o": "claude-sonnet-4-5",
				"claude-default": "",
			},
		},
		"pool": {
			"process_count": 2,
			"idle_pool_size": 4,
			"reuse_conversations": true,
		},
	}`
	if err := os.WriteFile(cfgPath, []byte(content), 0o644); err != nil {
		t.Fatalf("write temp config: %v", err)
	}

	cfg, err := LoadFile(cfgPath)
	if err != nil {
		t.Fatalf("LoadFile failed: %v", err)
	}
	if err := cfg.NormalizeAndValidate(); err != nil {
		t.Fatalf("NormalizeAndValidate failed: %v", err)
	}

	if cfg.Server.ListenAddr != "127.0.0.1:18080" {
		t.Fatalf("unexpected listen_addr: %s", cfg.Server.ListenAddr)
	}
	if len(cfg.Server.APIKeys) != 2 {
		t.Fatalf("expected 2 api keys, got %d", len(cfg.Server.APIKeys))
	}
	if len(cfg.ACP.Args) != 4 || cfg.ACP.Args[1] != "--url=https://example.com//not-a-comment" || cfg.ACP.Args[2] != "/*also-string*/" || cfg.ACP.Args[3] != "a,]b" {
		t.Fatalf("stripJSONC corrupted string literals: %+v", cfg.ACP.Args)
	}
	if len(cfg.ACP.MCPServers) != 1 || cfg.ACP.MCPServers[0].Args == nil || cfg.ACP.MCPServers[0].Env == nil {
		t.Fatalf("expected MCPServers slices normalized to non-nil: %+v", cfg.ACP.MCPServers)
	}
	if cfg.Pool.ProcessCount != 2 || cfg.Pool.IdlePoolSize != 4 {
		t.Fatalf("unexpected pool config: %+v", cfg.Pool)
	}

	acpMod, disp := cfg.ResolveModel("gpt-4o")
	if acpMod != "claude-sonnet-4-5" || disp != "gpt-4o" {
		t.Fatalf("unexpected ResolveModel for alias: (%q, %q)", acpMod, disp)
	}
	acpMod, disp = cfg.ResolveModel("claude-default")
	if acpMod != "" || disp != "claude-default" {
		t.Fatalf("unexpected ResolveModel for empty alias: (%q, %q)", acpMod, disp)
	}
	acpMod, disp = cfg.ResolveModel("")
	if acpMod != "" || disp != "acp-default" {
		t.Fatalf("unexpected ResolveModel for empty string: (%q, %q)", acpMod, disp)
	}
	acpMod, disp = cfg.ResolveModel("custom-model-v2")
	if acpMod != "custom-model-v2" || disp != "custom-model-v2" {
		t.Fatalf("expected pass-through model, got (%q, %q)", acpMod, disp)
	}

	cfg.ACP.PassThroughModels = false
	acpMod, disp = cfg.ResolveModel("custom-model-v2")
	if acpMod != "" || disp != "custom-model-v2" {
		t.Fatalf("expected disabled pass-through to return empty ACP model, got (%q, %q)", acpMod, disp)
	}
}

func TestLoadFileErrors(t *testing.T) {
	if _, err := LoadFile("/nonexistent/path/to/config.json"); err == nil {
		t.Fatalf("expected error loading nonexistent file")
	}

	dir := t.TempDir()
	unknownFieldPath := filepath.Join(dir, "bad.json")
	_ = os.WriteFile(unknownFieldPath, []byte(`{"unknown_key": 123}`), 0o644)
	if _, err := LoadFile(unknownFieldPath); err == nil {
		t.Fatalf("expected error for unknown field in config")
	}
}

func TestApplyEnvAndValidationEdgeCases(t *testing.T) {
	t.Setenv("ACP2API_LISTEN", "127.0.0.1:9090")
	t.Setenv("ACP2API_API_KEY", " key1 , key2 , ")
	t.Setenv("ACP2API_COMMAND", "env-acp-bin")
	t.Setenv("ACP2API_ARGS", "--flag1 --flag2")
	t.Setenv("ACP2API_CWD", t.TempDir())
	t.Setenv("ACP2API_PERMISSION_POLICY", "deny_all")
	t.Setenv("ACP2API_LOG_LEVEL", "debug")

	cfg := Default()
	cfg.ApplyEnv()

	if cfg.Server.ListenAddr != "127.0.0.1:9090" {
		t.Fatalf("unexpected ListenAddr: %s", cfg.Server.ListenAddr)
	}
	if len(cfg.Server.APIKeys) != 2 || cfg.Server.APIKeys[0] != "key1" || cfg.Server.APIKeys[1] != "key2" {
		t.Fatalf("unexpected APIKeys: %+v", cfg.Server.APIKeys)
	}
	if cfg.ACP.Command != "env-acp-bin" || len(cfg.ACP.Args) != 2 {
		t.Fatalf("unexpected ACP command/args: %s %+v", cfg.ACP.Command, cfg.ACP.Args)
	}
	if cfg.ACP.PermissionPolicy != PermissionDenyAll || cfg.Log.Level != "debug" {
		t.Fatalf("unexpected policy/log: %s %s", cfg.ACP.PermissionPolicy, cfg.Log.Level)
	}

	// Test fallback defaults when zero/negative values are supplied.
	cfg.Server.ListenAddr = ""
	cfg.Server.ReadHeaderTimeoutSecs = 0
	cfg.Server.RequestTimeoutSecs = -1
	cfg.ACP.InitializeTimeoutSecs = 0
	cfg.ACP.SessionNewTimeoutSecs = 0
	cfg.ACP.PermissionPolicy = ""
	cfg.ACP.MCPServers = nil
	cfg.ACP.DefaultModel = ""
	cfg.Pool.ProcessCount = 0
	cfg.Pool.IdlePoolSize = -5
	cfg.Pool.MaxCachedSessions = -2
	cfg.Pool.IdleSessionTTLSeconds = 0
	cfg.Pool.CachedSessionTTLSeconds = 0

	if err := cfg.NormalizeAndValidate(); err != nil {
		t.Fatalf("NormalizeAndValidate failed: %v", err)
	}

	if cfg.Server.ListenAddr != ":8080" || cfg.ACP.PermissionPolicy != PermissionAllowAll || cfg.ACP.DefaultModel != "acp-default" {
		t.Fatalf("unexpected normalized defaults: %+v", cfg)
	}
	if cfg.Pool.ProcessCount != 1 || cfg.Pool.IdlePoolSize != 0 || cfg.Pool.MaxCachedSessions != 0 {
		t.Fatalf("unexpected normalized pool counts: %+v", cfg.Pool)
	}
	if cfg.InitializeTimeout() != 30*time.Second ||
		cfg.SessionNewTimeout() != 30*time.Second ||
		cfg.RequestTimeout() != 600*time.Second ||
		cfg.IdleSessionTTL() != 1800*time.Second ||
		cfg.CachedSessionTTL() != 1800*time.Second {
		t.Fatalf("unexpected timeout durations")
	}

	// Missing command must fail validation.
	missingCmd := Default()
	if err := missingCmd.NormalizeAndValidate(); err == nil || !strings.Contains(err.Error(), "acp.command") {
		t.Fatalf("expected missing acp.command validation error, got: %v", err)
	}

	// Invalid permission_policy must fail validation.
	badPerm := Default()
	badPerm.ACP.Command = "foo"
	badPerm.ACP.PermissionPolicy = "invalid_policy"
	if err := badPerm.NormalizeAndValidate(); err == nil || !strings.Contains(err.Error(), "permission_policy") {
		t.Fatalf("expected invalid permission_policy error, got: %v", err)
	}
}
