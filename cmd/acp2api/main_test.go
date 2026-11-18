package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"acp2api/internal/config"
)

func TestRunCLIFlagsHelpPrintConfigAndValidation(t *testing.T) {
	// 1. -help exits cleanly with nil error.
	if err := run([]string{"-h"}); err != nil {
		t.Fatalf("expected -h to return nil, got: %v", err)
	}

	// 2. -print-default-config exits cleanly with nil error.
	if err := run([]string{"-print-default-config"}); err != nil {
		t.Fatalf("expected -print-default-config to return nil, got: %v", err)
	}

	// 3. Missing ACP command returns clear validation error.
	if err := run([]string{"-listen", "127.0.0.1:0"}); err == nil || !strings.Contains(err.Error(), "acp.command") {
		t.Fatalf("expected missing acp.command error, got: %v", err)
	}

	// 4. Nonexistent -config file returns error.
	if err := run([]string{"-config", "/nonexistent/acp2api.json"}); err == nil {
		t.Fatalf("expected error for missing config file")
	}

	// 5. Exercise full flag parsing + passthrough "--" + invalid permission policy rejection (verifies all flags wire through).
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "base.json")
	_ = os.WriteFile(cfgFile, []byte(`{"acp":{"command":"placeholder"}}`), 0o644)

	err := run([]string{
		"-config", cfgFile,
		"-listen", "127.0.0.1:0",
		"-api-key", "k1,k2",
		"-acp-cmd", "overridden-cmd",
		"-acp-args", "--a --b",
		"-cwd", dir,
		"-processes", "2",
		"-idle-sessions", "1",
		"-max-cached-sessions", "10",
		"-reuse-sessions", "true",
		"-permission-policy", "bad_policy",
		"-enable-fs-read",
		"-enable-fs-write",
		"-enable-terminal",
		"-default-model", "my-model",
		"-log-level", "debug",
		"-log-format", "json",
		"--", "final-cmd", "--stdio",
	})
	if err == nil || !strings.Contains(err.Error(), "permission_policy") {
		t.Fatalf("expected permission_policy validation error after parsing all flags, got: %v", err)
	}
}

func TestBuildLoggerVariants(t *testing.T) {
	for _, lvl := range []string{"debug", "info", "warn", "error", ""} {
		for _, fmtType := range []string{"text", "json"} {
			l := buildLogger(config.LogConfig{Level: lvl, Format: fmtType})
			if l == nil {
				t.Fatalf("buildLogger returned nil for %s/%s", lvl, fmtType)
			}
		}
	}
}
