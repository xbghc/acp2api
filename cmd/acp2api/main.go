package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"acp2api/internal/config"
	"acp2api/internal/pool"
	"acp2api/internal/server"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "acp2api error: %v\n", err)
		os.Exit(1)
	}
}

func run(rawArgs []string) error {
	// Split args on "--" so users can pass: acp2api [flags] -- <acp-command> [acp-args...]
	flagArgs := rawArgs
	var passthroughCmd []string
	for i, a := range rawArgs {
		if a == "--" {
			flagArgs = rawArgs[:i]
			passthroughCmd = rawArgs[i+1:]
			break
		}
	}

	fs := flag.NewFlagSet("acp2api", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, `acp2api - Convert stdio ACP (Agent Client Protocol) servers into OpenAI & Anthropic HTTP APIs

Usage:
  acp2api -config config.json
  acp2api [flags] -- <acp-command> [acp-args...]
  acp2api -acp-cmd "gemini" -acp-args "--experimental-acp" -listen :8080 -idle-sessions 3

Flags:
`)
		fs.PrintDefaults()
	}

	var (
		configPath         = fs.String("config", "", "Path to JSON/JSONC configuration file")
		printDefaultConfig = fs.Bool("print-default-config", false, "Print default JSON configuration to stdout and exit")
		listenAddr         = fs.String("listen", "", "HTTP listen address (default :8080)")
		apiKey             = fs.String("api-key", "", "Optional comma-separated API keys for Bearer / x-api-key auth")
		acpCmd             = fs.String("acp-cmd", "", "ACP server executable launched over stdio")
		acpArgs            = fs.String("acp-args", "", "Space-separated arguments for -acp-cmd (or pass command after '--')")
		acpCwd             = fs.String("cwd", "", "Working directory for ACP sessions")
		processes          = fs.Int("processes", 0, "Number of ACP stdio subprocesses in the worker pool (default 1)")
		idleSessions       = fs.Int("idle-sessions", -1, "Number of pre-warmed idle sessions maintained ready (default 3)")
		maxCached          = fs.Int("max-cached-sessions", -1, "Max cached multi-turn sessions (default 64)")
		reuseSessions      = fs.String("reuse-sessions", "", "Reuse stateful ACP sessions across multi-turn requests (true/false)")
		permPolicy         = fs.String("permission-policy", "", "ACP permission callback policy: allow_all, allow_read_only, deny_all")
		enableFSRead       = fs.Bool("enable-fs-read", false, "Expose fs/read_text_file capability to ACP agent")
		enableFSWrite      = fs.Bool("enable-fs-write", false, "Expose fs/write_text_file capability to ACP agent")
		enableTerminal     = fs.Bool("enable-terminal", false, "Expose terminal/* capabilities to ACP agent")
		defaultModel       = fs.String("default-model", "", "Default model ID reported on /v1/models")
		logLevel           = fs.String("log-level", "", "Log level: debug, info, warn, error")
		logFormat          = fs.String("log-format", "", "Log format: text, json")
	)

	if err := fs.Parse(flagArgs); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	if *printDefaultConfig {
		def := config.Default()
		def.ACP.Command = "your-acp-agent"
		def.ACP.Args = []string{"--stdio"}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(def)
	}

	cfg := config.Default()
	if strings.TrimSpace(*configPath) != "" {
		loaded, err := config.LoadFile(*configPath)
		if err != nil {
			return err
		}
		cfg = loaded
	}

	cfg.ApplyEnv()

	// Apply explicit CLI flag overrides.
	if *listenAddr != "" {
		cfg.Server.ListenAddr = *listenAddr
	}
	if *apiKey != "" {
		var keys []string
		for _, k := range strings.Split(*apiKey, ",") {
			if t := strings.TrimSpace(k); t != "" {
				keys = append(keys, t)
			}
		}
		cfg.Server.APIKeys = keys
	}
	if *acpCmd != "" {
		cfg.ACP.Command = *acpCmd
	}
	if *acpArgs != "" {
		cfg.ACP.Args = strings.Fields(*acpArgs)
	}
	if len(passthroughCmd) > 0 {
		cfg.ACP.Command = passthroughCmd[0]
		cfg.ACP.Args = passthroughCmd[1:]
	}
	if *acpCwd != "" {
		cfg.ACP.Cwd = *acpCwd
	}
	if *processes > 0 {
		cfg.Pool.ProcessCount = *processes
	}
	if *idleSessions >= 0 {
		cfg.Pool.IdlePoolSize = *idleSessions
	}
	if *maxCached >= 0 {
		cfg.Pool.MaxCachedSessions = *maxCached
	}
	if *reuseSessions != "" {
		cfg.Pool.ReuseConversations = strings.EqualFold(*reuseSessions, "true") || *reuseSessions == "1"
	}
	if *permPolicy != "" {
		cfg.ACP.PermissionPolicy = config.PermissionPolicy(*permPolicy)
	}
	if *enableFSRead {
		cfg.ACP.Capabilities.EnableFSRead = true
	}
	if *enableFSWrite {
		cfg.ACP.Capabilities.EnableFSWrite = true
	}
	if *enableTerminal {
		cfg.ACP.Capabilities.EnableTerminal = true
	}
	if *defaultModel != "" {
		cfg.ACP.DefaultModel = *defaultModel
	}
	if *logLevel != "" {
		cfg.Log.Level = *logLevel
	}
	if *logFormat != "" {
		cfg.Log.Format = *logFormat
	}

	if err := cfg.NormalizeAndValidate(); err != nil {
		return err
	}

	logger := buildLogger(cfg.Log)

	sigCtx, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	sessionPool := pool.New(cfg, logger)
	if err := sessionPool.Start(sigCtx); err != nil {
		return err
	}
	defer func() {
		_ = sessionPool.Close()
	}()

	apiServer := server.New(cfg, sessionPool, logger)
	errCh := make(chan error, 1)
	go func() {
		if err := apiServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		} else {
			errCh <- nil
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-sigCtx.Done():
		logger.Info("shutting down acp2api gracefully...")
		shutCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = apiServer.Shutdown(shutCtx)
		return sessionPool.Close()
	}
}

func buildLogger(lc config.LogConfig) *slog.Logger {
	var lvl slog.Level
	switch strings.ToLower(strings.TrimSpace(lc.Level)) {
	case "debug":
		lvl = slog.LevelDebug
	case "warn", "warning":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	opts := &slog.HandlerOptions{Level: lvl}
	if strings.EqualFold(lc.Format, "json") {
		return slog.New(slog.NewJSONHandler(os.Stderr, opts))
	}
	return slog.New(slog.NewTextHandler(os.Stderr, opts))
}
