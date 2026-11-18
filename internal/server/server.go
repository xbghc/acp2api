package server

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"acp2api/internal/config"
	"acp2api/internal/pool"
)

// Server exposes OpenAI-compatible and Anthropic-compatible HTTP endpoints backed by the ACP session pool.
type Server struct {
	cfg     config.Config
	pool    *pool.Pool
	logger  *slog.Logger
	httpSrv *http.Server
	handler http.Handler
}

func New(cfg config.Config, sessionPool *pool.Pool, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	s := &Server{
		cfg:    cfg,
		pool:   sessionPool,
		logger: logger,
	}

	mux := http.NewServeMux()

	// Health & Pool Observability
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	mux.HandleFunc("GET /v1/pool/stats", s.handlePoolStats)

	// OpenAI Compatible Endpoints (including / and /v1/v1 convenience aliases)
	for _, p := range []string{"/v1/models", "/models", "/v1/v1/models"} {
		mux.HandleFunc("GET "+p, s.handleOpenAIListModels)
		mux.HandleFunc("GET "+p+"/{model}", s.handleOpenAIRetrieveModel)
	}
	for _, p := range []string{"/v1/chat/completions", "/chat/completions", "/v1/v1/chat/completions"} {
		mux.HandleFunc("POST "+p, s.handleOpenAIChatCompletions)
	}
	for _, p := range []string{"/v1/responses", "/responses", "/v1/v1/responses"} {
		mux.HandleFunc("POST "+p, s.handleOpenAIResponses)
	}

	// Anthropic Compatible Endpoints (including / and /v1/v1 convenience aliases)
	for _, p := range []string{"/v1/messages", "/messages", "/v1/v1/messages"} {
		mux.HandleFunc("POST "+p, s.handleAnthropicMessages)
		mux.HandleFunc("POST "+p+"/count_tokens", s.handleAnthropicCountTokens)
	}

	// Default 404 JSON handler
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if isAnthropicPath(r.URL.Path) {
			writeAnthropicError(w, http.StatusNotFound, "not_found_error", "route not found: "+r.Method+" "+r.URL.Path)
			return
		}
		writeOpenAIError(w, http.StatusNotFound, "invalid_request_error", "route not found: "+r.Method+" "+r.URL.Path)
	})

	s.handler = s.withMiddlewares(mux)
	s.httpSrv = &http.Server{
		Addr:              cfg.Server.ListenAddr,
		Handler:           s.handler,
		ReadHeaderTimeout: time.Duration(cfg.Server.ReadHeaderTimeoutSecs) * time.Second,
	}
	return s
}

// Handler returns the root HTTP handler (useful for httptest).
func (s *Server) Handler() http.Handler {
	return s.handler
}

func (s *Server) ListenAndServe() error {
	s.logger.Info("HTTP API server listening",
		"addr", s.cfg.Server.ListenAddr,
		"openai_chat", "/v1/chat/completions",
		"openai_responses", "/v1/responses",
		"anthropic_messages", "/v1/messages",
		"pool_stats", "/v1/pool/stats",
	)
	return s.httpSrv.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpSrv.Shutdown(ctx)
}

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	stats := s.pool.Stats()
	status := http.StatusOK
	state := "ok"
	if stats.HealthyWorkers == 0 {
		status = http.StatusServiceUnavailable
		state = "degraded"
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":              state,
		"healthy_workers":     stats.HealthyWorkers,
		"ready_idle_sessions": stats.ReadyIdleSessions,
		"active_turns":        stats.ActiveTurns,
	})
}

func (s *Server) handlePoolStats(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s.pool.Stats())
}
