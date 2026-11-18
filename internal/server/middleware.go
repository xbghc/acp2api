package server

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

var reqSeq atomic.Uint64

func newShortID(prefix string) string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err == nil {
		return prefix + hex.EncodeToString(b[:])
	}
	return fmt.Sprintf("%s%d", prefix, reqSeq.Add(1))
}

func extractClientAPIKey(r *http.Request) string {
	if auth := strings.TrimSpace(r.Header.Get("Authorization")); auth != "" {
		if len(auth) > 7 && strings.EqualFold(auth[:7], "Bearer ") {
			return strings.TrimSpace(auth[7:])
		}
		return auth
	}
	if key := strings.TrimSpace(r.Header.Get("X-Api-Key")); key != "" {
		return key
	}
	return ""
}

func isAnthropicPath(path string) bool {
	return strings.HasPrefix(path, "/v1/messages") || strings.HasPrefix(path, "/messages")
}

func writeOpenAIError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    errType,
			"param":   nil,
			"code":    nil,
		},
	})
}

func writeAnthropicError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    errType,
			"message": message,
		},
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (sr *statusRecorder) WriteHeader(code int) {
	sr.status = code
	sr.ResponseWriter.WriteHeader(code)
}

func (sr *statusRecorder) Flush() {
	if f, ok := sr.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (s *Server) withMiddlewares(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key, Anthropic-Version, Anthropic-Beta, X-Session-ID, OpenAI-Beta, OpenAI-Organization")
		w.Header().Set("Access-Control-Expose-Headers", "X-ACP-Session-ID, X-ACP-Session-Reused")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		defer func() {
			if rec := recover(); rec != nil {
				s.logger.Error("recovered HTTP handler panic", "panic", rec, "path", r.URL.Path)
				if isAnthropicPath(r.URL.Path) {
					writeAnthropicError(w, http.StatusInternalServerError, "api_error", "internal server error")
				} else {
					writeOpenAIError(w, http.StatusInternalServerError, "server_error", "internal server error")
				}
			}
		}()

		// Health endpoint bypasses API key check for container readiness probes.
		if r.URL.Path != "/healthz" && len(s.cfg.Server.APIKeys) > 0 {
			provided := extractClientAPIKey(r)
			matched := false
			for _, allowed := range s.cfg.Server.APIKeys {
				if len(provided) == len(allowed) && subtle.ConstantTimeCompare([]byte(provided), []byte(allowed)) == 1 {
					matched = true
					break
				}
			}
			if !matched {
				if isAnthropicPath(r.URL.Path) {
					writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", "invalid or missing API key")
				} else {
					writeOpenAIError(w, http.StatusUnauthorized, "invalid_request_error", "invalid or missing API key")
				}
				return
			}
		}

		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)

		s.logger.LogAttrs(r.Context(), slog.LevelInfo, "http request",
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int("status", rec.status),
			slog.Duration("duration", time.Since(start)),
		)
	})
}

func resolveSessionHint(r *http.Request, fallback string) string {
	if h := strings.TrimSpace(r.Header.Get("X-Session-ID")); h != "" {
		return h
	}
	return strings.TrimSpace(fallback)
}
