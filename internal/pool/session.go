package pool

import (
	"sync/atomic"
	"time"

	"acp2api/internal/acp"
)

type acpWorker struct {
	id            string
	slot          int
	client        *acp.Client
	activeTurns   atomic.Int64
	liveSessions  atomic.Int64
	totalSessions atomic.Int64
	rotating      atomic.Bool // replacement worker spawn is in flight
	draining      atomic.Bool // replacement worker is live; stop assigning new work and close when idle
}

// ManagedSession tracks an initialized ACP session either waiting in the pre-warmed idle pool,
// checked out for an active turn, or stored in the multi-turn conversation cache.
type ManagedSession struct {
	ID              string
	Worker          *acpWorker
	Meta            acp.SessionNewResult
	CurrentModel    string
	CurrentMode     string
	CurrentThinking string
	HistoryHash     string
	SessionHint     string
	TurnCount       int
	CreatedAt       time.Time
	LastUsedAt      time.Time
}

// Stats provides real-time observability into the process pool and session pools.
type Stats struct {
	TotalWorkers        int   `json:"total_workers"`
	HealthyWorkers      int   `json:"healthy_workers"`
	TargetIdleSessions  int   `json:"target_idle_sessions"`
	ReadyIdleSessions   int   `json:"ready_idle_sessions"`
	WarmingSessions     int   `json:"warming_sessions"`
	CachedConversations int   `json:"cached_conversations"`
	ActiveTurns         int64 `json:"active_turns"`
	TotalRequests       int64 `json:"total_requests"`
	IdlePoolHits        int64 `json:"idle_pool_hits"`
	ConversationHits    int64 `json:"conversation_reuse_hits"`
	SessionForkHits     int64 `json:"session_fork_hits"`
	ColdCreations       int64 `json:"cold_session_creations"`
	WorkerRestarts      int64 `json:"worker_restarts"`
}
