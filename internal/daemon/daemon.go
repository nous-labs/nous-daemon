// Package daemon implements the Nous daemon — the persistent event loop
// that listens for messages, thinks, acts, and remembers.
package daemon

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/nous-labs/nous/internal/channel/matrix"
	"github.com/nous-labs/nous/internal/llm"
	"github.com/nous-labs/nous/internal/tools"
	"github.com/nous-labs/nous/pkg/brain"
	"github.com/nous-labs/nous/pkg/channel"
	coredaemon "github.com/nous-labs/nous/pkg/daemon"
	"github.com/nous-labs/nous/pkg/dream"
	"github.com/nous-labs/nous/pkg/embeddings"
)

// chatSystemPrompt primes the direct LLM chat path for mobile messaging behavior.
const chatSystemPrompt = `You are Nous, a persistent AI assistant chatting via Matrix (mobile messaging).

Behavioral rules:
- Be concise and conversational — this is mobile chat, not a terminal.
- No markdown headers, no code fences unless explicitly asked.
- Short, natural responses. Think texting, not writing essays.
- You have persistent memory. Reference past decisions and context naturally.
- If asked to do coding/project work, say you'll dispatch it to OpenCode.
- You are a thinking partner in chat, not a code executor.

Tool usage:
- You have tools for fetching live data. USE THEM instead of guessing.
- When asked about software versions, releases, or current info — use github_latest_release or web_fetch.
- When asked about past decisions, project context, or memory — use brain_recall.
- When a decision is made or important info shared — use brain_capture to store it.
- NEVER make up version numbers or URLs. If you don't know, use a tool to look it up.`

const (
	// maxHistoryPerRoom caps message count even for large context windows.
	maxHistoryPerRoom = 100
	// defaultHistoryBudgetChars is the fallback if no context window is configured.
	defaultHistoryBudgetChars = 8000
	// historyBudgetRatio is the fraction of context window allocated to conversation history.
	// Remainder covers system prompt, output tokens, and safety margin.
	historyBudgetRatio = 0.60
	// charsPerToken is a rough estimate for converting token budgets to character budgets.
	charsPerToken = 4
)

// Daemon is the main Nous process.
type Daemon struct {
	brain    *brain.Brain
	config   *Config
	matrix   *matrix.Channel
	router   *llm.Router
	opencode *tools.OpenCodeClient
	dispatch *tools.DispatchClient

	// Runtime state
	identity     string // bootstrap context for system prompt
	ocSessionID  string // persistent OpenCode chat session
	hasDirectLLM bool   // whether direct LLM providers are configured
	startedAt    time.Time
	healthy      bool

	// Conversation memory — sliding window per room
	history   map[string][]llm.Message // roomID → recent messages
	historyMu sync.Mutex

	// Route stickiness — continue routing to same backend for follow-ups
	lastRoute   map[string]string    // roomID → last route used
	lastRouteAt map[string]time.Time // roomID → when last route was used

	// OC session protection (lightweight — rarely changes)
	ocMu sync.Mutex
	// Event bus for broadcasting to workspace TUI clients
	events *coredaemon.EventBus
	// Semantic memory (optional, requires pgvector + TEI)
	embedStore *embeddings.Store
	teiClient  *embeddings.TEIClient
	embedMu    sync.RWMutex // protects embedStore/teiClient for lazy reconnect
	// Dream worker (autonomous memory maintenance)
	dreamer *dream.Worker
}

// New creates a new daemon instance.
func New(b *brain.Brain, cfg *Config) (*Daemon, error) {
	d := &Daemon{
		brain:       b,
		config:      cfg,
		history:     make(map[string][]llm.Message),
		lastRoute:   make(map[string]string),
		lastRouteAt: make(map[string]time.Time),
		startedAt:   time.Now(),
		events:      coredaemon.NewEventBus(),
	}

	// Initialize Matrix channel
	d.matrix = matrix.New(matrix.Config{
		Homeserver:   cfg.Matrix.Homeserver,
		UserID:       cfg.Matrix.UserID,
		Password:     cfg.Matrix.Password,
		ServerName:   cfg.Matrix.ServerName,
		AllowedUsers: cfg.Matrix.AllowedUsers,
		DataDir:      cfg.Matrix.DataDir,
	})

	// Load AuthStore from OpenCode's auth.json (if configured).
	// This enables OAuth-based providers (e.g., Anthropic MAX plan).
	var authStore *llm.AuthStore
	if cfg.AuthJSONPath != "" {
		var err error
		authStore, err = llm.NewAuthStore(cfg.AuthJSONPath)
		if err != nil {
			slog.Warn("failed to load auth.json, falling back to config API keys",
				"path", cfg.AuthJSONPath, "error", err)
		} else {
			slog.Info("auth store loaded", "path", cfg.AuthJSONPath, "providers", authStore.ListProviders())
		}
	}

	// Initialize LLM providers
	providers := make(map[llm.Tier]llm.Provider)

	// Deep tier (Claude) — prefer OAuth from auth.json, fall back to static API key
	if authStore != nil && authStore.Get("anthropic") != nil {
		providers[llm.TierDeep] = llm.NewAnthropicOAuth(authStore, cfg.LLM.Deep.Model)
		slog.Info("LLM provider configured",
			"tier", "deep",
			"provider", "anthropic",
			"auth", "oauth",
			"model", cfg.LLM.Deep.Model,
		)
	} else if cfg.LLM.Deep.APIKey != "" {
		providers[llm.TierDeep] = llm.NewAnthropic(cfg.LLM.Deep.APIKey, cfg.LLM.Deep.Model)
		slog.Info("LLM provider configured",
			"tier", "deep",
			"provider", cfg.LLM.Deep.Provider,
			"auth", "api_key",
			"model", cfg.LLM.Deep.Model,
		)
	}

	// Fast tier (Kimi / Anthropic-compat or OpenAI-compat)
	if cfg.LLM.Fast.APIKey != "" && cfg.LLM.Fast.BaseURL != "" {
		if cfg.LLM.Fast.Provider == "kimi" {
			// Kimi uses Anthropic-format API at /coding/v1
			providers[llm.TierFast] = llm.NewAnthropicCompat(
				cfg.LLM.Fast.Provider,
				cfg.LLM.Fast.BaseURL,
				cfg.LLM.Fast.APIKey,
				cfg.LLM.Fast.Model,
			)
		} else {
			providers[llm.TierFast] = llm.NewOpenAICompat(
				cfg.LLM.Fast.Provider,
				cfg.LLM.Fast.BaseURL,
				cfg.LLM.Fast.APIKey,
				cfg.LLM.Fast.Model,
			)
		}
		slog.Info("LLM provider configured",
			"tier", "fast",
			"provider", cfg.LLM.Fast.Provider,
			"model", cfg.LLM.Fast.Model,
		)
	}

	// Mid tier (Copilot) — uses OAuth from auth.json (github-copilot entry)
	if cfg.LLM.Mid.Provider == "copilot" && authStore != nil && authStore.Get("github-copilot") != nil {
		providers[llm.TierMid] = llm.NewCopilot(authStore, cfg.LLM.Mid.Model)
		slog.Info("LLM provider configured",
			"tier", "mid",
			"provider", "copilot",
			"auth", "oauth",
			"model", cfg.LLM.Mid.Model,
		)
	} else if cfg.LLM.Mid.Provider == "copilot" {
		slog.Warn("copilot mid-tier configured but no github-copilot auth entry in auth.json")
	} else if cfg.LLM.Mid.APIKey != "" && cfg.LLM.Mid.BaseURL != "" {
		// Generic OpenAI-compat mid-tier (fallback)
		providers[llm.TierMid] = llm.NewOpenAICompat(
			cfg.LLM.Mid.Provider,
			cfg.LLM.Mid.BaseURL,
			cfg.LLM.Mid.APIKey,
			cfg.LLM.Mid.Model,
		)
		slog.Info("LLM provider configured",
			"tier", "mid",
			"provider", cfg.LLM.Mid.Provider,
			"model", cfg.LLM.Mid.Model,
		)
	}

	d.hasDirectLLM = len(providers) > 0
	d.router = llm.NewRouter(providers)

	// Initialize OpenCode client
	if cfg.OpenCode.APIUrl != "" {
		d.opencode = tools.NewOpenCode(cfg.OpenCode.APIUrl, cfg.OpenCode.Username, cfg.OpenCode.Password)
		slog.Info("OpenCode client configured", "url", cfg.OpenCode.APIUrl)
	}

	if !d.hasDirectLLM && d.opencode == nil {
		slog.Warn("no LLM providers and no OpenCode configured — chat will be unavailable")
	} else if !d.hasDirectLLM {
		slog.Info("no direct LLM providers — using OpenCode as chat backend")
	}

	// Restore persisted OC session ID from brain KV
	if sid, _ := b.KVGet("nous-daemon:oc-session-id"); sid != "" {
		d.ocSessionID = sid
		slog.Info("restored OpenCode session", "id", sid)
	}

	// Initialize dispatch client (routes tasks to host-side OpenCode instances)
	var dispatchTargets []tools.DispatchTarget
	for _, p := range cfg.Projects {
		if p.URL != "" {
			dispatchTargets = append(dispatchTargets, tools.DispatchTarget{Name: p.Name, URL: p.URL})
		}
	}
	if len(dispatchTargets) > 0 {
		d.dispatch = tools.NewDispatch(tools.DispatchConfig{
			Targets:  dispatchTargets,
			Username: cfg.OpenCode.Username,
			Password: cfg.OpenCode.Password,
			KV: tools.KVStore{
				Get: b.KVGet,
				Set: b.KVSet,
			},
		})
		slog.Info("dispatch configured", "projects", d.dispatch.ListProjects())
	}

	// Initialize semantic memory (optional, requires pgvector + TEI)
	// If pgvector is not ready yet (startup race), a background retry is started in Run().
	if cfg.Embeddings.Enabled && cfg.Embeddings.PostgresURL != "" && cfg.Embeddings.TEIURL != "" {
		if !d.tryInitSemanticMemory() {
			slog.Info("semantic memory will retry in background when pgvector becomes available")
		}
	} else if cfg.Embeddings.Enabled {
		slog.Warn("semantic memory enabled but missing config",
			"has_pg_url", cfg.Embeddings.PostgresURL != "",
			"has_tei_url", cfg.Embeddings.TEIURL != "",
		)
	}
	return d, nil
}

// tryInitSemanticMemory attempts to connect to pgvector and initialize the embedding store.
// Returns true if successful, false if connection failed (caller should retry later).
func (d *Daemon) tryInitSemanticMemory() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	store, err := embeddings.NewStore(ctx, d.config.Embeddings.PostgresURL)
	if err != nil {
		slog.Warn("semantic memory unavailable, pgvector connection failed", "error", err)
		return false
	}

	if err := store.Init(ctx); err != nil {
		slog.Warn("semantic memory unavailable, schema init failed", "error", err)
		store.Close()
		return false
	}

	d.embedMu.Lock()
	d.embedStore = store
	d.teiClient = embeddings.NewTEIClient(d.config.Embeddings.TEIURL)
	d.embedMu.Unlock()

	slog.Info("semantic memory initialized",
		"postgres", d.config.Embeddings.PostgresURL,
		"tei", d.config.Embeddings.TEIURL,
	)
	return true
}

// retrySemanticMemory runs a background loop to reconnect pgvector.
// Tries every 30s for up to 10 minutes, then gives up.
func (d *Daemon) retrySemanticMemory(ctx context.Context) {
	const maxRetries = 20
	const retryInterval = 30 * time.Second

	for attempt := 1; attempt <= maxRetries; attempt++ {
		select {
		case <-ctx.Done():
			slog.Info("semantic memory retry cancelled")
			return
		case <-time.After(retryInterval):
		}

		slog.Info("retrying semantic memory connection", "attempt", attempt, "max", maxRetries)
		if d.tryInitSemanticMemory() {
			slog.Info("semantic memory reconnected, starting embedding sync")
			// Start the embedding sync worker now that we have a connection
			d.startEmbeddingSyncWorker(ctx)
			return
		}
	}

	slog.Error("semantic memory permanently unavailable after retries", "attempts", maxRetries)
}

// startEmbeddingSyncWorker starts the background embedding sync goroutine.
func (d *Daemon) startEmbeddingSyncWorker(ctx context.Context) {
	d.embedMu.RLock()
	store := d.embedStore
	tei := d.teiClient
	d.embedMu.RUnlock()

	if store == nil || tei == nil {
		return
	}

	syncInterval := 30 * time.Second
	if d.config.Embeddings.SyncInterval != "" {
		if parsed, err := time.ParseDuration(d.config.Embeddings.SyncInterval); err == nil {
			syncInterval = parsed
		}
	}
	batchSize := d.config.Embeddings.BatchSize
	if batchSize <= 0 {
		batchSize = 50
	}
	worker := embeddings.NewSyncWorker(d.brain, store, tei, syncInterval, batchSize)
	go worker.Run(ctx)
}

// Run starts the daemon event loop. Blocks until ctx is cancelled.
func (d *Daemon) Run(ctx context.Context) error {
	slog.Info("nous daemon running",
		"name", d.config.Name,
		"matrix", d.config.Matrix.Homeserver,
		"direct_llm", d.hasDirectLLM,
	)

	// Bootstrap — load identity context from brain
	bootstrap, err := d.brain.Bootstrap("omo-config")
	if err != nil {
		slog.Warn("bootstrap failed, continuing without context", "error", err)
	} else {
		d.identity = bootstrap
		slog.Info("brain bootstrapped", "context_len", len(bootstrap))
	}



	// Start health endpoint in background
	go d.serveHealth(ctx)

	// Start workspace API on Unix socket (for TUI connections)
	if d.config.Workspace.Enabled {
		if err := d.startWorkspace(ctx); err != nil {
			slog.Error("workspace API failed to start", "error", err)
		} else {
			d.events.Publish(coredaemon.Event{Type: coredaemon.EventStatus, Message: "workspace API ready"})
		}
	}

	// Start embedding sync worker (if semantic memory is available)
	// If not available, start background retry to reconnect when pgvector comes up.
	d.embedMu.RLock()
	hasEmbed := d.embedStore != nil && d.teiClient != nil
	d.embedMu.RUnlock()
	if hasEmbed {
		d.startEmbeddingSyncWorker(ctx)
	} else if d.config.Embeddings.Enabled && d.config.Embeddings.PostgresURL != "" {
		go d.retrySemanticMemory(ctx)
	}

	// Start dream worker (autonomous memory maintenance)
	dreamCfg := dream.DefaultConfig()
	if d.config.Dream.Interval != "" {
		if parsed, err := time.ParseDuration(d.config.Dream.Interval); err == nil {
			dreamCfg.Interval = parsed
		}
	}
	if !d.config.Dream.Disabled {
		d.dreamer = dream.NewWorker(d.brain, func(typ, msg string) {
			d.events.Publish(coredaemon.Event{Type: coredaemon.EventStatus, Message: "[dream] " + msg})
		}, dreamCfg)
		go d.dreamer.Run(ctx)
	} else {
		slog.Info("dream worker disabled by config")
	}

	// Start Matrix listener in background
	errCh := make(chan error, 1)
	go func() {
		slog.Info("starting matrix channel")
		err := d.matrix.Start(ctx, d.onMessage)
		if err != nil {
			errCh <- err
		}
	}()

	// Mark healthy once Matrix starts syncing (give it a moment)
	go func() {
		time.Sleep(2 * time.Second)
		d.healthy = true
	}()

	// Wait for shutdown or fatal error
	select {
	case <-ctx.Done():
		slog.Info("context cancelled, shutting down")
	case err := <-errCh:
		if err != nil && ctx.Err() == nil {
			return fmt.Errorf("matrix channel fatal error: %w", err)
		}
	}

	// Graceful shutdown
	d.healthy = false
	d.matrix.Stop()

	d.embedMu.RLock()
	if d.embedStore != nil {
		d.embedStore.Close()
	}
	d.embedMu.RUnlock()


	slog.Info("nous daemon shutting down")
	return nil
}

// serveHealth runs the daemon's HTTP API on :8080.
// Endpoints:
//   - GET /health — health check
//   - GET /v1/recall — semantic/hybrid memory recall
func (d *Daemon) serveHealth(ctx context.Context) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if d.healthy {
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"status":"ok","uptime":"%s"}`, time.Since(d.startedAt).Round(time.Second))
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, `{"status":"starting"}`)
		}
	})

	mux.HandleFunc("/v1/recall", d.handleRecall)
	srv := &http.Server{Addr: ":8080", Handler: mux}
	go func() {
		<-ctx.Done()
		srv.Close()
	}()

	slog.Info("API listening", "addr", ":8080", "endpoints", []string{"/health", "/v1/recall"})
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		slog.Warn("API server error", "error", err)
	}
}

// recallResponse is the JSON response for /v1/recall.
type recallResponse struct {
	Memories []recallMemory `json:"memories"`
	Method   string         `json:"method"`
	Query    string         `json:"query"`
	Count    int            `json:"count"`
}

// recallMemory is a single memory in the recall response.
type recallMemory struct {
	ID        int    `json:"id"`
	Type      string `json:"type"`
	Scope     string `json:"scope"`
	Content   string `json:"content"`
	Tags      string `json:"tags,omitempty"`
	Source    string `json:"source,omitempty"`
	CreatedAt string `json:"created_at"`
}

// handleRecall serves semantic/hybrid memory recall.
// Query params:
//   - q: search query (required)
//   - limit: max results (default 10)
//   - type: filter by memory type
//   - scope: filter by scope
//   - tags: filter by tags
//
// Uses hybrid search (vector + FTS5 + RRF) when semantic memory is available,
// falls back to keyword-only FTS5 search otherwise.
func (d *Daemon) handleRecall(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprint(w, `{"error":"method not allowed"}`)
		return
	}

	query := r.URL.Query().Get("q")
	if query == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":"missing required parameter: q"}`)
		return
	}

	limit := 10
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 && parsed <= 100 {
			limit = parsed
		}
	}

	var memories []brain.Memory
	var method string
	var err error

	// Try hybrid search first (semantic + keyword)
	d.embedMu.RLock()
	store := d.embedStore
	tei := d.teiClient
	d.embedMu.RUnlock()
	if store != nil && tei != nil {
		method = "hybrid"
		memories, err = embeddings.HybridSearch(r.Context(), query, d.brain, store, tei, limit)
		if err != nil {
			slog.Warn("hybrid search failed, falling back to keyword", "error", err)
			err = nil // reset for keyword fallback
			method = "keyword"
		}
	}

	// Keyword fallback or no semantic memory available
	if memories == nil {
		if method == "" {
			method = "keyword"
		}
		opts := brain.RecallOptions{
			Limit: limit,
			Type:  r.URL.Query().Get("type"),
			Scope: r.URL.Query().Get("scope"),
			Tags:  r.URL.Query().Get("tags"),
		}
		memories, err = d.brain.Recall(query, opts)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintf(w, `{"error":%q}`, err.Error())
			return
		}
	}

	// Build response
	result := recallResponse{
		Memories: make([]recallMemory, 0, len(memories)),
		Method:   method,
		Query:    query,
		Count:    len(memories),
	}

	for _, m := range memories {
		result.Memories = append(result.Memories, recallMemory{
			ID:        m.ID,
			Type:      m.Type,
			Scope:     m.Scope,
			Content:   m.Content,
			Tags:      m.Tags,
			Source:    m.Source,
			CreatedAt: m.CreatedAt.Format(time.RFC3339),
		})
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		slog.Warn("failed to encode recall response", "error", err)
	}
}

// onMessage handles incoming messages from any channel.
// Routing priority:
//  1. /oc prefix → force OpenCode (coding tasks)
//  2. Coding request detected + OpenCode available → OpenCode
//  3. Direct LLM available → direct LLM (with OC fallback on failure)
//  4. OpenCode only → OpenCode
func (d *Daemon) onMessage(ctx context.Context, msg channel.Message) error {
	response, route, err := d.processMessage(ctx, msg)
	if err != nil {
		return err
	}

	// Send response back to Matrix
	err = d.matrix.Send(ctx, channel.Response{
		RoomID:  msg.RoomID,
		Content: response,
	})
	if err != nil {
		slog.Error("failed to send response", "error", err)
		return fmt.Errorf("send response: %w", err)
	}
	// Capture interaction in brain (async, don't block response)
	go func() {
		summary := msg.Content
		if len(summary) > 100 {
			summary = summary[:100] + "..."
		}
		d.brain.Capture("observation", "nous-daemon",
			fmt.Sprintf("Chat [%s] via %s: %s",
				msg.SenderID, route, summary),
			"chat,"+msg.Source, "nous-daemon",
		)
	}()
	return nil
}

// processMessage is the shared message processing pipeline.
// Used by both Matrix channel and workspace API.
// Returns (response, route, error).
func (d *Daemon) processMessage(ctx context.Context, msg channel.Message) (string, string, error) {
	start := time.Now()
	d.events.Publish(coredaemon.Event{Type: coredaemon.EventStatus, Message: fmt.Sprintf("processing %s message (%d chars)", msg.Source, len(msg.Content))})
	slog.Info("processing message",
		"source", msg.Source,
		"sender", msg.SenderID,
		"len", len(msg.Content),
	)

	// Record user message in history BEFORE routing (all routes share history)
	d.appendHistory(msg.RoomID, llm.Message{Role: "user", Content: msg.Content})

	var response string
	var err error
	var route string

	// Route stickiness: if recent message in this room went to a specific backend,
	// continue using that backend for conversational continuity (2 min window).
	// Commands (/dispatch, /oc) always override stickiness.
	stickyRoute := d.getStickyRoute(msg.RoomID)

	if strings.HasPrefix(msg.Content, "/dispatch") {
		return d.handleDispatch(ctx, msg)
	} else if strings.HasPrefix(msg.Content, "/exec") {
		return d.handleExec(ctx, msg)
	} else if strings.HasPrefix(msg.Content, "/oc ") {
		msg.Content = strings.TrimPrefix(msg.Content, "/oc ")
		if d.opencode != nil {
			route = "opencode (explicit)"
			response, err = d.handleViaOpenCode(ctx, msg)
		} else {
			return "", "", fmt.Errorf("OpenCode not configured \u2014 cannot handle /oc request")
		}
	} else if stickyRoute != "" {
		// Continue on same backend as recent conversation
		route = stickyRoute + " (sticky)"
		if strings.Contains(stickyRoute, "opencode") && d.opencode != nil {
			response, err = d.handleViaOpenCode(ctx, msg)
		} else if d.hasDirectLLM {
			response, err = d.handleDirectLLM(ctx, msg)
		}
	} else if d.hasDirectLLM && isInfraRequest(msg.Content) {
		route = "direct-llm (infra)"
		response, err = d.handleDirectLLM(ctx, msg)
	} else if d.opencode != nil && isCodingRequest(msg.Content) {
		route = "opencode (coding)"
		response, err = d.handleViaOpenCode(ctx, msg)
	} else if d.hasDirectLLM {
		route = "direct-llm"
		response, err = d.handleDirectLLM(ctx, msg)
		if err != nil && d.opencode != nil {
			slog.Warn("direct LLM failed, falling back to OpenCode", "error", err)
			route = "opencode (fallback)"
			response, err = d.handleViaOpenCode(ctx, msg)
		}
	} else if d.opencode != nil {
		route = "opencode (only)"
		response, err = d.handleViaOpenCode(ctx, msg)
	} else {
		return "", "", fmt.Errorf("no LLM backend available")
	}
	d.events.Publish(coredaemon.Event{Type: coredaemon.EventStatus, Message: fmt.Sprintf("route=%s", route)})
	if err != nil {
		d.events.Publish(coredaemon.Event{Type: coredaemon.EventError, Message: err.Error()})
		return "", route, err
	}
	elapsed := time.Since(start)
	slog.Info("response ready",
		"route", route,
		"elapsed", elapsed.Round(time.Millisecond),
		"len", len(response),
	)

	// Record assistant response in history and save sticky route
	d.appendHistory(msg.RoomID, llm.Message{Role: "assistant", Content: response})
	d.setStickyRoute(msg.RoomID, route)
	d.events.Publish(coredaemon.Event{Type: coredaemon.EventChat, Role: "assistant", Content: response})
	return response, route, nil
}

// handleDirectLLM processes a message using direct LLM providers.
// When the selected provider supports tools, uses the tool-use loop to
// fetch live data (web, GitHub, brain) instead of hallucinating.
func (d *Daemon) handleDirectLLM(ctx context.Context, msg channel.Message) (string, error) {
	sys := fmt.Sprintf("%s\n\n%s", chatSystemPrompt, d.identity)

	// Recall relevant memories for non-trivial messages
	if len(msg.Content) >= 20 {
		if recalledCtx := d.recallForChat(ctx, msg.Content); recalledCtx != "" {
			sys += "\n\n" + recalledCtx
		}
	}
	// Select tier based on message complexity:
	//   Fast  — trivial, short messages (<50 chars, no complex signals)
	//   Mid   — default for most conversation (Copilot/GPT)
	//   Deep  — complex reasoning, long context, analysis
	tier := llm.TierMid
	if len(msg.Content) > 200 || isComplexQuery(msg.Content) {
		tier = llm.TierDeep
	} else if len(msg.Content) < 50 && !isComplexQuery(msg.Content) {
		tier = llm.TierFast
	}

	// Get tier-specific config for model parameters
	cfg := d.tierConfig(tier)
	// User message already recorded in processMessage — just build history
	charBudget := d.historyCharBudget(cfg)
	messages := d.getHistory(msg.RoomID, charBudget)

	// Use provider config for output limits and temperature
	maxOutput := cfg.MaxOutput
	if maxOutput <= 0 {
		maxOutput = 4096
	}
	temp := cfg.Temperature
	if temp <= 0 {
		temp = 0.7
	}
	req := llm.CompletionRequest{
		System:      sys,
		Messages:    messages,
		MaxTokens:   maxOutput,
		Temperature: temp,
	}

	// Use tool-enhanced path when the provider supports tools
	if d.router.HasToolProvider(tier) {
		toolCount := len(d.chatToolExecutors())
		slog.Info("using tool-enhanced LLM path",
			"tier", tier,
			"tools", toolCount,
		)
		response, err := d.runToolLoop(ctx, tier, req, msg.Content)
		if err != nil {
			slog.Error("tool-enhanced LLM failed, falling back to plain", "error", err)
			// Fall through to plain completion below
		} else {
			return response, nil
		}
	}

	// Plain completion (no tools) — fallback or non-tool provider
	resp, err := d.router.Complete(ctx, tier, req)
	if err != nil {
		slog.Error("LLM completion failed", "tier", tier, "error", err)
		if tier == llm.TierFast {
			slog.Info("falling back to deep tier")
			resp, err = d.router.Complete(ctx, llm.TierDeep, req)
		}
		if err != nil {
			return "", fmt.Errorf("LLM error: %w", err)
		}
	}
	slog.Info("direct LLM response",
		"tier", tier,
		"model", resp.Model,
		"input_tokens", resp.InputTokens,
		"output_tokens", resp.OutputTokens,
		"history_len", len(messages),
		"char_budget", charBudget,
	)
	return resp.Content, nil
}

// handleViaOpenCode processes a message by proxying through OpenCode serve API.
// OpenCode manages its own session state, so no local history needed.
func (d *Daemon) handleViaOpenCode(ctx context.Context, msg channel.Message) (string, error) {
	// Protect ocSessionID — lightweight lock, only held for ID read/write, not during LLM call
	d.ocMu.Lock()
	currentSession := d.ocSessionID
	d.ocMu.Unlock()
	// Ensure we have a valid OC session
	sessionID, err := d.opencode.EnsureSession(ctx, currentSession)
	if err != nil {
		return "", fmt.Errorf("OpenCode session: %w", err)
	}

	// If session changed, prime it with chat system prompt and persist
	if sessionID != currentSession {
		slog.Info("new OpenCode session, priming with system prompt", "id", sessionID)
		_, err := d.opencode.SendMessage(ctx, sessionID, chatSystemPrompt)
		if err != nil {
			slog.Warn("failed to prime session", "error", err)
		}
		d.ocMu.Lock()
		d.ocSessionID = sessionID
		d.ocMu.Unlock()
		d.brain.KVSet("nous-daemon:oc-session-id", sessionID)
	}

	// Send the actual message
	response, err := d.opencode.SendMessage(ctx, sessionID, msg.Content)
	if err != nil {
		// Session might be stale — try creating a new one
		slog.Warn("message send failed, creating new session", "error", err)
		d.ocMu.Lock()
		d.ocSessionID = ""
		d.ocMu.Unlock()
		d.brain.KVSet("nous-daemon:oc-session-id", "")
		newSessionID, err := d.opencode.EnsureSession(ctx, "")
		if err != nil {
			return "", fmt.Errorf("new OpenCode session: %w", err)
		}
		d.ocMu.Lock()
		d.ocSessionID = newSessionID
		d.ocMu.Unlock()
		d.brain.KVSet("nous-daemon:oc-session-id", newSessionID)
		response, err = d.opencode.SendMessage(ctx, newSessionID, msg.Content)
		if err != nil {
			return "", fmt.Errorf("OpenCode retry failed: %w", err)
		}
	}

	slog.Info("OpenCode response", "session", sessionID, "len", len(response))
	return response, nil
}

// handleDispatch processes /dispatch commands.
// Routing:
//
//	/dispatch (or /dispatch list) → list available projects
//	/dispatch <project> <message> → dispatch task to project
func (d *Daemon) handleDispatch(ctx context.Context, msg channel.Message) (string, string, error) {
	if d.dispatch == nil {
		return "Dispatch not configured — no projects with URLs in config.", "dispatch", nil
	}

	content := strings.TrimPrefix(msg.Content, "/dispatch")
	content = strings.TrimSpace(content)

	// /dispatch or /dispatch list → show available projects
	if content == "" || content == "list" {
		projects := d.dispatch.ListProjects()
		if len(projects) == 0 {
			return "No projects configured for dispatch.", "dispatch", nil
		}
		var sb strings.Builder
		sb.WriteString("Available projects:\n")
		for _, p := range projects {
			sb.WriteString(fmt.Sprintf("• %s (%s)\n", p, d.dispatch.ProjectURL(p)))
		}
		sb.WriteString("\nUsage: /dispatch <project> <message>")
		return sb.String(), "dispatch", nil
	}

	// /dispatch <project> <message>
	parts := strings.SplitN(content, " ", 2)
	if len(parts) < 2 {
		return "Usage: /dispatch <project> <message>", "dispatch", nil
	}

	project, task := parts[0], parts[1]

	if !d.dispatch.HasProject(project) {
		return fmt.Sprintf("Unknown project: %s\nAvailable: %s", project, strings.Join(d.dispatch.ListProjects(), ", ")), "dispatch", nil
	}

	// Send intermediate "dispatching..." message
	if msg.Source == "matrix" {
		d.matrix.Send(ctx, channel.Response{
			RoomID:  msg.RoomID,
			Content: fmt.Sprintf("⏳ Dispatching to %s...", project),
		})
	}
	d.events.Publish(coredaemon.Event{Type: coredaemon.EventStatus, Message: fmt.Sprintf("dispatching to %s", project)})

	// Execute dispatch
	result, err := d.dispatch.Send(ctx, project, task)
	if err != nil {
		errMsg := fmt.Sprintf("❌ Dispatch to %s failed: %v", project, err)
		return errMsg, fmt.Sprintf("dispatch:%s", project), nil
	}

	response := fmt.Sprintf("📋 [%s] (%.1fs)\n\n%s", project, result.Duration.Seconds(), result.Text)
	return response, fmt.Sprintf("dispatch:%s", project), nil
}

// handleExec processes /exec commands for direct host command execution.
// Routing:
//
//	/exec (or /exec help) → usage
//	/exec <project> <command...> → execute via nous-proxy
func (d *Daemon) handleExec(ctx context.Context, msg channel.Message) (string, string, error) {
	content := strings.TrimPrefix(msg.Content, "/exec")
	content = strings.TrimSpace(content)

	if content == "" || content == "help" {
		return "Usage: /exec <project> <command...>\nExample: /exec sona-mp docker compose up -d --build api", "exec", nil
	}

	parts := strings.SplitN(content, " ", 2)
	if len(parts) < 2 {
		return "Usage: /exec <project> <command...>", "exec", nil
	}

	project := parts[0]
	cmdStr := parts[1]
	command := strings.Fields(cmdStr)

	// Send intermediate message
	if msg.Source == "matrix" {
		d.matrix.Send(ctx, channel.Response{
			RoomID:  msg.RoomID,
			Content: fmt.Sprintf("⚙️ Running on %s: %s", project, cmdStr),
		})
	}
	d.events.Publish(coredaemon.Event{Type: coredaemon.EventStatus, Message: fmt.Sprintf("exec %s: %s", project, cmdStr)})

	// Use a generous timeout for host commands (docker builds etc)
	execCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	result, err := d.proxyExec(execCtx, project, command, "")
	if err != nil {
		return fmt.Sprintf("❌ Failed: %v", err), "exec", nil
	}

	return result, fmt.Sprintf("exec:%s", project), nil
}

// --- Conversation History ---

// appendHistory adds a message to a room's conversation history.
func (d *Daemon) appendHistory(roomID string, msg llm.Message) {
	d.historyMu.Lock()
	defer d.historyMu.Unlock()

	d.history[roomID] = append(d.history[roomID], msg)

	// Trim to max entries
	if len(d.history[roomID]) > maxHistoryPerRoom {
		d.history[roomID] = d.history[roomID][len(d.history[roomID])-maxHistoryPerRoom:]
	}
}

// getHistory returns the conversation history for a room, trimmed to fit
// within the given character budget. Returns a copy safe for concurrent use.
func (d *Daemon) getHistory(roomID string, charBudget int) []llm.Message {
	d.historyMu.Lock()
	msgs := make([]llm.Message, len(d.history[roomID]))
	copy(msgs, d.history[roomID])
	d.historyMu.Unlock()
	totalChars := 0
	for _, m := range msgs {
		totalChars += len(m.Content)
	}

	for totalChars > charBudget && len(msgs) > 1 {
		totalChars -= len(msgs[0].Content)
		msgs = msgs[1:]
	}
	// Ensure history starts with a user message (LLM APIs require this)
	for len(msgs) > 0 && msgs[0].Role != "user" {
		msgs = msgs[1:]
	}
	return msgs
}

// --- Route Stickiness ---

const routeStickyDuration = 2 * time.Minute

// getStickyRoute returns the last route used for a room if it's within the sticky window.
// Returns empty string if no sticky route or if it has expired.
func (d *Daemon) getStickyRoute(roomID string) string {
	d.historyMu.Lock()
	defer d.historyMu.Unlock()

	if t, ok := d.lastRouteAt[roomID]; ok && time.Since(t) < routeStickyDuration {
		return d.lastRoute[roomID]
	}
	return ""
}

// setStickyRoute records the route used for a room.
func (d *Daemon) setStickyRoute(roomID string, route string) {
	// Strip sticky suffix to store the base route
	base := strings.TrimSuffix(route, " (sticky)")
	d.historyMu.Lock()
	defer d.historyMu.Unlock()
	d.lastRoute[roomID] = base
	d.lastRouteAt[roomID] = time.Now()
}

// tierConfig returns the ProviderConfig for a given tier.
func (d *Daemon) tierConfig(tier llm.Tier) ProviderConfig {
	switch tier {
	case llm.TierDeep:
		return d.config.LLM.Deep
	case llm.TierMid:
		return d.config.LLM.Mid
	default:
		return d.config.LLM.Fast
	}
}

// historyCharBudget computes the character budget for conversation history
// based on the provider's context window. Uses 60% of context for history,
// leaving room for system prompt, output tokens, and safety margin.
func (d *Daemon) historyCharBudget(cfg ProviderConfig) int {
	if cfg.ContextWindow > 0 {
		return int(float64(cfg.ContextWindow) * historyBudgetRatio * charsPerToken)
	}
	return defaultHistoryBudgetChars
}

// --- Helpers ---
func isComplexQuery(s string) bool {
	s = strings.ToLower(s)
	complexSignals := []string{
		"explain", "analyze", "compare", "design", "architect",
		"why does", "how should", "what if", "trade-off", "tradeoff",
		"strategy", "approach", "recommend", "evaluate", "review",
	}
	for _, signal := range complexSignals {
		if strings.Contains(s, signal) {
			return true
		}
	}
	return false
}

// isInfraRequest detects messages about infrastructure/ops that should use
// the direct LLM with proxy_exec tool instead of being routed to OpenCode.
func isInfraRequest(s string) bool {
	s = strings.ToLower(s)
	infraSignals := []string{
		"rebuild", "restart", "docker", "container",
		"compose up", "compose down", "compose build",
		"systemctl", "proxy", "ollama",
		"smp-", "sona-mp", "omo-config",
	}
	for _, signal := range infraSignals {
		if strings.Contains(s, signal) {
			return true
		}
	}
	return false
}

// isCodingRequest detects messages that should be routed to OpenCode.
// These are tasks requiring file access, code changes, or project operations.
func isCodingRequest(s string) bool {
	s = strings.ToLower(s)
	codingSignals := []string{
		// Explicit coding actions
		"fix the", "fix bug", "implement", "add feature", "create file",
		"refactor", "update the code", "write code", "write a function",
		"debug", "compile", "build", "deploy",
		// Project/file references
		"in the repo", "in the project", "in the codebase",
		".ts ", ".go ", ".py ", ".js ",
		"src/", "internal/", "package.json",
		// PR/git operations
		"pull request", "create pr", "merge", "commit",
		"git ", "branch",
		// Run/test operations
		"run tests", "run the", "execute", "npm ", "go test",
	}
	for _, signal := range codingSignals {
		if strings.Contains(s, signal) {
			return true
		}
	}
	return false
}

// recallForChat queries the brain for memories relevant to the user's message.
// Uses hybrid search (semantic + keyword) if available, keyword-only otherwise.
// Returns a formatted context block or empty string if no relevant memories found.
// Enforces a 500ms timeout to avoid slowing down chat.
func (d *Daemon) recallForChat(ctx context.Context, query string) string {
	const recallLimit = 5
	const recallTimeout = 500 * time.Millisecond
	const maxContextChars = 2000

	recallCtx, cancel := context.WithTimeout(ctx, recallTimeout)
	defer cancel()

	var memories []brain.Memory
	var err error

	// Try hybrid search first (semantic + keyword)
	d.embedMu.RLock()
	store := d.embedStore
	tei := d.teiClient
	d.embedMu.RUnlock()

	if store != nil && tei != nil {
		memories, err = embeddings.HybridSearch(recallCtx, query, d.brain, store, tei, recallLimit)
		if err != nil {
			slog.Debug("chat recall hybrid search failed", "error", err)
			err = nil
		}
	}

	// Keyword fallback
	if memories == nil {
		memories, err = d.brain.Recall(query, brain.RecallOptions{Limit: recallLimit})
		if err != nil {
			slog.Debug("chat recall keyword search failed", "error", err)
			return ""
		}
	}

	if len(memories) == 0 {
		return ""
	}

	// Format as concise context block
	var sb strings.Builder
	sb.WriteString("## Relevant Context (from persistent memory)\n")
	budget := maxContextChars
	for _, m := range memories {
		line := fmt.Sprintf("- [%s, %s] %s\n", m.Type, m.CreatedAt.Format("2006-01-02"), m.Content)
		if len(line) > 200 {
			line = line[:197] + "...\n"
		}
		if budget-len(line) < 0 {
			break
		}
		sb.WriteString(line)
		budget -= len(line)
	}

	slog.Debug("chat recall injected", "count", len(memories), "chars", sb.Len())
	return sb.String()
}
