package daemon

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/nous-labs/nous/pkg/brain"
	"github.com/nous-labs/nous/pkg/dream"
	"github.com/nous-labs/nous/pkg/embeddings"
)

type Daemon struct {
	Brain   *brain.Brain
	Config  *Config
	Events  *EventBus
	Modules map[string]Module

	startedAt  time.Time
	healthyMu  sync.RWMutex
	healthy    bool
	httpServer *http.Server

	embedMu    sync.RWMutex
	embedStore *embeddings.Store
	teiClient  *embeddings.TEIClient
	dreamer    *dream.Worker
}

func New(b *brain.Brain, cfg *Config) (*Daemon, error) {
	if b == nil {
		return nil, fmt.Errorf("brain is required")
	}
	if cfg == nil {
		cfg = defaultConfig()
	}
	if cfg.Modules == nil {
		cfg.Modules = map[string]json.RawMessage{}
	}
	if cfg.HTTPAddr == "" {
		cfg.HTTPAddr = ":8080"
	}

	return &Daemon{
		Brain:     b,
		Config:    cfg,
		Events:    NewEventBus(),
		Modules:   map[string]Module{},
		startedAt: time.Now(),
	}, nil
}

func (d *Daemon) RegisterModule(m Module) error {
	if m == nil {
		return fmt.Errorf("module is nil")
	}
	name := m.Name()
	if name == "" {
		return fmt.Errorf("module name is empty")
	}
	if _, exists := d.Modules[name]; exists {
		return fmt.Errorf("module already registered: %s", name)
	}
	d.Modules[name] = m
	return nil
}

func (d *Daemon) setHealthy(v bool) {
	d.healthyMu.Lock()
	d.healthy = v
	d.healthyMu.Unlock()
}

func (d *Daemon) isHealthy() bool {
	d.healthyMu.RLock()
	v := d.healthy
	d.healthyMu.RUnlock()
	return v
}

func (d *Daemon) Run(ctx context.Context) error {
	if err := d.initModules(); err != nil {
		return err
	}
	d.initSemanticMemory()
	d.startEmbeddingSyncWorker(ctx)
	d.startDreamWorker(ctx)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", d.handleHealth)
	mux.HandleFunc("/v1/recall", d.handleRecall)
	mux.HandleFunc("/v1/events", d.handleEvents)
	for _, m := range d.Modules {
		m.RegisterRoutes(mux)
	}

	d.httpServer = &http.Server{Addr: d.Config.HTTPAddr, Handler: mux}
	errCh := make(chan error, 1)
	go func() {
		err := d.httpServer.ListenAndServe()
		if err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	for _, m := range d.Modules {
		mod := m
		go func() {
			if err := mod.Start(ctx); err != nil && ctx.Err() == nil {
				slog.Error("module start failed", "module", mod.Name(), "error", err)
			}
		}()
	}

	d.setHealthy(true)

	select {
	case <-ctx.Done():
	case err := <-errCh:
		d.setHealthy(false)
		return err
	}

	d.setHealthy(false)
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if d.httpServer != nil {
		_ = d.httpServer.Shutdown(shutdownCtx)
	}

	for _, m := range d.Modules {
		if err := m.Stop(); err != nil {
			slog.Warn("module stop failed", "module", m.Name(), "error", err)
		}
	}

	d.embedMu.Lock()
	if d.embedStore != nil {
		d.embedStore.Close()
		d.embedStore = nil
		d.teiClient = nil
	}
	d.embedMu.Unlock()

	return nil
}

func (d *Daemon) initModules() error {
	for _, m := range d.Modules {
		if err := m.Init(d); err != nil {
			return fmt.Errorf("init module %s: %w", m.Name(), err)
		}
	}
	return nil
}

func (d *Daemon) initSemanticMemory() {
	if !d.Config.Embeddings.Enabled || d.Config.Embeddings.PostgresURL == "" || d.Config.Embeddings.TEIURL == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	store, err := embeddings.NewStore(ctx, d.Config.Embeddings.PostgresURL)
	if err != nil {
		slog.Warn("semantic memory unavailable", "error", err)
		return
	}
	if err := store.Init(ctx); err != nil {
		slog.Warn("semantic memory init failed", "error", err)
		store.Close()
		return
	}

	d.embedMu.Lock()
	d.embedStore = store
	d.teiClient = embeddings.NewTEIClient(d.Config.Embeddings.TEIURL)
	d.embedMu.Unlock()
}

func (d *Daemon) startEmbeddingSyncWorker(ctx context.Context) {
	d.embedMu.RLock()
	store := d.embedStore
	tei := d.teiClient
	d.embedMu.RUnlock()
	if store == nil || tei == nil {
		return
	}

	interval := 30 * time.Second
	if d.Config.Embeddings.SyncInterval != "" {
		if parsed, err := time.ParseDuration(d.Config.Embeddings.SyncInterval); err == nil {
			interval = parsed
		}
	}
	batchSize := d.Config.Embeddings.BatchSize
	if batchSize <= 0 {
		batchSize = 50
	}
	worker := embeddings.NewSyncWorker(d.Brain, store, tei, interval, batchSize)
	go worker.Run(ctx)
}

func (d *Daemon) startDreamWorker(ctx context.Context) {
	if d.Config.Dream.Disabled {
		return
	}
	cfg := dream.DefaultConfig()
	if d.Config.Dream.Interval != "" {
		if parsed, err := time.ParseDuration(d.Config.Dream.Interval); err == nil {
			cfg.Interval = parsed
		}
	}
	d.dreamer = dream.NewWorker(d.Brain, func(typ, msg string) {
		d.Events.Publish(Event{Type: EventStatus, Message: "[dream] " + msg})
	}, cfg)
	go d.dreamer.Run(ctx)
}

func (d *Daemon) handleHealth(w http.ResponseWriter, _ *http.Request) {
	if d.isHealthy() {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"ok","uptime":"%s"}`, time.Since(d.startedAt).Round(time.Second))
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
	fmt.Fprint(w, `{"status":"starting"}`)
}

func (d *Daemon) handleEvents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	events, done := d.Events.Subscribe()
	defer d.Events.Unsubscribe(done)

	for _, e := range d.Events.Recent(50) {
		fmt.Fprintf(w, "data: %s\n\n", e.MarshalEvent())
	}
	flusher.Flush()

	for {
		select {
		case <-r.Context().Done():
			return
		case evt, ok := <-events:
			if !ok {
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", evt.MarshalEvent())
			flusher.Flush()
		}
	}
}

type recallResponse struct {
	Memories []recallMemory `json:"memories"`
	Method   string         `json:"method"`
	Query    string         `json:"query"`
	Count    int            `json:"count"`
}

type recallMemory struct {
	ID        int    `json:"id"`
	Type      string `json:"type"`
	Scope     string `json:"scope"`
	Content   string `json:"content"`
	Tags      string `json:"tags,omitempty"`
	Source    string `json:"source,omitempty"`
	CreatedAt string `json:"created_at"`
}

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

	d.embedMu.RLock()
	store := d.embedStore
	tei := d.teiClient
	d.embedMu.RUnlock()

	if store != nil && tei != nil {
		method = "hybrid"
		memories, err = embeddings.HybridSearch(r.Context(), query, d.Brain, store, tei, limit)
		if err != nil {
			slog.Warn("hybrid search failed", "error", err)
			method = "keyword"
			memories = nil
		}
	}

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
		memories, err = d.Brain.Recall(query, opts)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintf(w, `{"error":%q}`, err.Error())
			return
		}
	}

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
