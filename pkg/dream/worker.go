// Package dream implements Nous's autonomous memory maintenance daemon.
//
// The dream worker runs as a background goroutine, periodically scanning
// the memory system for maintenance opportunities:
//   - Stale memory detection (flag memories past their TTL/half-life)
//   - Orphaned episode cleanup (close abandoned active episodes)
//   - Access score decay (recalculate frequency scores)
//   - Memory health reporting (stats + findings via events)
//
// Design principles:
//   - Read-mostly: never deletes, only flags or bi-temporally invalidates
//   - Safe ops autonomous: stale detection, orphan cleanup, decay
//   - Destructive ops gated: invalidation only for extreme staleness (≥0.95)
//   - Observable: all actions published to event bus for TUI visibility
package dream

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/nous-labs/nous/pkg/brain"
)

// EventFunc is a callback for publishing dream events.
// Parameters: event type, message.
type EventFunc func(typ, message string)

// Report holds the results of a single dream cycle.
type Report struct {
	CycleNumber int            `json:"cycle_number"`
	StartedAt   time.Time      `json:"started_at"`
	Duration    string         `json:"duration"`
	MemoryStats map[string]int `json:"memory_stats"`

	// Stale memories
	StaleCount    int          `json:"stale_count"`
	StaleWarning  []StaleEntry `json:"stale_warning,omitempty"`  // score ≥ 0.7
	StaleCritical []StaleEntry `json:"stale_critical,omitempty"` // score ≥ 0.9
	Invalidated   int          `json:"invalidated"`              // auto-invalidated (≥ 0.95)

	// Episodes
	OrphansClosed int `json:"orphans_closed"`

	// Access scores
	ScoresDecayed int `json:"scores_decayed"`

	// Errors (non-fatal)
	Errors []string `json:"errors,omitempty"`
}

// StaleEntry is a single stale memory in the report.
type StaleEntry struct {
	ID      int     `json:"id"`
	Type    string  `json:"type"`
	Score   float64 `json:"score"`
	Days    float64 `json:"days_since_verified"`
	Content string  `json:"content"` // truncated
}

// Worker is the dream daemon background worker.
type Worker struct {
	brain    *brain.Brain
	onEvent  EventFunc
	interval time.Duration

	// Thresholds
	staleWarningThreshold   float64       // flag as [aging]
	staleCriticalThreshold  float64       // flag as [STALE]
	autoInvalidateThreshold float64       // auto-invalidate (bi-temporal)
	orphanStaleAfter        time.Duration // close episodes older than this

	// State
	mu         sync.RWMutex
	lastReport *Report
	cycleCount int
}

// Config holds dream worker configuration.
type Config struct {
	Interval                time.Duration // how often to dream (default 6h)
	StaleWarningThreshold   float64       // flag as aging (default 0.4)
	StaleCriticalThreshold  float64       // flag as STALE (default 0.7)
	AutoInvalidateThreshold float64       // auto-invalidate (default 0.95)
	OrphanStaleAfter        time.Duration // close episodes older than (default 48h)
}

// DefaultConfig returns sensible defaults for the dream worker.
func DefaultConfig() Config {
	return Config{
		Interval:                6 * time.Hour,
		StaleWarningThreshold:   0.4,
		StaleCriticalThreshold:  0.7,
		AutoInvalidateThreshold: 0.95,
		OrphanStaleAfter:        48 * time.Hour,
	}
}

// NewWorker creates a new dream worker.
func NewWorker(b *brain.Brain, onEvent EventFunc, cfg Config) *Worker {
	if cfg.Interval <= 0 {
		cfg.Interval = 6 * time.Hour
	}
	if cfg.StaleWarningThreshold <= 0 {
		cfg.StaleWarningThreshold = 0.4
	}
	if cfg.StaleCriticalThreshold <= 0 {
		cfg.StaleCriticalThreshold = 0.7
	}
	if cfg.AutoInvalidateThreshold <= 0 {
		cfg.AutoInvalidateThreshold = 0.95
	}
	if cfg.OrphanStaleAfter <= 0 {
		cfg.OrphanStaleAfter = 48 * time.Hour
	}

	return &Worker{
		brain:                   b,
		onEvent:                 onEvent,
		interval:                cfg.Interval,
		staleWarningThreshold:   cfg.StaleWarningThreshold,
		staleCriticalThreshold:  cfg.StaleCriticalThreshold,
		autoInvalidateThreshold: cfg.AutoInvalidateThreshold,
		orphanStaleAfter:        cfg.OrphanStaleAfter,
	}
}

// Run starts the dream loop. Blocks until ctx is cancelled.
func (w *Worker) Run(ctx context.Context) {
	slog.Info("dream worker started",
		"interval", w.interval,
		"stale_warn", w.staleWarningThreshold,
		"stale_crit", w.staleCriticalThreshold,
		"auto_invalidate", w.autoInvalidateThreshold,
		"orphan_stale_after", w.orphanStaleAfter,
	)
	w.emit("status", "Dream worker started")

	// Initial dream on startup (after a short delay to let other systems initialize)
	select {
	case <-ctx.Done():
		return
	case <-time.After(30 * time.Second):
	}

	if report := w.DreamOnce(ctx); report != nil {
		w.logReport(report)
	}

	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Info("dream worker stopping")
			w.emit("status", "Dream worker stopped")
			return
		case <-ticker.C:
			if report := w.DreamOnce(ctx); report != nil {
				w.logReport(report)
			}
		}
	}
}

// DreamOnce runs a single dream cycle. Returns the report.
func (w *Worker) DreamOnce(ctx context.Context) *Report {
	w.mu.Lock()
	w.cycleCount++
	cycle := w.cycleCount
	w.mu.Unlock()

	start := time.Now()
	w.emit("status", fmt.Sprintf("Dream cycle %d starting", cycle))

	report := &Report{
		CycleNumber: cycle,
		StartedAt:   start,
	}

	// 1. Memory stats
	report.MemoryStats = w.brain.MemoryStats()

	// 2. Stale memory detection
	w.detectStale(ctx, report)

	// 3. Orphaned episode cleanup
	w.cleanOrphanedEpisodes(ctx, report)

	// 4. Access score decay
	w.decayScores(ctx, report)

	report.Duration = time.Since(start).Round(time.Millisecond).String()

	// Store report
	w.mu.Lock()
	w.lastReport = report
	w.mu.Unlock()

	return report
}

// LastReport returns the most recent dream report.
func (w *Worker) LastReport() *Report {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.lastReport
}

// detectStale finds stale memories and handles them by severity.
func (w *Worker) detectStale(ctx context.Context, report *Report) {
	stale, err := w.brain.FindStaleMemories(w.staleWarningThreshold)
	if err != nil {
		report.Errors = append(report.Errors, fmt.Sprintf("stale scan: %v", err))
		slog.Warn("dream: stale scan failed", "error", err)
		return
	}

	report.StaleCount = len(stale)

	for _, sm := range stale {
		entry := StaleEntry{
			ID:      sm.ID,
			Type:    sm.Type,
			Score:   sm.StalenessScore,
			Days:    sm.DaysSinceVerified,
			Content: truncate(sm.Content, 80),
		}

		if sm.StalenessScore >= w.autoInvalidateThreshold {
			// Auto-invalidate extremely stale memories (bi-temporal, not delete)
			if err := w.brain.InvalidateMemory(sm.ID, fmt.Sprintf(
				"auto-invalidated by dream: staleness=%.2f, %dd since verified",
				sm.StalenessScore, int(sm.DaysSinceVerified),
			)); err != nil {
				report.Errors = append(report.Errors, fmt.Sprintf("invalidate [%d]: %v", sm.ID, err))
			} else {
				report.Invalidated++
				w.emit("status", fmt.Sprintf("Dream: invalidated memory [%d] (staleness=%.2f)", sm.ID, sm.StalenessScore))
			}
		} else if sm.StalenessScore >= w.staleCriticalThreshold {
			report.StaleCritical = append(report.StaleCritical, entry)
		} else {
			report.StaleWarning = append(report.StaleWarning, entry)
		}
	}

	if report.StaleCount > 0 {
		slog.Info("dream: stale scan complete",
			"total", report.StaleCount,
			"warning", len(report.StaleWarning),
			"critical", len(report.StaleCritical),
			"invalidated", report.Invalidated,
		)
	}
}

// cleanOrphanedEpisodes closes episodes that have been active too long.
func (w *Worker) cleanOrphanedEpisodes(ctx context.Context, report *Report) {
	orphans, err := w.brain.FindOrphanedEpisodes(w.orphanStaleAfter)
	if err != nil {
		report.Errors = append(report.Errors, fmt.Sprintf("orphan scan: %v", err))
		slog.Warn("dream: orphan scan failed", "error", err)
		return
	}

	for _, ep := range orphans {
		summary := fmt.Sprintf("Auto-closed by dream daemon: episode was active for %s with no completion",
			time.Since(ep.StartedAt).Round(time.Hour))
		if err := w.brain.CloseEpisode(ep.ID, summary); err != nil {
			report.Errors = append(report.Errors, fmt.Sprintf("close episode %s: %v", ep.ID, err))
		} else {
			report.OrphansClosed++
			w.emit("status", fmt.Sprintf("Dream: closed orphaned episode %s (scope=%s, age=%s)",
				ep.ID, ep.Scope, time.Since(ep.StartedAt).Round(time.Hour)))
		}
	}
}

// decayScores recalculates access frequency scores with exponential decay.
func (w *Worker) decayScores(ctx context.Context, report *Report) {
	decayed, err := w.brain.DecayAllAccessScores()
	if err != nil {
		report.Errors = append(report.Errors, fmt.Sprintf("decay scores: %v", err))
		slog.Warn("dream: decay scores failed", "error", err)
		return
	}
	report.ScoresDecayed = decayed
}

// logReport logs the dream report summary and publishes events.
func (w *Worker) logReport(report *Report) {
	totalMemories := 0
	for _, count := range report.MemoryStats {
		totalMemories += count
	}

	summary := fmt.Sprintf(
		"Dream cycle %d complete (%s): %d memories, %d stale (%d critical, %d invalidated), %d orphans closed, %d scores decayed",
		report.CycleNumber,
		report.Duration,
		totalMemories,
		report.StaleCount,
		len(report.StaleCritical),
		report.Invalidated,
		report.OrphansClosed,
		report.ScoresDecayed,
	)

	if len(report.Errors) > 0 {
		summary += fmt.Sprintf(", %d errors", len(report.Errors))
	}

	slog.Info("dream: cycle complete", "summary", summary)
	w.emit("status", summary)

	// Capture dream report as observation in brain
	w.brain.Capture("observation", "nous-daemon", summary, "dream,maintenance", "dream-worker")
}

// emit publishes an event if the callback is set.
func (w *Worker) emit(typ, message string) {
	if w.onEvent != nil {
		w.onEvent(typ, message)
	}
}

// truncate shortens a string to maxLen, adding "..." if truncated.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
