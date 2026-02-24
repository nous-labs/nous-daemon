package embeddings

import (
	"context"
	"crypto/md5"
	"fmt"
	"log/slog"
	"time"

	"github.com/nous-labs/nous/pkg/brain"
)

// SyncWorker keeps pgvector embeddings in sync with SQLite memories.
// It polls for un-embedded or stale memories and processes them in batches.
type SyncWorker struct {
	brain     *brain.Brain
	store     *Store
	tei       *TEIClient
	interval  time.Duration
	batchSize int
}

// NewSyncWorker creates a new background sync worker.
func NewSyncWorker(b *brain.Brain, store *Store, tei *TEIClient, interval time.Duration, batchSize int) *SyncWorker {
	if interval <= 0 {
		interval = 30 * time.Second
	}
	if batchSize <= 0 {
		batchSize = 32
	}
	return &SyncWorker{
		brain:     b,
		store:     store,
		tei:       tei,
		interval:  interval,
		batchSize: batchSize,
	}
}

// Run starts the sync loop. Blocks until ctx is cancelled.
func (w *SyncWorker) Run(ctx context.Context) {
	slog.Info("embedding sync worker started",
		"interval", w.interval,
		"batch_size", w.batchSize,
	)

	// Initial sync on startup (backfill)
	if embedded, err := w.SyncOnce(ctx); err != nil {
		slog.Warn("initial embedding sync failed", "error", err)
	} else if embedded > 0 {
		slog.Info("initial embedding sync complete", "embedded", embedded)
	}

	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Info("embedding sync worker stopping")
			return
		case <-ticker.C:
			if embedded, err := w.SyncOnce(ctx); err != nil {
				slog.Warn("embedding sync cycle failed", "error", err)
			} else if embedded > 0 {
				slog.Info("embedding sync cycle", "embedded", embedded)
			}
		}
	}
}

// SyncOnce runs a single sync cycle:
//  1. Get all active memory IDs + content from SQLite
//  2. Get all embedded IDs + content hashes from pgvector
//  3. Find un-embedded or stale (hash mismatch) memories
//  4. Batch embed via TEI
//  5. Store in pgvector
func (w *SyncWorker) SyncOnce(ctx context.Context) (int, error) {
	// Step 1: Get all memory references from SQLite
	refs, err := w.brain.GetAllMemoryRefs()
	if err != nil {
		return 0, fmt.Errorf("get memory refs: %w", err)
	}

	// Step 2: Get all embedded references from pgvector
	embedded, err := w.store.GetEmbedded(ctx)
	if err != nil {
		return 0, fmt.Errorf("get embedded: %w", err)
	}

	// Step 3: Find memories that need embedding (new or stale)
	var toEmbed []brain.MemoryRef
	for _, ref := range refs {
		existingHash, exists := embedded[int64(ref.ID)]
		if !exists || existingHash != ref.ContentHash {
			toEmbed = append(toEmbed, ref)
		}
	}

	if len(toEmbed) == 0 {
		return 0, nil
	}

	slog.Info("memories need embedding",
		"total", len(refs),
		"already_embedded", len(embedded),
		"to_embed", len(toEmbed),
	)

	// Step 4 & 5: Process in batches
	totalEmbedded := 0
	for i := 0; i < len(toEmbed); i += w.batchSize {
		end := i + w.batchSize
		if end > len(toEmbed) {
			end = len(toEmbed)
		}
		batch := toEmbed[i:end]

		// Fetch full content for this batch
		ids := make([]int64, len(batch))
		for j, ref := range batch {
			ids[j] = int64(ref.ID)
		}
		memories, err := w.brain.GetMemoriesByIDs(ids)
		if err != nil {
			slog.Warn("fetch batch memories failed", "error", err, "batch_start", i)
			continue
		}

		// Build texts for embedding
		texts := make([]string, len(memories))
		memIDs := make([]int64, len(memories))
		hashes := make([]string, len(memories))
		for j, m := range memories {
			texts[j] = m.Content
			memIDs[j] = int64(m.ID)
			hashes[j] = ContentHash(m.Content)
		}

		// Embed batch via TEI
		embeddings, err := w.tei.EmbedDocuments(ctx, texts)
		if err != nil {
			slog.Warn("embed batch failed", "error", err, "batch_start", i, "batch_size", len(texts))
			continue
		}

		// Store in pgvector
		if err := w.store.InsertBatch(ctx, memIDs, embeddings, hashes); err != nil {
			slog.Warn("store batch failed", "error", err, "batch_start", i)
			continue
		}

		totalEmbedded += len(embeddings)
		slog.Debug("batch embedded",
			"batch", i/w.batchSize+1,
			"count", len(embeddings),
			"total_so_far", totalEmbedded,
		)
	}

	return totalEmbedded, nil
}

// ContentHash computes an MD5 hash of content for staleness detection.
func ContentHash(content string) string {
	return fmt.Sprintf("%x", md5.Sum([]byte(content)))
}
