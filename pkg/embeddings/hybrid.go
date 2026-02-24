package embeddings

import (
	"context"
	"log/slog"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/nous-labs/nous/pkg/brain"
)

const (
	// rrfK is the smoothing constant for Reciprocal Rank Fusion.
	// Standard value from Cormack et al. (2009).
	rrfK = 60
	// overFetchMultiplier fetches more results from each source for better fusion.
	overFetchMultiplier = 3
)

// FusedResult holds a hybrid search result with combined RRF score.
type FusedResult struct {
	MemoryID int64
	Score    float64 // RRF score (higher = more relevant)
}

// HybridSearch combines vector similarity with FTS5 keyword search
// using Reciprocal Rank Fusion (RRF, k=60).
//
// Flow:
//  1. Embed query via TEI
//  2. Vector search in pgvector (parallel)
//  3. Keyword search in SQLite FTS5 (parallel)
//  4. Fuse results with RRF
//  5. Fetch full memories from SQLite, ordered by RRF score
//
// Degrades gracefully: if vector search fails, returns keyword-only results.
func HybridSearch(
	ctx context.Context,
	query string,
	b *brain.Brain,
	store *Store,
	tei *TEIClient,
	limit int,
) ([]brain.Memory, error) {
	// Step 1: Embed the query
	queryEmbedding, err := tei.EmbedQuery(ctx, query)
	if err != nil {
		slog.Warn("semantic embed failed, falling back to keyword-only", "error", err)
		return b.Recall(query, brain.RecallOptions{Limit: limit})
	}

	fetchLimit := limit * overFetchMultiplier

	// Step 2 & 3: Parallel vector + keyword search
	var vectorResults []SearchResult
	var keywordResults []brain.Memory
	var vectorErr, keywordErr error
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		vectorResults, vectorErr = store.Search(ctx, queryEmbedding, fetchLimit)
	}()

	go func() {
		defer wg.Done()
		keywordResults, keywordErr = b.Recall(query, brain.RecallOptions{Limit: fetchLimit})
	}()

	wg.Wait()

	// Handle errors gracefully — degrade to whichever source works
	if vectorErr != nil && keywordErr != nil {
		return nil, vectorErr
	}

	if vectorErr != nil {
		slog.Warn("vector search failed, using keyword-only", "error", vectorErr)
		if len(keywordResults) > limit {
			keywordResults = keywordResults[:limit]
		}
		return keywordResults, nil
	}

	if keywordErr != nil {
		slog.Warn("keyword search failed, using vector-only", "error", keywordErr)
		ids := make([]int64, len(vectorResults))
		for i, r := range vectorResults {
			ids[i] = r.MemoryID
		}
		memories, err := b.GetMemoriesByIDs(ids)
		if err != nil {
			return nil, err
		}
		if len(memories) > limit {
			memories = memories[:limit]
		}
		return memories, nil
	}

	// Step 4: Build ranked lists for RRF
	vectorRanked := make([]FusedResult, len(vectorResults))
	for i, r := range vectorResults {
		vectorRanked[i] = FusedResult{MemoryID: r.MemoryID}
	}

	keywordRanked := make([]FusedResult, len(keywordResults))
	for i, m := range keywordResults {
		keywordRanked[i] = FusedResult{MemoryID: int64(m.ID)}
	}

	fused := reciprocalRankFusion([][]FusedResult{vectorRanked, keywordRanked}, rrfK)

	// Step 5: Fetch full memories from SQLite
	ids := make([]int64, len(fused))
	statsIDs := make([]int, len(fused))
	for i, r := range fused {
		ids[i] = r.MemoryID
		statsIDs[i] = int(r.MemoryID)
	}

	memories, err := b.GetMemoriesByIDs(ids)
	if err != nil {
		return nil, err
	}

	accessStats, err := b.GetAccessStats(statsIDs)
	if err != nil {
		return nil, err
	}

	memByID := make(map[int64]brain.Memory, len(memories))
	for _, m := range memories {
		memByID[int64(m.ID)] = m
	}

	now := time.Now().UTC()
	compositeByID := make(map[int64]float64, len(fused))
	for _, r := range fused {
		m, ok := memByID[r.MemoryID]
		if !ok {
			continue
		}
		access := accessStats[int(r.MemoryID)]
		staleness := computeStaleness(m, now)
		composite := r.Score * (1 + 0.2*math.Log1p(access.AccessScore)) * (1 - 0.3*staleness)
		compositeByID[r.MemoryID] = composite
	}

	sort.Slice(fused, func(i, j int) bool {
		return compositeByID[fused[i].MemoryID] > compositeByID[fused[j].MemoryID]
	})

	if len(fused) > limit {
		fused = fused[:limit]
	}

	ordered := make([]brain.Memory, 0, len(fused))
	for _, result := range fused {
		if m, ok := memByID[result.MemoryID]; ok {
			ordered = append(ordered, m)
		}
	}

	return ordered, nil
}

func computeStaleness(m brain.Memory, now time.Time) float64 {
	policy := "ttl"
	if m.StalenessPolicy != nil && *m.StalenessPolicy != "" {
		policy = *m.StalenessPolicy
	}

	if policy == "none" {
		return 0.0
	}

	refTime := m.CreatedAt
	if m.VerifiedAt != nil && !m.VerifiedAt.IsZero() {
		refTime = *m.VerifiedAt
	}
	ageDays := now.Sub(refTime).Seconds() / 86400.0
	if ageDays < 0 {
		ageDays = 0
	}

	if policy == "ttl" {
		ttl := 60.0
		if m.TTLDays != nil && *m.TTLDays > 0 {
			ttl = float64(*m.TTLDays)
		}
		if ageDays <= ttl {
			return 0.0
		}
		return clamp01((ageDays - ttl) / 14.0)
	}

	if policy == "half_life" {
		return 1.0 - math.Exp(-math.Ln2*ageDays/30.0)
	}

	return 0.0
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// reciprocalRankFusion merges multiple ranked lists using RRF.
// Formula: RRF_score(d) = Σ 1/(k + rank_i(d))
func reciprocalRankFusion(lists [][]FusedResult, k int) []FusedResult {
	scores := make(map[int64]float64)

	for _, list := range lists {
		for rank, result := range list {
			// rank is 0-indexed, RRF uses 1-indexed
			scores[result.MemoryID] += 1.0 / (float64(k) + float64(rank+1))
		}
	}

	fused := make([]FusedResult, 0, len(scores))
	for id, score := range scores {
		fused = append(fused, FusedResult{MemoryID: id, Score: score})
	}

	sort.Slice(fused, func(i, j int) bool {
		return fused[i].Score > fused[j].Score
	})

	return fused
}
