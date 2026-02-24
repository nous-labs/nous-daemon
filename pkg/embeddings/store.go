package embeddings

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

// Store provides pgvector-backed embedding storage and search.
type Store struct {
	pool *pgxpool.Pool
}

// SearchResult holds a vector similarity search result.
type SearchResult struct {
	MemoryID int64
	Distance float64 // cosine distance (lower = more similar)
}

// NewStore creates a new pgvector store and verifies the connection.
func NewStore(ctx context.Context, pgURL string) (*Store, error) {
	config, err := pgxpool.ParseConfig(pgURL)
	if err != nil {
		return nil, fmt.Errorf("parse postgres URL: %w", err)
	}

	// Register pgvector types on each new connection
	config.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		return pgxvec.RegisterTypes(ctx, conn)
	}

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("create postgres pool: %w", err)
	}

	// Verify connection
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}

	return &Store{pool: pool}, nil
}

// Init creates the pgvector extension, table, and indexes if they don't exist.
func (s *Store) Init(ctx context.Context) error {
	// Enable pgvector extension
	_, err := s.pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	if err != nil {
		return fmt.Errorf("create vector extension: %w", err)
	}

	// Create embeddings table
	_, err = s.pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS memory_embeddings (
			memory_id    BIGINT PRIMARY KEY,
			embedding    vector(768) NOT NULL,
			content_hash TEXT NOT NULL,
			model_name   TEXT NOT NULL DEFAULT 'nomic-embed-text-v1.5',
			embedded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
		)
	`)
	if err != nil {
		return fmt.Errorf("create embeddings table: %w", err)
	}

	// Create HNSW index for cosine similarity search
	_, err = s.pool.Exec(ctx, `
		CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
		ON memory_embeddings
		USING hnsw (embedding vector_cosine_ops)
		WITH (m = 16, ef_construction = 64)
	`)
	if err != nil {
		return fmt.Errorf("create HNSW index: %w", err)
	}

	slog.Info("embedding store initialized")
	return nil
}

// Close closes the database connection pool.
func (s *Store) Close() {
	s.pool.Close()
}

// Insert stores or updates an embedding for a memory.
func (s *Store) Insert(ctx context.Context, memoryID int64, embedding []float32, contentHash string) error {
	vec := pgvector.NewVector(embedding)
	_, err := s.pool.Exec(ctx, `
		INSERT INTO memory_embeddings (memory_id, embedding, content_hash, embedded_at)
		VALUES ($1, $2, $3, now())
		ON CONFLICT (memory_id) DO UPDATE
		SET embedding = EXCLUDED.embedding,
			content_hash = EXCLUDED.content_hash,
			embedded_at = now()
	`, memoryID, vec, contentHash)
	if err != nil {
		return fmt.Errorf("insert embedding %d: %w", memoryID, err)
	}
	return nil
}

// InsertBatch stores embeddings for multiple memories in a single transaction.
func (s *Store) InsertBatch(ctx context.Context, memoryIDs []int64, embeddings [][]float32, contentHashes []string) error {
	if len(memoryIDs) != len(embeddings) || len(memoryIDs) != len(contentHashes) {
		return fmt.Errorf("mismatched batch sizes: ids=%d embeddings=%d hashes=%d",
			len(memoryIDs), len(embeddings), len(contentHashes))
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin batch tx: %w", err)
	}
	defer tx.Rollback(ctx)

	for i := range memoryIDs {
		vec := pgvector.NewVector(embeddings[i])
		_, err := tx.Exec(ctx, `
			INSERT INTO memory_embeddings (memory_id, embedding, content_hash, embedded_at)
			VALUES ($1, $2, $3, now())
			ON CONFLICT (memory_id) DO UPDATE
			SET embedding = EXCLUDED.embedding,
				content_hash = EXCLUDED.content_hash,
				embedded_at = now()
		`, memoryIDs[i], vec, contentHashes[i])
		if err != nil {
			return fmt.Errorf("insert embedding %d: %w", memoryIDs[i], err)
		}
	}

	return tx.Commit(ctx)
}

// Search returns the top-K most similar memories by cosine distance.
func (s *Store) Search(ctx context.Context, queryEmbedding []float32, limit int) ([]SearchResult, error) {
	vec := pgvector.NewVector(queryEmbedding)
	rows, err := s.pool.Query(ctx, `
		SELECT memory_id, embedding <=> $1 AS distance
		FROM memory_embeddings
		ORDER BY embedding <=> $1
		LIMIT $2
	`, vec, limit)
	if err != nil {
		return nil, fmt.Errorf("vector search: %w", err)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var r SearchResult
		if err := rows.Scan(&r.MemoryID, &r.Distance); err != nil {
			return nil, fmt.Errorf("scan search result: %w", err)
		}
		results = append(results, r)
	}
	return results, rows.Err()
}

// GetEmbedded returns all embedded memory IDs with their content hashes.
func (s *Store) GetEmbedded(ctx context.Context) (map[int64]string, error) {
	rows, err := s.pool.Query(ctx, "SELECT memory_id, content_hash FROM memory_embeddings")
	if err != nil {
		return nil, fmt.Errorf("get embedded: %w", err)
	}
	defer rows.Close()

	embedded := make(map[int64]string)
	for rows.Next() {
		var id int64
		var hash string
		if err := rows.Scan(&id, &hash); err != nil {
			return nil, fmt.Errorf("scan embedded: %w", err)
		}
		embedded[id] = hash
	}
	return embedded, rows.Err()
}

// Delete removes an embedding for a memory.
func (s *Store) Delete(ctx context.Context, memoryID int64) error {
	_, err := s.pool.Exec(ctx, "DELETE FROM memory_embeddings WHERE memory_id = $1", memoryID)
	return err
}

// Stats returns embedding count.
func (s *Store) Stats(ctx context.Context) (count int, err error) {
	err = s.pool.QueryRow(ctx, "SELECT COUNT(*) FROM memory_embeddings").Scan(&count)
	return
}
