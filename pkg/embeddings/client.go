// Package embeddings provides semantic memory via vector embeddings.
//
// It connects to HuggingFace Text Embeddings Inference (TEI) for generating
// embeddings and pgvector (PostgreSQL) for storing and searching them.
// The package keeps embeddings in sync with the SQLite-based brain memories
// via a background worker, and provides hybrid search (vector + FTS5 + RRF).
package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	// PrefixDocument is the task prefix for document embeddings (storage).
	// Required by nomic-embed-text for optimal performance.
	PrefixDocument = "search_document: "
	// PrefixQuery is the task prefix for query embeddings (search).
	PrefixQuery = "search_query: "
)

// TEIClient is an HTTP client for HuggingFace Text Embeddings Inference.
type TEIClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewTEIClient creates a new TEI client.
func NewTEIClient(baseURL string) *TEIClient {
	return &TEIClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// embedRequest is the TEI /embed request body.
type embedRequest struct {
	Inputs interface{} `json:"inputs"` // string or []string
}

// Embed generates embeddings for one or more texts.
// taskPrefix should be PrefixDocument or PrefixQuery.
func (c *TEIClient) Embed(ctx context.Context, texts []string, taskPrefix string) ([][]float32, error) {
	// Prepend task prefix to all texts
	prefixed := make([]string, len(texts))
	for i, t := range texts {
		prefixed[i] = taskPrefix + t
	}

	var body embedRequest
	if len(prefixed) == 1 {
		body = embedRequest{Inputs: prefixed[0]}
	} else {
		body = embedRequest{Inputs: prefixed}
	}

	reqBytes, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/embed", bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("create embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("TEI returned %d: %s", resp.StatusCode, string(respBody))
	}

	var embeddings [][]float32
	if err := json.NewDecoder(resp.Body).Decode(&embeddings); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}

	return embeddings, nil
}

// EmbedDocument generates an embedding for document storage.
func (c *TEIClient) EmbedDocument(ctx context.Context, text string) ([]float32, error) {
	results, err := c.Embed(ctx, []string{text}, PrefixDocument)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}
	return results[0], nil
}

// EmbedQuery generates an embedding for search queries.
func (c *TEIClient) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	results, err := c.Embed(ctx, []string{text}, PrefixQuery)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}
	return results[0], nil
}

// EmbedDocuments generates embeddings for multiple documents in a single batch.
func (c *TEIClient) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	return c.Embed(ctx, texts, PrefixDocument)
}

// Health checks if the TEI service is available.
func (c *TEIClient) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("TEI health check: %w", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("TEI unhealthy: status %d", resp.StatusCode)
	}
	return nil
}
