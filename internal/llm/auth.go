// Package llm - auth.go provides OpenCode auth.json integration.
// This reads credentials from OpenCode's auth store, supporting both
// API keys and OAuth tokens (with automatic refresh for Anthropic MAX).
package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"time"
)

// AnthropicOAuthClientID is the OAuth client ID used by OpenCode's
// anthropic auth plugin for Claude Pro/Max.
const AnthropicOAuthClientID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

// AuthEntry represents a single provider's auth data from auth.json.
type AuthEntry struct {
	Type      string `json:"type"`                // "api" or "oauth"
	Key       string `json:"key,omitempty"`        // for type=api
	Access    string `json:"access,omitempty"`     // for type=oauth
	Refresh   string `json:"refresh,omitempty"`    // for type=oauth
	Expires   int64  `json:"expires,omitempty"`    // unix ms, for type=oauth
	AccountID string `json:"accountId,omitempty"` // optional
}

// AuthStore reads and manages credentials from OpenCode's auth.json.
type AuthStore struct {
	path    string
	entries map[string]*AuthEntry
	mu      sync.RWMutex
}

// NewAuthStore creates an auth store from the given auth.json path.
func NewAuthStore(path string) (*AuthStore, error) {
	s := &AuthStore{path: path}
	if err := s.load(); err != nil {
		return nil, err
	}
	return s, nil
}

// load reads auth.json from disk.
func (s *AuthStore) load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			s.entries = make(map[string]*AuthEntry)
			return nil
		}
		return fmt.Errorf("read auth.json: %w", err)
	}

	entries := make(map[string]*AuthEntry)
	if err := json.Unmarshal(data, &entries); err != nil {
		return fmt.Errorf("parse auth.json: %w", err)
	}
	s.entries = entries
	return nil
}

// save writes auth.json back to disk (for token refresh).
func (s *AuthStore) save() error {
	data, err := json.MarshalIndent(s.entries, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, data, 0o600)
}

// Get returns the auth entry for a provider.
func (s *AuthStore) Get(providerID string) *AuthEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.entries[providerID]
}

// GetAPIKey returns a usable API key for a provider.
// For type=api, returns the key directly.
// For type=oauth, returns the access token (refreshing if expired).
func (s *AuthStore) GetAPIKey(providerID string) (string, error) {
	entry := s.Get(providerID)
	if entry == nil {
		return "", fmt.Errorf("no auth for provider %q", providerID)
	}

	switch entry.Type {
	case "api":
		return entry.Key, nil
	case "oauth":
		return s.getOAuthToken(providerID, entry)
	default:
		return "", fmt.Errorf("unknown auth type %q for %q", entry.Type, providerID)
	}
}

// getOAuthToken returns a valid access token, refreshing if expired.
func (s *AuthStore) getOAuthToken(providerID string, entry *AuthEntry) (string, error) {
	// Check if token is still valid (with 5 min buffer)
	if entry.Access != "" && entry.Expires > time.Now().UnixMilli()+5*60*1000 {
		return entry.Access, nil
	}

	// Need to refresh
	slog.Info("refreshing OAuth token", "provider", providerID)

	switch providerID {
	case "anthropic":
		return s.refreshAnthropic(entry)
	default:
		// For unknown OAuth providers, return existing token and hope it works
		if entry.Access != "" {
			return entry.Access, nil
		}
		return "", fmt.Errorf("OAuth token expired for %q and no refresh implementation", providerID)
	}
}

// refreshAnthropic refreshes an Anthropic OAuth token.
func (s *AuthStore) refreshAnthropic(entry *AuthEntry) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Double-check after acquiring write lock
	if entry.Access != "" && entry.Expires > time.Now().UnixMilli()+5*60*1000 {
		return entry.Access, nil
	}

	body, _ := json.Marshal(map[string]string{
		"grant_type":    "refresh_token",
		"refresh_token": entry.Refresh,
		"client_id":     AnthropicOAuthClientID,
	})

	resp, err := http.Post(
		"https://console.anthropic.com/v1/oauth/token",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		return "", fmt.Errorf("anthropic token refresh: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("anthropic token refresh HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var tokenResp struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresIn    int    `json:"expires_in"`
	}
	if err := json.Unmarshal(respBody, &tokenResp); err != nil {
		return "", fmt.Errorf("parse token response: %w", err)
	}

	// Update entry
	entry.Access = tokenResp.AccessToken
	if tokenResp.RefreshToken != "" {
		entry.Refresh = tokenResp.RefreshToken
	}
	entry.Expires = time.Now().UnixMilli() + int64(tokenResp.ExpiresIn)*1000

	// Persist to disk so OpenCode also sees the refreshed token
	if err := s.save(); err != nil {
		slog.Warn("failed to save refreshed token", "error", err)
	}

	slog.Info("OAuth token refreshed",
		"provider", "anthropic",
		"expires_in", fmt.Sprintf("%dh", tokenResp.ExpiresIn/3600),
	)

	return entry.Access, nil
}

// ListProviders returns all provider IDs with auth entries.
func (s *AuthStore) ListProviders() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var ids []string
	for id := range s.entries {
		ids = append(ids, id)
	}
	return ids
}