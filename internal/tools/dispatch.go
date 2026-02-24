// Package tools — DispatchClient manages dispatch to multiple OpenCode project instances.
//
// Dispatch enables routing tasks from Matrix (or workspace API) to host-side
// OpenCode instances running on different ports for different projects.
// Each project gets its own persistent session (stored via brain KV).
package tools

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"
)

// DispatchTarget describes a project's OpenCode instance.
type DispatchTarget struct {
	Name string // project name (e.g., "my-project")
	URL  string // OpenCode serve API URL (e.g., "http://host.docker.internal:4097")
}

// DispatchResult holds the result of a dispatched task.
type DispatchResult struct {
	Text      string
	SessionID string
	Project   string
	Duration  time.Duration
}

// KVStore provides key-value persistence for session IDs.
type KVStore struct {
	Get func(key string) (string, error)
	Set func(key, value string) error
}

// DispatchClient manages dispatch to multiple OpenCode project instances.
type DispatchClient struct {
	targets  map[string]*DispatchTarget
	clients  map[string]*OpenCodeClient
	sessions map[string]string // project → session ID
	username string
	password string
	kv       KVStore
	mu       sync.Mutex
}

// DispatchConfig holds configuration for creating a DispatchClient.
type DispatchConfig struct {
	Targets  []DispatchTarget
	Username string
	Password string
	KV       KVStore
}

// NewDispatch creates a new DispatchClient from the given configuration.
// Restores persisted session IDs from brain KV on startup.
func NewDispatch(cfg DispatchConfig) *DispatchClient {
	dc := &DispatchClient{
		targets:  make(map[string]*DispatchTarget),
		clients:  make(map[string]*OpenCodeClient),
		sessions: make(map[string]string),
		username: cfg.Username,
		password: cfg.Password,
		kv:       cfg.KV,
	}

	for _, t := range cfg.Targets {
		t := t
		dc.targets[t.Name] = &t
	}

	// Restore persisted session IDs
	for name := range dc.targets {
		key := kvKey(name)
		if sid, err := dc.kv.Get(key); err == nil && sid != "" {
			dc.sessions[name] = sid
			slog.Info("restored dispatch session", "project", name, "session", sid)
		}
	}

	slog.Info("dispatch client initialized", "projects", dc.ListProjects())
	return dc
}

// ListProjects returns names of all dispatchable projects, sorted alphabetically.
func (dc *DispatchClient) ListProjects() []string {
	names := make([]string, 0, len(dc.targets))
	for name := range dc.targets {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// HasProject checks if a project is registered for dispatch.
func (dc *DispatchClient) HasProject(name string) bool {
	_, ok := dc.targets[name]
	return ok
}

// ProjectURL returns the OpenCode API URL for a project (for diagnostics).
func (dc *DispatchClient) ProjectURL(name string) string {
	if t, ok := dc.targets[name]; ok {
		return t.URL
	}
	return ""
}

// Send dispatches a message to a project's OpenCode instance.
// Creates or reuses a persistent session per project.
// On session failure, retries once with a fresh session.
func (dc *DispatchClient) Send(ctx context.Context, project, message string) (*DispatchResult, error) {
	dc.mu.Lock()
	target, ok := dc.targets[project]
	if !ok {
		dc.mu.Unlock()
		return nil, fmt.Errorf("unknown project: %s (known: %s)", project, strings.Join(dc.ListProjects(), ", "))
	}

	// Get or create client for this project
	client, ok := dc.clients[project]
	if !ok {
		client = NewOpenCode(target.URL, dc.username, dc.password)
		dc.clients[project] = client
	}
	sessionID := dc.sessions[project]
	dc.mu.Unlock()

	start := time.Now()

	// Ensure session exists
	newSessionID, err := client.EnsureSession(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("dispatch session for %s: %w", project, err)
	}

	// Persist session ID if changed
	if newSessionID != sessionID {
		dc.mu.Lock()
		dc.sessions[project] = newSessionID
		dc.mu.Unlock()
		dc.kv.Set(kvKey(project), newSessionID)
		slog.Info("new dispatch session", "project", project, "session", newSessionID)
	}

	// Send the message
	response, err := client.SendMessage(ctx, newSessionID, message)
	if err != nil {
		// Session might be stale — create new and retry once
		slog.Warn("dispatch send failed, retrying with new session",
			"project", project, "error", err)

		dc.mu.Lock()
		dc.sessions[project] = ""
		dc.mu.Unlock()
		dc.kv.Set(kvKey(project), "")

		newSessionID, err = client.EnsureSession(ctx, "")
		if err != nil {
			return nil, fmt.Errorf("dispatch new session for %s: %w", project, err)
		}

		dc.mu.Lock()
		dc.sessions[project] = newSessionID
		dc.mu.Unlock()
		dc.kv.Set(kvKey(project), newSessionID)

		response, err = client.SendMessage(ctx, newSessionID, message)
		if err != nil {
			return nil, fmt.Errorf("dispatch retry for %s: %w", project, err)
		}
	}

	elapsed := time.Since(start)
	slog.Info("dispatch complete",
		"project", project,
		"session", newSessionID,
		"elapsed", elapsed.Round(time.Millisecond),
		"response_len", len(response),
	)

	return &DispatchResult{
		Text:      response,
		SessionID: newSessionID,
		Project:   project,
		Duration:  elapsed,
	}, nil
}

// IsAvailable checks if a specific project's OpenCode instance is reachable.
func (dc *DispatchClient) IsAvailable(ctx context.Context, project string) bool {
	dc.mu.Lock()
	client, ok := dc.clients[project]
	if !ok {
		target, ok := dc.targets[project]
		if !ok {
			dc.mu.Unlock()
			return false
		}
		client = NewOpenCode(target.URL, dc.username, dc.password)
		dc.clients[project] = client
	}
	dc.mu.Unlock()
	return client.IsAvailable(ctx)
}

// kvKey returns the brain KV key for a project's dispatch session.
func kvKey(project string) string {
	return "nous-daemon:dispatch-session:" + project
}
