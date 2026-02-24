// Package tools provides external tool integrations for Nous.
// The primary tool is OpenCode — Nous's coding interface.
package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"
)

// OpenCodeClient manages communication with OpenCode serve API.
type OpenCodeClient struct {
	baseURL  string
	username string
	password string
	client   *http.Client
}

// NewOpenCode creates a new OpenCode API client.
func NewOpenCode(baseURL, username, password string) *OpenCodeClient {
	return &OpenCodeClient{
		baseURL:  baseURL,
		username: username,
		password: password,
		client: &http.Client{
			Timeout: 5 * time.Minute, // coding tasks can take a while
		},
	}
}

// Session represents an OpenCode session.
type Session struct {
	ID string `json:"id"`
}

// MessageResponse holds the response from sending a message.
type MessageResponse struct {
	Parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"parts"`
}

// CreateSession creates a new OpenCode session.
func (oc *OpenCodeClient) CreateSession(ctx context.Context) (*Session, error) {
	body := []byte(`{}`)
	resp, err := oc.doRequest(ctx, "POST", "/session", body)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	var session Session
	if err := json.Unmarshal(resp, &session); err != nil {
		return nil, fmt.Errorf("parse session: %w", err)
	}

	slog.Info("opencode session created", "id", session.ID)
	return &session, nil
}

// GetSession checks if a session exists and is valid.
func (oc *OpenCodeClient) GetSession(ctx context.Context, id string) (*Session, error) {
	resp, err := oc.doRequest(ctx, "GET", "/session/"+id, nil)
	if err != nil {
		return nil, err
	}

	var session Session
	if err := json.Unmarshal(resp, &session); err != nil {
		return nil, fmt.Errorf("parse session: %w", err)
	}
	return &session, nil
}

// SendMessage sends a message to an OpenCode session and returns the response text.
func (oc *OpenCodeClient) SendMessage(ctx context.Context, sessionID, message string) (string, error) {
	payload := map[string]interface{}{
		"parts": []map[string]string{
			{"type": "text", "text": message},
		},
	}
	body, _ := json.Marshal(payload)

	slog.Info("opencode message sent",
		"session", sessionID,
		"content", truncateStr(message, 100),
	)

	resp, err := oc.doRequest(ctx, "POST", "/session/"+sessionID+"/message", body)
	if err != nil {
		return "", fmt.Errorf("send message: %w", err)
	}

	var msgResp MessageResponse
	if err := json.Unmarshal(resp, &msgResp); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}

	// Extract text parts
	var parts []string
	for _, p := range msgResp.Parts {
		if p.Type == "text" && p.Text != "" {
			parts = append(parts, p.Text)
		}
	}

	if len(parts) == 0 {
		return "*(No response)*", nil
	}

	result := ""
	for i, p := range parts {
		if i > 0 {
			result += "\n"
		}
		result += p
	}
	return result, nil
}

// EnsureSession gets or creates a session, reusing an existing one if valid.
func (oc *OpenCodeClient) EnsureSession(ctx context.Context, existingID string) (string, error) {
	if existingID != "" {
		_, err := oc.GetSession(ctx, existingID)
		if err == nil {
			return existingID, nil
		}
		slog.Debug("existing session invalid, creating new", "old_id", existingID)
	}

	session, err := oc.CreateSession(ctx)
	if err != nil {
		return "", err
	}
	return session.ID, nil
}

// IsAvailable checks if OpenCode serve is reachable.
func (oc *OpenCodeClient) IsAvailable(ctx context.Context) bool {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	_, err := oc.doRequest(ctx, "GET", "/session", nil)
	return err == nil
}

// --- HTTP helpers ---

func (oc *OpenCodeClient) doRequest(ctx context.Context, method, path string, body []byte) ([]byte, error) {
	var bodyReader io.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	}

	req, err := http.NewRequestWithContext(ctx, method, oc.baseURL+path, bodyReader)
	if err != nil {
		return nil, err
	}

	req.SetBasicAuth(oc.username, oc.password)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := oc.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

func truncateStr(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}
