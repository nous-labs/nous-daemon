// Package matrix implements the Matrix channel for Nous.
// This replaces the Python nous_bridge.py with a native Go implementation
// using mautrix-go, running inside the daemon process directly.
package matrix

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"maunium.net/go/mautrix"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"

	"github.com/nous-labs/nous/pkg/channel"
)

// Config holds Matrix channel configuration.
type Config struct {
	Homeserver   string
	UserID       string // e.g., "nous"
	Password     string
	ServerName   string // e.g., "matrix.example.com"
	AllowedUsers []string
	DataDir      string
}

// Channel implements the channel.Channel interface for Matrix.
type Channel struct {
	config    Config
	client    *mautrix.Client
	handler   channel.MessageHandler
	startTime int64
	mu        sync.Mutex

	// Persistent state
	credFile  string
	stateFile string
}

// credentials holds saved Matrix login state.
type credentials struct {
	AccessToken string `json:"access_token"`
	UserID      string `json:"user_id"`
	DeviceID    string `json:"device_id"`
}

// state holds persistent channel state.
type state struct {
	RoomID string `json:"room_id"`
}

// New creates a new Matrix channel.
func New(cfg Config) *Channel {
	return &Channel{
		config:    cfg,
		credFile:  filepath.Join(cfg.DataDir, "matrix_credentials.json"),
		stateFile: filepath.Join(cfg.DataDir, "matrix_state.json"),
	}
}

// Name returns the channel identifier.
func (c *Channel) Name() string { return "matrix" }

// Start connects to Matrix and begins listening for messages.
// Retries login with exponential backoff on failure.
func (c *Channel) Start(ctx context.Context, handler channel.MessageHandler) error {
	c.handler = handler
	c.startTime = time.Now().UnixMilli()

	// Ensure data directory exists
	os.MkdirAll(c.config.DataDir, 0o755)

	// Build full user ID
	fullUserID := fmt.Sprintf("@%s:%s", c.config.UserID, c.config.ServerName)

	// Create client
	client, err := mautrix.NewClient(c.config.Homeserver, id.UserID(fullUserID), "")
	if err != nil {
		return fmt.Errorf("create matrix client: %w", err)
	}
	c.client = client

	// Use in-memory sync store (will resync on restart, which is fine)
	client.Store = mautrix.NewMemorySyncStore()

	// Login with retry/backoff
	if err := c.loginWithRetry(ctx, fullUserID); err != nil {
		return err
	}

	// Register event handlers
	syncer := client.Syncer.(*mautrix.DefaultSyncer)

	// Handle text messages
	syncer.OnEventType(event.EventMessage, func(ctx context.Context, evt *event.Event) {
		c.onMessage(ctx, evt)
	})

	// Handle invites — auto-join from allowed users
	syncer.OnEventType(event.StateMember, func(ctx context.Context, evt *event.Event) {
		c.onMemberEvent(ctx, evt)
	})

	slog.Info("matrix channel ready, starting sync")

	// Sync loop with reconnection
	for {
		err := client.SyncWithContext(ctx)
		if ctx.Err() != nil {
			return nil // graceful shutdown
		}
		if err != nil {
			slog.Warn("matrix sync error, reconnecting in 15s", "error", err)
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(15 * time.Second):
			}
		}
	}
}

// loginWithRetry handles Matrix login with exponential backoff.
// Tries saved credentials first, then password login with retry.
func (c *Channel) loginWithRetry(ctx context.Context, fullUserID string) error {
	// Try loading saved credentials first (no retry needed)
	if err := c.loadCredentials(); err == nil {
		slog.Info("loaded saved Matrix credentials", "user", fullUserID)
		return nil
	}

	// Password login with exponential backoff
	backoff := 2 * time.Second
	maxBackoff := 2 * time.Minute
	maxAttempts := 10

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		slog.Info("logging into Matrix",
			"user", fullUserID,
			"homeserver", c.config.Homeserver,
			"attempt", attempt,
		)

		resp, err := c.client.Login(ctx, &mautrix.ReqLogin{
			Type: mautrix.AuthTypePassword,
			Identifier: mautrix.UserIdentifier{
				Type: mautrix.IdentifierTypeUser,
				User: c.config.UserID,
			},
			Password:         c.config.Password,
			StoreCredentials: true,
		})

		if err == nil {
			slog.Info("logged into Matrix", "user", resp.UserID, "device", resp.DeviceID)
			c.saveCredentials(credentials{
				AccessToken: resp.AccessToken,
				UserID:      string(resp.UserID),
				DeviceID:    string(resp.DeviceID),
			})
			return nil
		}

		// Check if this is a non-retryable error
		errStr := err.Error()
		if strings.Contains(errStr, "M_FORBIDDEN") ||
			strings.Contains(errStr, "M_UNKNOWN_TOKEN") ||
			strings.Contains(errStr, "M_INVALID_PARAM") {
			return fmt.Errorf("matrix login: %w (non-retryable)", err)
		}

		if attempt == maxAttempts {
			return fmt.Errorf("matrix login: %w (after %d attempts)", err, maxAttempts)
		}

		slog.Warn("matrix login failed, retrying",
			"error", err,
			"attempt", attempt,
			"backoff", backoff,
		)

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}

		// Exponential backoff with cap
		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}

	return fmt.Errorf("matrix login: exhausted retries")
}

// Send sends a message to a Matrix room, splitting long messages.
func (c *Channel) Send(ctx context.Context, resp channel.Response) error {
	const maxLen = 4000

	content := resp.Content
	roomID := id.RoomID(resp.RoomID)

	if len(content) <= maxLen {
		_, err := c.client.SendText(ctx, roomID, content)
		if err != nil {
			slog.Error("matrix send failed", "room", roomID, "len", len(content), "error", err)
		} else {
			slog.Info("matrix message sent", "room", roomID, "len", len(content))
		}
		return err
	}

	// Split long messages
	chunks := splitMessage(content, maxLen)
	for i, chunk := range chunks {
		prefix := ""
		if len(chunks) > 1 {
			prefix = fmt.Sprintf("[%d/%d] ", i+1, len(chunks))
		}
		_, err := c.client.SendText(ctx, roomID, prefix+chunk)
		if err != nil {
			slog.Error("matrix send failed", "room", roomID, "chunk", i+1, "error", err)
			return err
		}
		if i < len(chunks)-1 {
			time.Sleep(500 * time.Millisecond)
		}
	}
	slog.Info("matrix message sent", "room", roomID, "chunks", len(chunks), "total_len", len(content))
	return nil
}

// Stop gracefully shuts down the Matrix channel.
func (c *Channel) Stop() error {
	if c.client != nil {
		c.client.StopSync()
	}
	return nil
}

// --- Event Handlers ---

func (c *Channel) onMessage(ctx context.Context, evt *event.Event) {
	// Skip own messages
	if evt.Sender == c.client.UserID {
		return
	}

	// Skip messages from before we started
	if evt.Timestamp < c.startTime {
		return
	}

	// Check allowed users
	if !c.isAllowed(evt.Sender) {
		return
	}

	msgContent := evt.Content.AsMessage()
	if msgContent == nil || msgContent.Body == "" {
		return
	}

	slog.Info("matrix message received",
		"sender", evt.Sender,
		"room", evt.RoomID,
		"content", truncate(msgContent.Body, 100),
	)

	// Build channel message
	msg := channel.Message{
		Source:    "matrix",
		SenderID:  string(evt.Sender),
		RoomID:    string(evt.RoomID),
		Content:   msgContent.Body,
		Timestamp: evt.Timestamp,
	}

	// Dispatch to handler
	if err := c.handler(ctx, msg); err != nil {
		slog.Error("message handler error", "error", err)
		// Send error back to user
		c.Send(ctx, channel.Response{
			RoomID:  string(evt.RoomID),
			Content: fmt.Sprintf("*(Error: %s)*", err),
		})
	}
}

func (c *Channel) onMemberEvent(ctx context.Context, evt *event.Event) {
	// Only handle invites for us
	if evt.GetStateKey() != string(c.client.UserID) {
		return
	}

	memberContent := evt.Content.AsMember()
	if memberContent == nil || memberContent.Membership != event.MembershipInvite {
		return
	}

	// Check if inviter is allowed
	if !c.isAllowed(evt.Sender) {
		slog.Warn("rejecting invite from unauthorized user", "sender", evt.Sender)
		return
	}

	slog.Info("accepting room invite", "room", evt.RoomID, "from", evt.Sender)
	_, err := c.client.JoinRoomByID(ctx, evt.RoomID)
	if err != nil {
		slog.Error("failed to join room", "room", evt.RoomID, "error", err)
	}
}

// --- Credentials ---

func (c *Channel) loadCredentials() error {
	data, err := os.ReadFile(c.credFile)
	if err != nil {
		return err
	}
	var creds credentials
	if err := json.Unmarshal(data, &creds); err != nil {
		return err
	}
	c.client.AccessToken = creds.AccessToken
	c.client.UserID = id.UserID(creds.UserID)
	c.client.DeviceID = id.DeviceID(creds.DeviceID)
	return nil
}

func (c *Channel) saveCredentials(creds credentials) {
	data, _ := json.MarshalIndent(creds, "", "  ")
	os.WriteFile(c.credFile, data, 0o600)
}

// --- Helpers ---

func (c *Channel) isAllowed(sender id.UserID) bool {
	if len(c.config.AllowedUsers) == 0 || c.config.AllowedUsers[0] == "" {
		return true // no restriction
	}
	for _, allowed := range c.config.AllowedUsers {
		if string(sender) == allowed {
			return true
		}
	}
	return false
}

func splitMessage(s string, maxLen int) []string {
	var chunks []string
	for len(s) > maxLen {
		chunks = append(chunks, s[:maxLen])
		s = s[maxLen:]
	}
	if len(s) > 0 {
		chunks = append(chunks, s)
	}
	return chunks
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}
