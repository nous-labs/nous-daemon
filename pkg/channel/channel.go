// Package channel defines the interface for communication channels.
// Channels are how Nous talks to the world — Matrix, future Telegram, CLI, etc.
package channel

import "context"

// Message represents an incoming message from any channel.
type Message struct {
	// Source identifies the channel (e.g., "matrix", "telegram", "cli")
	Source string

	// SenderID is the channel-specific sender identifier
	SenderID string

	// RoomID is the channel-specific room/conversation identifier
	RoomID string

	// Content is the message text
	Content string

	// IsVoice indicates this was transcribed from audio
	IsVoice bool

	// Timestamp is the message timestamp in milliseconds
	Timestamp int64
}

// Response represents an outgoing message to a channel.
type Response struct {
	// Content is the text to send
	Content string

	// RoomID is the target room/conversation
	RoomID string
}

// Channel is the interface for a communication channel.
type Channel interface {
	// Name returns the channel identifier (e.g., "matrix").
	Name() string

	// Start begins listening for messages. Blocks until ctx is cancelled.
	// Received messages are sent to the handler function.
	Start(ctx context.Context, handler MessageHandler) error

	// Send sends a response to a specific room on this channel.
	Send(ctx context.Context, resp Response) error

	// Stop gracefully shuts down the channel.
	Stop() error
}

// MessageHandler is called when a message is received from any channel.
type MessageHandler func(ctx context.Context, msg Message) error
