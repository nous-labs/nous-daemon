// Package daemon — event bus for broadcasting events to workspace TUI clients.
package daemon

import (
	"encoding/json"
	"sync"
	"time"
)

// Event types for the workspace event stream.
const (
	EventChat   = "chat"   // Chat message (user or assistant)
	EventPanel  = "panel"  // Side panel content update
	EventStatus = "status" // Status/routing info
	EventTask   = "task"   // Brain task update
	EventError  = "error"  // Error notification
)

// Event is a single event broadcast to workspace TUI clients.
type Event struct {
	Type    string `json:"type"`
	Content string `json:"content,omitempty"` // For chat/panel content
	Role    string `json:"role,omitempty"`    // For chat: "user" or "assistant"
	Panel   string `json:"panel,omitempty"`   // For panel: "activity", "tasks", "opencode"
	Message string `json:"message,omitempty"` // For status/error messages
	Level   string `json:"level,omitempty"`   // For status: "info", "warn", "error"
	TS      string `json:"ts"`
}

// MarshalEvent serializes an event to JSON with timestamp.
func (e Event) MarshalEvent() []byte {
	if e.TS == "" {
		e.TS = time.Now().Format(time.RFC3339)
	}
	b, _ := json.Marshal(e)
	return b
}

// subscriber is a connected TUI client receiving events via SSE.
type subscriber struct {
	ch   chan Event
	done chan struct{}
}

// EventBus fans out events to all connected workspace TUI clients.
// Thread-safe. Subscribers that fall behind are dropped.
type EventBus struct {
	mu          sync.RWMutex
	subscribers map[*subscriber]struct{}

	// Ring buffer for recent events (so new connections get context)
	recent    []Event
	recentMu  sync.RWMutex
	maxRecent int
}

// NewEventBus creates a new event bus.
func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[*subscriber]struct{}),
		maxRecent:   200, // Keep last 200 events for new subscribers
	}
}

// Publish sends an event to all connected subscribers.
// Non-blocking: slow subscribers are dropped.
func (eb *EventBus) Publish(e Event) {
	if e.TS == "" {
		e.TS = time.Now().Format(time.RFC3339)
	}

	// Store in ring buffer
	eb.recentMu.Lock()
	eb.recent = append(eb.recent, e)
	if len(eb.recent) > eb.maxRecent {
		eb.recent = eb.recent[len(eb.recent)-eb.maxRecent:]
	}
	eb.recentMu.Unlock()

	// Fan out to subscribers
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	for sub := range eb.subscribers {
		select {
		case sub.ch <- e:
			// delivered
		default:
			// subscriber too slow — drop event (they'll catch up via recent buffer)
		}
	}
}

// Subscribe creates a new subscriber. Returns a channel of events and a
// done channel to signal unsubscription. Caller MUST call Unsubscribe when done.
func (eb *EventBus) Subscribe() (<-chan Event, chan struct{}) {
	sub := &subscriber{
		ch:   make(chan Event, 64), // buffered to absorb bursts
		done: make(chan struct{}),
	}

	eb.mu.Lock()
	eb.subscribers[sub] = struct{}{}
	eb.mu.Unlock()

	return sub.ch, sub.done
}

// Unsubscribe removes a subscriber.
func (eb *EventBus) Unsubscribe(done chan struct{}) {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	for sub := range eb.subscribers {
		if sub.done == done {
			close(sub.ch)
			delete(eb.subscribers, sub)
			return
		}
	}
}

// Recent returns the last N events from the ring buffer.
// Used to hydrate new TUI connections with recent context.
func (eb *EventBus) Recent(n int) []Event {
	eb.recentMu.RLock()
	defer eb.recentMu.RUnlock()

	if n <= 0 || n > len(eb.recent) {
		n = len(eb.recent)
	}
	// Return a copy from the tail
	result := make([]Event, n)
	copy(result, eb.recent[len(eb.recent)-n:])
	return result
}

// SubscriberCount returns the number of connected subscribers.
func (eb *EventBus) SubscriberCount() int {
	eb.mu.RLock()
	defer eb.mu.RUnlock()
	return len(eb.subscribers)
}
