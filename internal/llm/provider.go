// Package llm provides LLM provider interfaces and implementations
// for Nous's own reasoning capabilities.
package llm

import "context"

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`    // system, user, assistant
	Content string `json:"content"`
}

// CompletionRequest holds parameters for an LLM completion.
type CompletionRequest struct {
	Messages    []Message `json:"messages"`
	Model       string    `json:"model,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	System      string    `json:"system,omitempty"` // Anthropic-style system prompt
}

// CompletionResponse holds the LLM's response.
type CompletionResponse struct {
	Content      string     `json:"content"`
	Model        string     `json:"model"`
	InputTokens  int        `json:"input_tokens"`
	OutputTokens int        `json:"output_tokens"`
	StopReason   string     `json:"stop_reason"`
	ToolCalls    []ToolCall `json:"tool_calls,omitempty"`
}

// Provider is the interface for LLM providers.
type Provider interface {
	// Name returns the provider identifier (e.g., "anthropic", "kimi").
	Name() string

	// Complete sends a completion request and returns the response.
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)
}

type ToolProvider interface {
	Provider
	CompleteWithTools(ctx context.Context, req CompletionRequest, tools []ToolDefinition, toolMessages []ToolMessage) (*CompletionResponse, error)
}

// Tier represents the quality/cost tier for model selection.
type Tier int

const (
	TierFast Tier = iota // Cheap, fast — for simple tasks (Kimi K2.5)
	TierMid              // Mid-tier — good balance of quality and cost (Copilot/GPT)
	TierDeep             // Expensive, thorough — for complex reasoning (Claude)
)

// Router selects the appropriate provider based on task tier.
type Router struct {
	providers map[Tier]Provider
}

// NewRouter creates a provider router with the given tier mappings.
func NewRouter(providers map[Tier]Provider) *Router {
	return &Router{providers: providers}
}

// Complete routes a request to the appropriate provider based on tier.
// Fallback chain: requested tier → deep → mid → fast.
func (r *Router) Complete(ctx context.Context, tier Tier, req CompletionRequest) (*CompletionResponse, error) {
	p, ok := r.providers[tier]
	if !ok {
		// Fallback chain: try higher tiers first, then lower
		for _, fallback := range []Tier{TierDeep, TierMid, TierFast} {
			if fallback == tier {
				continue
			}
			p, ok = r.providers[fallback]
			if ok {
				break
			}
		}
		if !ok {
			return nil, ErrNoProvider
		}
	}
	return p.Complete(ctx, req)
}

// CompleteWithTools routes a request with tools to the appropriate provider.
// If the provider supports tools (implements ToolProvider), uses CompleteWithTools;
// otherwise falls back to regular Complete (ignoring tools).
func (r *Router) CompleteWithTools(ctx context.Context, tier Tier, req CompletionRequest, tools []ToolDefinition, toolMessages []ToolMessage) (*CompletionResponse, error) {
	p := r.resolveProvider(tier)
	if p == nil {
		return nil, ErrNoProvider
	}
	if tp, ok := p.(ToolProvider); ok {
		return tp.CompleteWithTools(ctx, req, tools, toolMessages)
	}
	return p.Complete(ctx, req)
}

// resolveProvider finds the best provider for the given tier using the fallback chain.
func (r *Router) resolveProvider(tier Tier) Provider {
	if p, ok := r.providers[tier]; ok {
		return p
	}
	for _, fallback := range []Tier{TierDeep, TierMid, TierFast} {
		if fallback == tier {
			continue
		}
		if p, ok := r.providers[fallback]; ok {
			return p
		}
	}
	return nil
}

// HasToolProvider returns true if any provider in the given tier supports tools.
func (r *Router) HasToolProvider(tier Tier) bool {
	p := r.resolveProvider(tier)
	if p == nil {
		return false
	}
	_, ok := p.(ToolProvider)
	return ok
}

// ErrNoProvider is returned when no provider is configured for the requested tier.
var ErrNoProvider = &ProviderError{Message: "no provider configured for requested tier"}

// ProviderError represents an LLM provider error.
type ProviderError struct {
	Message    string
	StatusCode int
	Provider   string
}

func (e *ProviderError) Error() string {
	if e.Provider != "" {
		return e.Provider + ": " + e.Message
	}
	return e.Message
}
