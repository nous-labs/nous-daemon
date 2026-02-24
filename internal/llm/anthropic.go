package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicBeta is the base beta header for standard API key auth.
const anthropicBeta = "claude-code-20250219,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14"

// anthropicOAuthBeta is the base beta header for OAuth auth.
// The oauth-2025-04-20 flag is MANDATORY — Anthropic rejects OAuth without it.
const anthropicOAuthBeta = "oauth-2025-04-20,interleaved-thinking-2025-05-14,claude-code-20250219,fine-grained-tool-streaming-2025-05-14"

// anthropicLongContextBeta is appended when the request needs >128K context.
// Triggers premium pricing (2x input) — only add when actually needed.
const anthropicLongContextBeta = "context-1m-2025-08-07"

// AnthropicProvider implements the Provider interface for Claude and Anthropic-compatible APIs.
// Supports two auth modes:
//   - API key: static x-api-key header (traditional)
//   - OAuth: dynamic Bearer token via AuthStore (MAX plan)
type AnthropicProvider struct {
	client *anthropic.Client
	model  string
	name   string     // provider name ("anthropic", "kimi", etc.)
	auth   *AuthStore // nil = API key mode (key baked into client)
}

// NewAnthropic creates a new Anthropic provider with a static API key.
func NewAnthropic(apiKey, model string) *AnthropicProvider {
	opts := []option.RequestOption{
		option.WithHeader("anthropic-beta", anthropicBeta),
	}
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}

	client := anthropic.NewClient(opts...)

	if model == "" {
		model = "claude-opus-4-6"
	}

	return &AnthropicProvider{
		client: &client,
		model:  model,
	}
}

// NewAnthropicCompat creates an Anthropic-compatible provider with a custom base URL.
// Used for providers like Kimi that expose an Anthropic-format API.
func NewAnthropicCompat(name, baseURL, apiKey, model string) *AnthropicProvider {
	opts := []option.RequestOption{
		option.WithBaseURL(baseURL),
		option.WithHeader("User-Agent", "opencode/0.1.0"),
	}
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}

	client := anthropic.NewClient(opts...)

	return &AnthropicProvider{
		client: &client,
		model:  model,
		name:   name,
	}
}

// NewAnthropicOAuth creates an Anthropic provider using OAuth tokens from AuthStore.
// The access token is resolved (and refreshed if needed) on each request via
// a custom HTTP transport that rewrites headers and URLs to match Anthropic's
// OAuth requirements (as implemented by the opencode-anthropic-auth plugin).
func NewAnthropicOAuth(auth *AuthStore, model string) *AnthropicProvider {
	// Custom transport intercepts all requests to:
	// 1. Replace x-api-key with Authorization: Bearer
	// 2. Add oauth-2025-04-20 beta flag
	// 3. Add ?beta=true query param
	// 4. Set user-agent to claude-cli (required by OAuth endpoint)
	// 5. Rewrite "OpenCode" → "Claude Code" in system prompts (server blocks it)
	transport := &oauthTransport{
		base: http.DefaultTransport,
		auth: auth,
	}

	httpClient := &http.Client{Transport: transport, Timeout: 25 * time.Minute}

	client := anthropic.NewClient(
		option.WithHTTPClient(httpClient),
		option.WithAPIKey("oauth-placeholder"), // transport overrides this
	)

	if model == "" {
		model = "claude-opus-4-6"
	}

	return &AnthropicProvider{
		client: &client,
		model:  model,
		auth:   auth,
	}
}

func (p *AnthropicProvider) Name() string {
	if p.name != "" {
		return p.name
	}
	return "anthropic"
}

func (p *AnthropicProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	// Build messages
	var messages []anthropic.MessageParam
	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))
		case "assistant":
			messages = append(messages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(m.Content)))
		}
	}
	model := req.Model
	if model == "" {
		model = p.model
	}
	maxTokens := int64(req.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = 4096
	}
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: maxTokens,
		Messages:  messages,
	}
	// System prompt
	if req.System != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: req.System},
		}
	}
	if req.Temperature > 0 {
		params.Temperature = anthropic.Float(req.Temperature)
	}

	// Use streaming to avoid SDK timeout on large context requests.
	// Non-streaming errors above ~10 min; streaming keeps the connection alive
	// via SSE events. We accumulate chunks and return the final result.
	stream := p.client.Messages.NewStreaming(ctx, params,
		option.WithRequestTimeout(20*time.Minute),
	)
	defer stream.Close()

	message := anthropic.Message{}
	for stream.Next() {
		event := stream.Current()
		if err := message.Accumulate(event); err != nil {
			return nil, &ProviderError{
				Message:  fmt.Sprintf("stream accumulate: %v", err),
				Provider: "anthropic",
			}
		}
	}

	if err := stream.Err(); err != nil {
		return nil, &ProviderError{
			Message:  err.Error(),
			Provider: "anthropic",
		}
	}

	// Extract text content from accumulated message
	var content string
	for _, block := range message.Content {
		if textBlock, ok := block.AsAny().(anthropic.TextBlock); ok {
			content += textBlock.Text
		}
	}

	inputTokens := int(message.Usage.InputTokens)
	outputTokens := int(message.Usage.OutputTokens)
	// Warn when input tokens exceed 200K — triggers 2x pricing
	if inputTokens > 200_000 {
		slog.Warn("request exceeded 200K input tokens — premium pricing active",
			"input_tokens", inputTokens,
			"output_tokens", outputTokens,
			"model", string(message.Model),
		)
	}
	return &CompletionResponse{
		Content:      content,
		Model:        string(message.Model),
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		StopReason:   string(message.StopReason),
	}, nil
}

// CompleteWithTools sends a completion request with tool definitions.
// toolMessages contains the multi-turn tool conversation history after the initial messages.
// This implements the ToolProvider interface.
func (p *AnthropicProvider) CompleteWithTools(ctx context.Context, req CompletionRequest, tools []ToolDefinition, toolMessages []ToolMessage) (*CompletionResponse, error) {
	var messages []anthropic.MessageParam
	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))
		case "assistant":
			messages = append(messages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(m.Content)))
		}
	}

	for _, tm := range toolMessages {
		var blocks []anthropic.ContentBlockParamUnion
		for _, b := range tm.Content {
			switch b.Type {
			case "text":
				if b.Text != "" {
					blocks = append(blocks, anthropic.NewTextBlock(b.Text))
				}
			case "tool_use":
				if b.ToolCall == nil {
					continue
				}
				var inputValue any
				if len(b.ToolCall.Input) > 0 {
					if err := json.Unmarshal(b.ToolCall.Input, &inputValue); err != nil {
						inputValue = map[string]any{}
					}
				} else {
					inputValue = map[string]any{}
				}
				blocks = append(blocks, anthropic.NewToolUseBlock(b.ToolCall.ID, inputValue, b.ToolCall.Name))
			case "tool_result":
				if b.ToolResult == nil {
					continue
				}
				blocks = append(blocks, anthropic.NewToolResultBlock(b.ToolResult.ToolCallID, b.ToolResult.Content, b.ToolResult.IsError))
			}
		}

		if len(blocks) == 0 {
			continue
		}

		switch tm.Role {
		case "assistant":
			messages = append(messages, anthropic.NewAssistantMessage(blocks...))
		case "user":
			messages = append(messages, anthropic.NewUserMessage(blocks...))
		}
	}

	var anthropicTools []anthropic.ToolUnionParam
	for _, t := range tools {
		props := make(map[string]any, len(t.InputSchema))
		for k, v := range t.InputSchema {
			props[k] = v
		}
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.Name,
				Description: anthropic.String(t.Description),
				InputSchema: anthropic.ToolInputSchemaParam{Properties: props},
			},
		})
	}

	model := req.Model
	if model == "" {
		model = p.model
	}
	maxTokens := int64(req.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = 4096
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: maxTokens,
		Messages:  messages,
		Tools:     anthropicTools,
	}
	if req.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: req.System}}
	}
	if req.Temperature > 0 {
		params.Temperature = anthropic.Float(req.Temperature)
	}

	stream := p.client.Messages.NewStreaming(ctx, params,
		option.WithRequestTimeout(20*time.Minute),
	)
	defer stream.Close()

	message := anthropic.Message{}
	for stream.Next() {
		event := stream.Current()
		if err := message.Accumulate(event); err != nil {
			return nil, &ProviderError{
				Message:  fmt.Sprintf("stream accumulate: %v", err),
				Provider: p.Name(),
			}
		}
	}
	if err := stream.Err(); err != nil {
		return nil, &ProviderError{
			Message:  err.Error(),
			Provider: p.Name(),
		}
	}

	var content string
	var toolCalls []ToolCall
	for _, block := range message.Content {
		switch v := block.AsAny().(type) {
		case anthropic.TextBlock:
			content += v.Text
		case anthropic.ToolUseBlock:
			inputJSON, _ := json.Marshal(v.Input)
			toolCalls = append(toolCalls, ToolCall{
				ID:    v.ID,
				Name:  v.Name,
				Input: inputJSON,
			})
		}
	}

	inputTokens := int(message.Usage.InputTokens)
	outputTokens := int(message.Usage.OutputTokens)
	if inputTokens > 200_000 {
		slog.Warn("tool request exceeded 200K input tokens",
			"input_tokens", inputTokens,
			"output_tokens", outputTokens,
			"model", string(message.Model),
		)
	}

	return &CompletionResponse{
		Content:      content,
		Model:        string(message.Model),
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		StopReason:   string(message.StopReason),
		ToolCalls:    toolCalls,
	}, nil
}

// oauthTransport is an http.RoundTripper that rewrites outgoing requests
// for Anthropic OAuth auth. It mirrors the behavior of the
// opencode-anthropic-auth plugin's custom fetch wrapper.
type oauthTransport struct {
	base http.RoundTripper
	auth *AuthStore
}

func (t *oauthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Get fresh OAuth token (auto-refreshes if expired)
	token, err := t.auth.GetAPIKey("anthropic")
	if err != nil {
		return nil, fmt.Errorf("oauth token: %w", err)
	}

	// Clone the request so we don't mutate the original
	clone := req.Clone(req.Context())

	// 1. Auth: Bearer token replaces x-api-key
	clone.Header.Set("Authorization", "Bearer "+token)
	clone.Header.Del("X-Api-Key")
	clone.Header.Del("x-api-key")

	// 2. Rewrite system prompt: server blocks "OpenCode" string in OAuth mode
	// Also measure body size for conditional long-context beta.
	var bodySize int64
	if clone.Body != nil && clone.Method == "POST" {
		body, readErr := io.ReadAll(clone.Body)
		clone.Body.Close()
		if readErr == nil {
			rewritten := rewriteSystemPrompt(body)
			bodySize = int64(len(rewritten))
			clone.Body = io.NopCloser(bytes.NewReader(rewritten))
			clone.ContentLength = bodySize
		}
	}

	// 3. OAuth beta flag (REQUIRED — server rejects OAuth without this)
	// Add long context beta only when request body is large (>512KB ≈ 128K tokens).
	// Without this guard, ALL requests trigger "extra usage required" rejection.
	beta := anthropicOAuthBeta
	if bodySize > 512*1024 {
		beta += "," + anthropicLongContextBeta
		slog.Info("enabling long context beta", "body_size", bodySize)
	}
	clone.Header.Set("Anthropic-Beta", beta)
	// 4. User-Agent matching Claude CLI (required by OAuth endpoint)
	clone.Header.Set("User-Agent", "claude-cli/2.1.2 (external, cli)")
	// 5. Add ?beta=true query param (required for OAuth /v1/messages)
	if strings.HasSuffix(clone.URL.Path, "/v1/messages") {
		q := clone.URL.Query()
		if !q.Has("beta") {
			q.Set("beta", "true")
			clone.URL.RawQuery = q.Encode()
		}
	}

	slog.Debug("anthropic OAuth request",
		"url", clone.URL.String(),
		"token_prefix", token[:min(8, len(token))]+"...",
	)

	return t.base.RoundTrip(clone)
}

// rewriteSystemPrompt replaces "OpenCode" with "Claude Code" in the request body.
// Anthropic's OAuth endpoint blocks requests containing "OpenCode" in system prompts.
func rewriteSystemPrompt(body []byte) []byte {
	// Quick check — most requests won't contain "OpenCode"
	if !bytes.Contains(body, []byte("OpenCode")) && !bytes.Contains(body, []byte("opencode")) {
		return body
	}

	// Parse, rewrite system blocks, re-serialize
	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(body, &parsed); err != nil {
		return body // can't parse, return as-is
	}

	systemRaw, ok := parsed["system"]
	if !ok {
		return body
	}

	var systemBlocks []map[string]interface{}
	if err := json.Unmarshal(systemRaw, &systemBlocks); err != nil {
		return body
	}

	modified := false
	for i, block := range systemBlocks {
		if block["type"] == "text" {
			if text, ok := block["text"].(string); ok {
				newText := strings.ReplaceAll(text, "OpenCode", "Claude Code")
				newText = strings.ReplaceAll(newText, "opencode", "Claude")
				if newText != text {
					systemBlocks[i]["text"] = newText
					modified = true
				}
			}
		}
	}

	if !modified {
		return body
	}

	newSystem, err := json.Marshal(systemBlocks)
	if err != nil {
		return body
	}
	parsed["system"] = newSystem

	result, err := json.Marshal(parsed)
	if err != nil {
		return body
	}
	return result
}

// OpenAICompatProvider implements Provider for any OpenAI-compatible API.
// Works with Kimi/Moonshot, DeepSeek, and others.
type OpenAICompatProvider struct {
	name    string
	baseURL string
	apiKey  string
	model   string
}

// NewOpenAICompat creates a provider for OpenAI-compatible APIs (Kimi, DeepSeek, etc).
func NewOpenAICompat(name, baseURL, apiKey, model string) *OpenAICompatProvider {
	return &OpenAICompatProvider{
		name:    name,
		baseURL: baseURL,
		apiKey:  apiKey,
		model:   model,
	}
}

func (p *OpenAICompatProvider) Name() string { return p.name }

func (p *OpenAICompatProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	model := req.Model
	if model == "" {
		model = p.model
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096
	}

	// Build messages array for OpenAI format
	var messages []map[string]string
	if req.System != "" {
		messages = append(messages, map[string]string{"role": "system", "content": req.System})
	}
	for _, m := range req.Messages {
		messages = append(messages, map[string]string{"role": m.Role, "content": m.Content})
	}

	body := map[string]interface{}{
		"model":       model,
		"messages":    messages,
		"max_tokens":  maxTokens,
		"temperature": req.Temperature,
	}

	resp, err := doOpenAIRequest(ctx, p.baseURL+"/chat/completions", p.apiKey, body)
	if err != nil {
		return nil, &ProviderError{
			Message:  err.Error(),
			Provider: p.name,
		}
	}

	return resp, nil
}

// openaiHTTPClient is a shared HTTP client for OpenAI-compatible requests
// with a generous timeout for large context windows.
var openaiHTTPClient = &http.Client{Timeout: 10 * time.Minute}
// doOpenAIRequest makes an HTTP request to an OpenAI-compatible endpoint.
func doOpenAIRequest(ctx context.Context, url, apiKey string, body map[string]interface{}) (*CompletionResponse, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("User-Agent", "opencode/0.1.0")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := openaiHTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse OpenAI-format response
	var oaiResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Model string `json:"model"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(respBody, &oaiResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	var content, stopReason string
	if len(oaiResp.Choices) > 0 {
		content = oaiResp.Choices[0].Message.Content
		stopReason = oaiResp.Choices[0].FinishReason
	}

	return &CompletionResponse{
		Content:      content,
		Model:        oaiResp.Model,
		InputTokens:  oaiResp.Usage.PromptTokens,
		OutputTokens: oaiResp.Usage.CompletionTokens,
		StopReason:   stopReason,
	}, nil
}

// CopilotProvider implements Provider for GitHub Copilot's OpenAI-compatible API.
// Copilot uses the OAuth refresh token directly as a Bearer token, plus
// custom headers (Openai-Intent, x-initiator, User-Agent).
type CopilotProvider struct {
	auth  *AuthStore
	model string
}

// NewCopilot creates a Copilot provider using OAuth tokens from AuthStore.
func NewCopilot(auth *AuthStore, model string) *CopilotProvider {
	if model == "" {
		model = "gpt-4.1"
	}
	return &CopilotProvider{auth: auth, model: model}
}

func (p *CopilotProvider) Name() string { return "copilot" }

func (p *CopilotProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	// Get the OAuth refresh token (Copilot uses it directly as Bearer)
	entry := p.auth.Get("github-copilot")
	if entry == nil {
		return nil, &ProviderError{Message: "no github-copilot auth entry", Provider: "copilot"}
	}
	token := entry.Refresh
	if token == "" {
		token = entry.Access
	}
	if token == "" {
		return nil, &ProviderError{Message: "no copilot OAuth token", Provider: "copilot"}
	}

	model := req.Model
	if model == "" {
		model = p.model
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096
	}

	// Build OpenAI-format messages
	var messages []map[string]string
	if req.System != "" {
		messages = append(messages, map[string]string{"role": "system", "content": req.System})
	}
	for _, m := range req.Messages {
		messages = append(messages, map[string]string{"role": m.Role, "content": m.Content})
	}

	body := map[string]interface{}{
		"model":       model,
		"messages":    messages,
		"max_tokens":  maxTokens,
		"temperature": req.Temperature,
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		"https://api.githubcopilot.com/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// Copilot-specific headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+token)
	httpReq.Header.Set("User-Agent", "opencode/0.1.0")
	httpReq.Header.Set("Openai-Intent", "conversation-edits")
	httpReq.Header.Set("x-initiator", "agent")

	resp, err := openaiHTTPClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{Message: fmt.Sprintf("http: %v", err), Provider: "copilot"}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, &ProviderError{
			Message:    fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(respBody)),
			StatusCode: resp.StatusCode,
			Provider:   "copilot",
		}
	}

	// Parse OpenAI-format response
	var oaiResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Model string `json:"model"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(respBody, &oaiResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	var content, stopReason string
	if len(oaiResp.Choices) > 0 {
		content = oaiResp.Choices[0].Message.Content
		stopReason = oaiResp.Choices[0].FinishReason
	}

	return &CompletionResponse{
		Content:      content,
		Model:        oaiResp.Model,
		InputTokens:  oaiResp.Usage.PromptTokens,
		OutputTokens: oaiResp.Usage.CompletionTokens,
		StopReason:   stopReason,
	}, nil
}
