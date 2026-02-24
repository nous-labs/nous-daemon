package llm

import (
	"encoding/json"
)

// ToolDefinition describes a tool the LLM can call.
type ToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

// ToolCall represents the LLM requesting a tool execution.
type ToolCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

// ToolResult is the result of executing a tool.
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
	IsError    bool   `json:"is_error"`
}

type ContentBlock struct {
	Type       string      `json:"type"`
	Text       string      `json:"text,omitempty"`
	ToolCall   *ToolCall   `json:"tool_call,omitempty"`
	ToolResult *ToolResult `json:"tool_result,omitempty"`
}

type ToolMessage struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}
