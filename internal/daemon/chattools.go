package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/nous-labs/nous/internal/llm"
	"github.com/nous-labs/nous/pkg/brain"
)

const (
	maxToolTurns         = 5
	maxToolLoopDuration  = 5 * time.Minute
	maxSingleToolTimeout = 10 * time.Second
	maxWebFetchBytes     = 10 * 1024
	proxySocketPath      = "/var/lib/nousctl/nous-proxy.sock"
	proxyExecTimeout     = 5 * time.Minute
)

var chatToolHTTPClient = &http.Client{
	Timeout: maxSingleToolTimeout,
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 5 {
			return fmt.Errorf("too many redirects")
		}
		if !strings.EqualFold(req.URL.Scheme, "https") {
			return fmt.Errorf("redirected to non-https URL")
		}
		if err := validateExternalHost(req.Context(), req.URL.Hostname()); err != nil {
			return err
		}
		return nil
	},
}

type ToolExecutor interface {
	Definition() llm.ToolDefinition
	Execute(ctx context.Context, input json.RawMessage, userPrompt string) (string, error)
}

type daemonToolExecutor struct {
	definition llm.ToolDefinition
	run        func(ctx context.Context, input map[string]any, userPrompt string) (string, error)
	timeout    time.Duration // 0 means use default maxSingleToolTimeout
}

func (e daemonToolExecutor) Definition() llm.ToolDefinition {
	return e.definition
}

func (e daemonToolExecutor) Execute(ctx context.Context, input json.RawMessage, userPrompt string) (string, error) {
	parsed := map[string]any{}
	if len(input) > 0 {
		if err := json.Unmarshal(input, &parsed); err != nil {
			return "", fmt.Errorf("parse tool input: %w", err)
		}
	}
	return e.run(ctx, parsed, userPrompt)
}

func (d *Daemon) chatToolExecutors() []ToolExecutor {
	return []ToolExecutor{
		daemonToolExecutor{
			definition: llm.ToolDefinition{
				Name:        "web_fetch",
				Description: "Fetch HTTPS URL content for factual lookups. Blocks internal/private hosts.",
				InputSchema: map[string]interface{}{
					"url": map[string]interface{}{
						"type": "string",
					},
				},
			},
			run: d.toolWebFetch,
		},
		daemonToolExecutor{
			definition: llm.ToolDefinition{
				Name:        "github_latest_release",
				Description: "Get latest release for a GitHub repository in owner/name format.",
				InputSchema: map[string]interface{}{
					"repo": map[string]interface{}{
						"type": "string",
					},
				},
			},
			run: d.toolGitHubLatestRelease,
		},
		daemonToolExecutor{
			definition: llm.ToolDefinition{
				Name:        "brain_recall",
				Description: "Search persistent memory by query.",
				InputSchema: map[string]interface{}{
					"query": map[string]interface{}{
						"type": "string",
					},
					"limit": map[string]interface{}{
						"type": "number",
					},
					"scope": map[string]interface{}{
						"type": "string",
					},
					"type": map[string]interface{}{
						"type": "string",
					},
				},
			},
			run: d.toolBrainRecall,
		},
		daemonToolExecutor{
			definition: llm.ToolDefinition{
				Name:        "brain_capture",
				Description: "Store a memory when the user explicitly asks to remember/store it.",
				InputSchema: map[string]interface{}{
					"type": map[string]interface{}{
						"type": "string",
					},
					"scope": map[string]interface{}{
						"type": "string",
					},
					"content": map[string]interface{}{
						"type": "string",
					},
					"tags": map[string]interface{}{
						"type": "string",
					},
				},
			},
			run: d.toolBrainCapture,
		},
		daemonToolExecutor{
			definition: llm.ToolDefinition{
				Name:        "proxy_exec",
				Description: "Execute a whitelisted command on a host project via nous-proxy. Use for docker, git, build operations. Examples: docker compose up -d --build api, git status, ollama list.",
				InputSchema: map[string]interface{}{
					"project": map[string]interface{}{
						"type":        "string",
						"description": "Project name (e.g. sona-mp, omo-config)",
					},
					"command": map[string]interface{}{
						"type":        "string",
						"description": "Space-separated command (e.g. 'docker compose up -d --build api')",
					},
				},
			},
			run:     d.toolProxyExec,
			timeout: proxyExecTimeout,
		},
	}
}

func (d *Daemon) executeToolCall(ctx context.Context, byName map[string]ToolExecutor, call llm.ToolCall, userPrompt string) llm.ToolResult {
	start := time.Now()
	result := llm.ToolResult{ToolCallID: call.ID}
	executor, ok := byName[call.Name]
	if !ok {
		result.IsError = true
		result.Content = fmt.Sprintf("unknown tool: %s", call.Name)
		slog.Info("chat tool call", "tool", call.Name, "duration", time.Since(start).Round(time.Millisecond), "is_error", true)
		return result
	}

	timeout := maxSingleToolTimeout
	if dte, ok := executor.(daemonToolExecutor); ok && dte.timeout > 0 {
		timeout = dte.timeout
	}
	toolCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	content, err := executor.Execute(toolCtx, call.Input, userPrompt)
	if err != nil {
		result.IsError = true
		result.Content = err.Error()
	} else {
		result.Content = content
	}

	slog.Info("chat tool call",
		"tool", call.Name,
		"duration", time.Since(start).Round(time.Millisecond),
		"is_error", result.IsError,
	)
	return result
}

func (d *Daemon) toolWebFetch(ctx context.Context, input map[string]any, _ string) (string, error) {
	urlText, _ := input["url"].(string)
	if strings.TrimSpace(urlText) == "" {
		return "", fmt.Errorf("missing required parameter: url")
	}

	parsedURL, err := url.Parse(urlText)
	if err != nil {
		return "", fmt.Errorf("invalid url: %w", err)
	}
	if !strings.EqualFold(parsedURL.Scheme, "https") {
		return "", fmt.Errorf("only https URLs are allowed")
	}
	host := parsedURL.Hostname()
	if host == "" {
		return "", fmt.Errorf("url must include a hostname")
	}
	if err := validateExternalHost(ctx, host); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, parsedURL.String(), nil)
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("User-Agent", "nous-daemon/1.0")

	resp, err := chatToolHTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxWebFetchBytes+1))
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}
	truncated := false
	if len(body) > maxWebFetchBytes {
		body = body[:maxWebFetchBytes]
		truncated = true
	}

	content := string(body)
	if truncated {
		content += "\n[truncated at 10KB]"
	}
	return content, nil
}

func (d *Daemon) toolGitHubLatestRelease(ctx context.Context, input map[string]any, _ string) (string, error) {
	repo, _ := input["repo"].(string)
	repo = strings.TrimSpace(repo)
	parts := strings.Split(repo, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", fmt.Errorf("repo must be in owner/name format")
	}

	endpoint := "https://api.github.com/repos/" + repo + "/releases/latest"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "nous-daemon/1.0")

	resp, err := chatToolHTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("github request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxWebFetchBytes+1))
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode == http.StatusNotFound {
		return "", fmt.Errorf("no release found for %s", repo)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("github API HTTP %d", resp.StatusCode)
	}

	var payload struct {
		TagName     string `json:"tag_name"`
		Name        string `json:"name"`
		HTMLURL     string `json:"html_url"`
		PublishedAt string `json:"published_at"`
		Body        string `json:"body"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}

	b := strings.Builder{}
	b.WriteString("Repository: ")
	b.WriteString(repo)
	b.WriteString("\nLatest release: ")
	b.WriteString(payload.TagName)
	if payload.Name != "" && payload.Name != payload.TagName {
		b.WriteString(" (")
		b.WriteString(payload.Name)
		b.WriteString(")")
	}
	if payload.PublishedAt != "" {
		if t, err := time.Parse(time.RFC3339, payload.PublishedAt); err == nil {
			b.WriteString("\nPublished: ")
			b.WriteString(t.Format("2006-01-02"))
		}
	}
	if payload.HTMLURL != "" {
		b.WriteString("\nURL: ")
		b.WriteString(payload.HTMLURL)
	}
	if payload.Body != "" {
		notes := payload.Body
		if len(notes) > 1500 {
			notes = notes[:1500] + "\n[truncated]"
		}
		b.WriteString("\n\nRelease notes:\n")
		b.WriteString(notes)
	}

	return b.String(), nil
}

func (d *Daemon) toolBrainRecall(_ context.Context, input map[string]any, _ string) (string, error) {
	query, _ := input["query"].(string)
	if strings.TrimSpace(query) == "" {
		return "", fmt.Errorf("missing required parameter: query")
	}

	limit := 5
	if v, ok := input["limit"]; ok {
		switch n := v.(type) {
		case float64:
			limit = int(n)
		case string:
			if parsed, err := strconv.Atoi(n); err == nil {
				limit = parsed
			}
		}
	}
	if limit <= 0 {
		limit = 5
	}
	if limit > 20 {
		limit = 20
	}

	opts := brain.RecallOptions{Limit: limit}
	if scope, _ := input["scope"].(string); scope != "" {
		opts.Scope = scope
	}
	if typ, _ := input["type"].(string); typ != "" {
		opts.Type = typ
	}

	memories, err := d.brain.Recall(query, opts)
	if err != nil {
		return "", fmt.Errorf("brain recall failed: %w", err)
	}
	if len(memories) == 0 {
		return "No matching memories found.", nil
	}

	b := strings.Builder{}
	b.WriteString("Memories:\n")
	for _, m := range memories {
		b.WriteString("[")
		b.WriteString(strconv.Itoa(m.ID))
		b.WriteString("] ")
		b.WriteString(m.Type)
		if m.Scope != "" {
			b.WriteString("/")
			b.WriteString(m.Scope)
		}
		b.WriteString(" - ")
		content := m.Content
		if len(content) > 300 {
			content = content[:300] + "..."
		}
		b.WriteString(content)
		b.WriteString("\n")
	}

	return b.String(), nil
}

func (d *Daemon) toolBrainCapture(_ context.Context, input map[string]any, userPrompt string) (string, error) {
	if !isExplicitBrainCaptureRequest(userPrompt) {
		return "", fmt.Errorf("brain_capture blocked: user did not explicitly request memory capture")
	}

	typ, _ := input["type"].(string)
	if typ == "" {
		typ = "observation"
	}
	scope, _ := input["scope"].(string)
	if scope == "" {
		scope = "global"
	}
	content, _ := input["content"].(string)
	if strings.TrimSpace(content) == "" {
		return "", fmt.Errorf("missing required parameter: content")
	}
	tags, _ := input["tags"].(string)

	id, err := d.brain.Capture(typ, scope, content, tags, "nous-daemon-chat")
	if err != nil {
		return "", fmt.Errorf("brain capture failed: %w", err)
	}

	return fmt.Sprintf("captured memory %d", id), nil
}

func isExplicitBrainCaptureRequest(userPrompt string) bool {
	s := strings.ToLower(userPrompt)
	phrases := []string{
		"remember this",
		"save this",
		"store this",
		"capture this",
		"record this",
		"add to memory",
		"note this",
		"memorize this",
	}
	for _, p := range phrases {
		if strings.Contains(s, p) {
			return true
		}
	}
	return false
}

func validateExternalHost(ctx context.Context, host string) error {
	if strings.EqualFold(host, "localhost") {
		return fmt.Errorf("localhost is blocked")
	}
	if ip := net.ParseIP(host); ip != nil {
		if isBlockedIP(ip) {
			return fmt.Errorf("internal IP is blocked")
		}
		return nil
	}

	resolver := net.Resolver{}
	resolved, err := resolver.LookupIPAddr(ctx, host)
	if err != nil {
		return fmt.Errorf("dns lookup failed: %w", err)
	}
	if len(resolved) == 0 {
		return fmt.Errorf("host did not resolve")
	}
	for _, addr := range resolved {
		if isBlockedIP(addr.IP) {
			return fmt.Errorf("host resolves to blocked IP")
		}
	}
	return nil
}

func isBlockedIP(ip net.IP) bool {
	if ip == nil {
		return true
	}
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified() {
		return true
	}
	if ip4 := ip.To4(); ip4 != nil {
		if ip4[0] == 127 {
			return true
		}
	}
	return false
}

func (d *Daemon) runToolLoop(ctx context.Context, tier llm.Tier, req llm.CompletionRequest, userPrompt string) (string, error) {
	loopCtx, cancel := context.WithTimeout(ctx, maxToolLoopDuration)
	defer cancel()

	executors := d.chatToolExecutors()
	tools := make([]llm.ToolDefinition, 0, len(executors))
	byName := make(map[string]ToolExecutor, len(executors))
	for _, exec := range executors {
		def := exec.Definition()
		tools = append(tools, def)
		byName[def.Name] = exec
	}

	toolMessages := make([]llm.ToolMessage, 0, maxToolTurns*2)
	for turn := 0; turn < maxToolTurns; turn++ {
		resp, err := d.router.CompleteWithTools(loopCtx, tier, req, tools, toolMessages)
		if err != nil {
			return "", fmt.Errorf("tool turn %d failed: %w", turn+1, err)
		}

		if resp.StopReason != "tool_use" || len(resp.ToolCalls) == 0 {
			return resp.Content, nil
		}

		assistantBlocks := make([]llm.ContentBlock, 0, len(resp.ToolCalls)+1)
		if resp.Content != "" {
			assistantBlocks = append(assistantBlocks, llm.ContentBlock{Type: "text", Text: resp.Content})
		}
		for _, tc := range resp.ToolCalls {
			copyCall := tc
			assistantBlocks = append(assistantBlocks, llm.ContentBlock{Type: "tool_use", ToolCall: &copyCall})
		}
		toolMessages = append(toolMessages, llm.ToolMessage{Role: "assistant", Content: assistantBlocks})

		userBlocks := make([]llm.ContentBlock, 0, len(resp.ToolCalls))
		for _, tc := range resp.ToolCalls {
			result := d.executeToolCall(loopCtx, byName, tc, userPrompt)
			copyResult := result
			userBlocks = append(userBlocks, llm.ContentBlock{Type: "tool_result", ToolResult: &copyResult})
		}
		toolMessages = append(toolMessages, llm.ToolMessage{Role: "user", Content: userBlocks})
	}

	return "", fmt.Errorf("tool loop exceeded %d turns", maxToolTurns)
}

// --- Nous Proxy (host command execution) ---

// proxyExec executes a command on the host via nous-proxy Unix socket.
// Used by both the /exec Matrix command and the proxy_exec chat tool.
func (d *Daemon) proxyExec(ctx context.Context, project string, command []string, cwd string) (string, error) {
	reqBody := map[string]interface{}{
		"project": project,
		"command": command,
	}
	if cwd != "" {
		reqBody["cwd"] = cwd
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	// HTTP client over Unix socket
	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
				return net.DialTimeout("unix", proxySocketPath, 5*time.Second)
			},
		},
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://proxy/v1/exec", bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("proxy request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 50*1024)) // 50KB limit
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	var result struct {
		Status     string `json:"status"`
		ReturnCode int    `json:"returncode"`
		Stdout     string `json:"stdout"`
		Stderr     string `json:"stderr"`
		Error      string `json:"error"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}

	if result.Error != "" {
		return "", fmt.Errorf("proxy: %s", result.Error)
	}

	// Format output
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Project: %s\nCommand: %s\nExit: %d\n",
		project, strings.Join(command, " "), result.ReturnCode))
	if result.Stdout != "" {
		stdout := result.Stdout
		if len(stdout) > 5000 {
			stdout = stdout[:5000] + "\n[truncated]"
		}
		sb.WriteString("\n")
		sb.WriteString(stdout)
	}
	if result.Stderr != "" {
		stderr := result.Stderr
		if len(stderr) > 2000 {
			stderr = stderr[:2000] + "\n[truncated]"
		}
		if result.ReturnCode != 0 {
			sb.WriteString("\n--- stderr ---\n")
		}
		sb.WriteString(stderr)
	}
	return sb.String(), nil
}

// toolProxyExec is the chat tool wrapper for proxyExec.
func (d *Daemon) toolProxyExec(ctx context.Context, input map[string]any, _ string) (string, error) {
	project, _ := input["project"].(string)
	if project == "" {
		return "", fmt.Errorf("missing required parameter: project")
	}

	var command []string
	switch v := input["command"].(type) {
	case []interface{}:
		for _, item := range v {
			if s, ok := item.(string); ok {
				command = append(command, s)
			}
		}
	case string:
		command = strings.Fields(v)
	}
	if len(command) == 0 {
		return "", fmt.Errorf("missing required parameter: command")
	}

	cwd, _ := input["cwd"].(string)
	return d.proxyExec(ctx, project, command, cwd)
}
