// Package daemon — workspace API handlers for TUI connections.
//
// These endpoints are served over a Unix socket for local TUI access.
// Protocol matches the plan in brain/plans/nous-workspace/current.md:
//
//	POST /v1/chat     — send message, get response
//	POST /v1/command  — execute command (/oc, /project, etc.)
//	GET  /v1/events   — SSE stream (chat responses, panel frames, status)
//	GET  /v1/panels   — list active panels and their latest content
//	GET  /v1/tasks    — brain tasks
//	GET  /v1/health   — health check
package daemon

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/nous-labs/nous/pkg/channel"
	coredaemon "github.com/nous-labs/nous/pkg/daemon"
)

const (
	// DefaultSocketPath is the default Unix socket for workspace TUI connections.
	DefaultSocketPath = "/tmp/nous.sock"
)

// WorkspaceConfig holds workspace API configuration.
type WorkspaceConfig struct {
	SocketPath string `json:"socket_path"` // Unix socket path
	TCPAddr    string `json:"tcp_addr"`    // TCP address (e.g. ":8090")
	Enabled    bool   `json:"enabled"`
}

// workspaceServer handles TUI connections over Unix socket.
type workspaceServer struct {
	daemon   *Daemon
	listener net.Listener
}

// startWorkspace starts the workspace API server for TUI connections.
// Listens on Unix socket (for host-local TUI) and optionally TCP (for containers/testing).
// Runs in background; cancels via context.
func (d *Daemon) startWorkspace(ctx context.Context) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat", d.handleWorkspaceChat)
	mux.HandleFunc("/v1/events", d.handleWorkspaceEvents)
	mux.HandleFunc("/v1/tasks", d.handleWorkspaceTasks)
	mux.HandleFunc("/v1/panels", d.handleWorkspacePanels)
	mux.HandleFunc("/v1/command", d.handleWorkspaceCommand)
	mux.HandleFunc("/v1/health", d.handleWorkspaceHealth)
	mux.HandleFunc("/v1/dream", d.handleWorkspaceDream)
	mux.HandleFunc("/v1/dispatch", d.handleWorkspaceDispatch)
	// Unix socket listener
	sockPath := d.config.Workspace.SocketPath
	if sockPath == "" {
		sockPath = DefaultSocketPath
	}
	if _, err := os.Stat(sockPath); err == nil {
		os.Remove(sockPath)
	}
	unixListener, err := net.Listen("unix", sockPath)
	if err != nil {
		slog.Warn("workspace unix socket failed, trying TCP only", "error", err)
	} else {
		os.Chmod(sockPath, 0660)
		unixSrv := &http.Server{Handler: mux}
		go func() {
			<-ctx.Done()
			unixSrv.Close()
			unixListener.Close()
			os.Remove(sockPath)
		}()
		go func() {
			if err := unixSrv.Serve(unixListener); err != http.ErrServerClosed {
				slog.Error("workspace unix server error", "error", err)
			}
		}()
		slog.Info("workspace API listening", "socket", sockPath)
	}

	// TCP listener (for container-to-container or testing)
	tcpAddr := d.config.Workspace.TCPAddr
	if tcpAddr == "" {
		tcpAddr = ":8090" // default workspace TCP port
	}
	tcpListener, err := net.Listen("tcp", tcpAddr)
	if err != nil {
		slog.Warn("workspace TCP listener failed", "error", err)
	} else {
		tcpSrv := &http.Server{Handler: mux}
		go func() {
			<-ctx.Done()
			tcpSrv.Close()
		}()
		go func() {
			if err := tcpSrv.Serve(tcpListener); err != http.ErrServerClosed {
				slog.Error("workspace TCP server error", "error", err)
			}
		}()
		slog.Info("workspace API listening", "tcp", tcpAddr)
	}

	slog.Info("workspace endpoints registered",
		"endpoints", []string{"/v1/chat", "/v1/events", "/v1/tasks", "/v1/panels", "/v1/command", "/v1/health", "/v1/dream", "/v1/dispatch"})

	return nil
}

// --- Chat Handler ---

// chatRequest is the JSON body for POST /v1/chat.
type chatRequest struct {
	Message string `json:"message"`
}

// chatResponse is the JSON response for POST /v1/chat.
type chatResponse struct {
	Content string `json:"content"`
	Route   string `json:"route"`
	Model   string `json:"model,omitempty"`
	Elapsed string `json:"elapsed"`
}

// handleWorkspaceChat handles POST /v1/chat — send message, get response.
func (d *Daemon) handleWorkspaceChat(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprint(w, `{"error":"method not allowed, use POST"}`)
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Message == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":"missing or invalid message field"}`)
		return
	}

	// Build a channel.Message from the workspace request
	msg := channel.Message{
		Content:  req.Message,
		Source:   "workspace",
		SenderID: "tui",
		RoomID:   "workspace", // workspace gets its own conversation history
	}

	// Publish the user message as an event
	d.events.Publish(coredaemon.Event{Type: coredaemon.EventChat, Role: "user", Content: req.Message})

	// Process through the same onMessage pipeline
	// but capture response instead of sending to Matrix
	start := time.Now()
	response, route, err := d.processMessage(r.Context(), msg)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	elapsed := time.Since(start)

	resp := chatResponse{
		Content: response,
		Route:   route,
		Elapsed: elapsed.Round(time.Millisecond).String(),
	}
	json.NewEncoder(w).Encode(resp)
}

// --- SSE Events Handler ---

// handleWorkspaceEvents handles GET /v1/events — SSE stream.
// Sends recent events on connect, then streams new events in real-time.
func (d *Daemon) handleWorkspaceEvents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // disable nginx buffering

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	// Subscribe to event bus
	events, done := d.events.Subscribe()
	defer d.events.Unsubscribe(done)

	slog.Info("workspace SSE client connected", "subscribers", d.events.SubscriberCount())

	// Send recent events to hydrate the TUI
	for _, e := range d.events.Recent(50) {
		fmt.Fprintf(w, "data: %s\n\n", e.MarshalEvent())
	}
	flusher.Flush()

	// Stream new events
	for {
		select {
		case <-r.Context().Done():
			slog.Info("workspace SSE client disconnected", "subscribers", d.events.SubscriberCount()-1)
			return
		case evt, ok := <-events:
			if !ok {
				return // channel closed
			}
			fmt.Fprintf(w, "data: %s\n\n", evt.MarshalEvent())
			flusher.Flush()
		}
	}
}

// --- Tasks Handler ---

// handleWorkspaceTasks handles GET /v1/tasks — brain tasks list.
func (d *Daemon) handleWorkspaceTasks(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprint(w, `{"error":"method not allowed"}`)
		return
	}

	tasks, err := d.brain.ListTasks()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"tasks": tasks,
		"count": len(tasks),
	})
}

// --- Panels Handler ---

// handleWorkspacePanels handles GET /v1/panels.
// Returns current state of side panels.
func (d *Daemon) handleWorkspacePanels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprint(w, `{"error":"method not allowed"}`)
		return
	}

	// For v1, panels are: activity (recent events), tasks, opencode status
	json.NewEncoder(w).Encode(map[string]interface{}{
		"panels": []map[string]interface{}{
			{
				"id":     "activity",
				"name":   "Activity",
				"events": len(d.events.Recent(0)), // total event count
			},
			{
				"id":   "tasks",
				"name": "Tasks",
			},
			{
				"id":         "opencode",
				"name":       "OpenCode",
				"session_id": d.ocSessionID,
				"available":  d.opencode != nil,
			},
		},
	})
}

// --- Command Handler ---

// handleWorkspaceCommand handles POST /v1/command.
// Executes workspace commands like /oc, /project, /tasks.
func (d *Daemon) handleWorkspaceCommand(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprint(w, `{"error":"method not allowed"}`)
		return
	}

	var req struct {
		Command string `json:"command"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Command == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":"missing or invalid command field"}`)
		return
	}

	// Route commands
	switch {
	case req.Command == "/tasks":
		d.handleWorkspaceTasks(w, r)
	default:
		// Treat as a chat message with the command as content
		msg := channel.Message{
			Content:  req.Command,
			Source:   "workspace",
			SenderID: "tui",
			RoomID:   "workspace",
		}
		response, route, err := d.processMessage(r.Context(), msg)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		json.NewEncoder(w).Encode(map[string]string{
			"response": response,
			"route":    route,
		})
	}
}

// --- Health Handler ---

// handleWorkspaceHealth handles GET /v1/health for workspace.
func (d *Daemon) handleWorkspaceHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	stats := d.brain.Stats()
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      "ok",
		"uptime":      time.Since(d.startedAt).Round(time.Second).String(),
		"memories":    stats.Memories,
		"tasks":       stats.Tasks,
		"direct_llm":  d.hasDirectLLM,
		"opencode":    d.opencode != nil,
		"subscribers": d.events.SubscriberCount(),
	})
}

// --- Dream Handler ---

// handleWorkspaceDream handles GET /v1/dream — last dream report.
// POST /v1/dream triggers an immediate dream cycle.
func (d *Daemon) handleWorkspaceDream(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if d.dreamer == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprint(w, `{"error":"dream worker not running"}`)
		return
	}

	switch r.Method {
	case http.MethodGet:
		// Return last dream report
		report := d.dreamer.LastReport()
		if report == nil {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"no dream report yet"}`)
			return
		}
		json.NewEncoder(w).Encode(report)

	case http.MethodPost:
		// Trigger immediate dream cycle
		report := d.dreamer.DreamOnce(r.Context())
		if report == nil {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, `{"error":"dream cycle failed"}`)
			return
		}
		json.NewEncoder(w).Encode(report)

	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprint(w, `{"error":"method not allowed, use GET or POST"}`)
	}
}

// --- Dispatch Handler ---

// handleWorkspaceDispatch handles dispatch API requests.
//
//	GET  /v1/dispatch → list available projects
//	POST /v1/dispatch → dispatch task to a project
func (d *Daemon) handleWorkspaceDispatch(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case http.MethodGet:
		// List available dispatch targets
		if d.dispatch == nil {
			json.NewEncoder(w).Encode(map[string]interface{}{
				"projects":   []string{},
				"configured": false,
			})
			return
		}
		type projectInfo struct {
			Name string `json:"name"`
			URL  string `json:"url"`
		}
		projects := make([]projectInfo, 0)
		for _, name := range d.dispatch.ListProjects() {
			projects = append(projects, projectInfo{Name: name, URL: d.dispatch.ProjectURL(name)})
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"projects":   projects,
			"configured": true,
		})

	case http.MethodPost:
		// Dispatch task to a project
		var req struct {
			Project string `json:"project"`
			Message string `json:"message"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Project == "" || req.Message == "" {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":"missing project or message field"}`)
			return
		}

		if d.dispatch == nil {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, `{"error":"dispatch not configured"}`)
			return
		}

		if !d.dispatch.HasProject(req.Project) {
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error":    fmt.Sprintf("unknown project: %s", req.Project),
				"projects": d.dispatch.ListProjects(),
			})
			return
		}

		start := time.Now()
		result, err := d.dispatch.Send(r.Context(), req.Project, req.Message)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"project":    result.Project,
			"response":   result.Text,
			"session_id": result.SessionID,
			"elapsed":    time.Since(start).Round(time.Millisecond).String(),
		})

	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprint(w, `{"error":"method not allowed, use GET or POST"}`)
	}
}
