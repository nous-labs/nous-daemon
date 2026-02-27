# nous-daemon

A modular Go daemon framework for building AI assistants with persistent memory, semantic search, and autonomous maintenance.

## Architecture

```
cmd/nous/          Entry point
pkg/               Open-source core (importable)
├── brain/         Persistent memory (SQLite, FTS5, bi-temporal)
├── channel/       Communication channel interface
├── daemon/        Module system, config, event bus, HTTP server
├── dream/         Autonomous memory maintenance worker
└── embeddings/    Semantic search (pgvector + TEI + hybrid RRF)
internal/          Private implementation
├── channel/       Channel implementations (Matrix)
├── daemon/        Config loading, workspace API, chat tools
├── llm/           LLM provider routing (Anthropic, etc.)
└── tools/         Tool implementations (dispatch, opencode)
configs/           Configuration templates
```

## Core Packages (`pkg/`)

### `pkg/brain` — Persistent Memory

SQLite-backed memory system with FTS5 full-text search and bi-temporal versioning.

- **Memory types**: decision, preference, observation, failure, pattern, fact
- **Scoping**: memories belong to scopes (global, per-project)
- **Tasks & entities**: built-in task tracker and entity registry
- **Key-value store**: simple persistent storage
- **Episodes**: session-like groupings with start/end
- **Access tracking**: frequency scoring with configurable decay

### `pkg/daemon` — Module System

Pluggable daemon framework with lifecycle management.

```go
type Module interface {
    Name() string
    Init(d *Daemon) error
    RegisterRoutes(mux *http.ServeMux)
    Start(ctx context.Context) error
    Stop() error
}
```

- **Config overlay**: base defaults → config file → `NOUS_PRIVATE_CONFIG` env overlay
- **EventBus**: fan-out event broadcasting with ring buffer for late joiners
- **HTTP server**: health endpoint, recall API, SSE event stream, module routes

### `pkg/dream` — Memory Maintenance

Background worker that autonomously maintains memory health:

- **Stale detection**: flags memories past their TTL using configurable half-life decay
- **Orphan cleanup**: closes abandoned episodes
- **Access decay**: recalculates frequency scores over time
- **Safe by design**: read-mostly, never deletes — only flags or bi-temporally invalidates
- **Gated destruction**: auto-invalidation only at extreme staleness (score ≥ 0.95)

### `pkg/embeddings` — Semantic Search

Vector embeddings via HuggingFace TEI + pgvector, with hybrid search:

- **TEI client**: generates embeddings via Text Embeddings Inference
- **pgvector store**: PostgreSQL vector storage with cosine similarity
- **Sync worker**: background sync from SQLite brain → pgvector
- **Hybrid search**: combines vector similarity + FTS5 keyword search via Reciprocal Rank Fusion (RRF)

### `pkg/channel` — Communication Interface

Abstract interface for communication channels:

```go
type Channel interface {
    Name() string
    Start(ctx context.Context, handler MessageHandler) error
    Send(ctx context.Context, resp Response) error
    Stop() error
}
```

## Configuration

The daemon uses a three-layer config system:

1. **Defaults** — sensible defaults, overridable via environment variables
2. **Config file** — JSON file passed via `-config` flag or `NOUS_CONFIG_PATH`
3. **Private overlay** — `NOUS_PRIVATE_CONFIG` env var points to a JSON file with secrets

Environment variables in config values (prefixed with `$`) are resolved at load time.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NOUS_BRAIN_PATH` | `brain` | Path to brain directory |
| `NOUS_CONFIG_PATH` | — | Config file path |
| `NOUS_PRIVATE_CONFIG` | — | Private config overlay (secrets) |
| `NOUS_HTTP_ADDR` | `:8080` | HTTP listen address |
| `NOUS_WORKSPACE_ENABLED` | `1` | Enable workspace API |
| `NOUS_WORKSPACE_SOCKET` | `/tmp/nous.sock` | Unix socket for workspace API |
| `NOUS_WORKSPACE_TCP` | `:8090` | TCP address for workspace API |
| `NOUS_EMBEDDINGS_ENABLED` | — | Set to enable embeddings module |
| `NOUS_PG_URL` | — | PostgreSQL connection URL (for embeddings) |
| `NOUS_TEI_URL` | — | TEI server URL (for embeddings) |
| `NOUS_EMBED_SYNC_INTERVAL` | `30s` | Embedding sync interval |
| `NOUS_DREAM_INTERVAL` | `6h` | Dream cycle interval |
| `NOUS_DREAM_DISABLED` | — | Set to disable dream worker |

## Building

```bash
make build          # Build binary to bin/nous
make test           # Run tests
make docker         # Build Docker image
make dev            # Build and run with local brain
make tidy           # Run go mod tidy
make clean          # Clean build artifacts

## Docker

```bash
docker build -t nous-daemon .
docker run -v /path/to/brain:/brain \
           -v /path/to/data:/data \
           -e NOUS_PRIVATE_CONFIG=/brain/secrets/config.json \
           nous-daemon
```

The image is a minimal Alpine container (~15MB) running as non-root.

## HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check with uptime |
| `/v1/recall?q=...` | GET | Memory recall (hybrid or keyword) |
| `/v1/events` | GET | SSE event stream |
| `/v1/chat` | POST | Workspace chat (via workspace API) |
| `/v1/tasks` | GET/POST | Task management (via workspace API) |
| `/v1/command` | POST | Execute workspace command |
| `/v1/dream` | POST | Trigger dream cycle |
| `/v1/dispatch` | POST | Cross-project dispatch |
| `/v1/panels` | GET | Panel data (via workspace API) |
| `/v1/health` | GET | Workspace health (via workspace API) |
Modules register additional routes via `RegisterRoutes`.

## Using as a Library

The `pkg/` packages are designed to be imported independently:

```go
import (
    "github.com/nous-labs/nous/pkg/brain"
    "github.com/nous-labs/nous/pkg/daemon"
    "github.com/nous-labs/nous/pkg/dream"
    "github.com/nous-labs/nous/pkg/embeddings"
    "github.com/nous-labs/nous/pkg/channel"
)

// Open a brain
b, _ := brain.Open("/path/to/brain")
defer b.Close()

// Create daemon with custom modules
cfg := &daemon.Config{Name: "my-bot", HTTPAddr: ":9090"}
d, _ := daemon.New(b, cfg)
d.RegisterModule(myModule)
d.Run(ctx)
```

## License

MIT
