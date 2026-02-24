package daemon

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds the daemon configuration.
type Config struct {
	// Identity
	Name string `json:"name"` // "nous"

	// Matrix channel
	Matrix MatrixConfig `json:"matrix"`

	// LLM providers
	LLM LLMConfig `json:"llm"`

	// Auth — path to OpenCode's auth.json for OAuth token management.
	// When set, providers can use OAuth tokens instead of static API keys.
	AuthJSONPath string `json:"auth_json_path,omitempty"`

	// OpenCode integration
	OpenCode OpenCodeConfig `json:"opencode"`

	// Workspace TUI
	Workspace WorkspaceConfig `json:"workspace"`
	// Embeddings (semantic memory)
	Embeddings EmbeddingsConfig `json:"embeddings"`


	// Dream worker (autonomous memory maintenance)
	Dream DreamConfig `json:"dream"`
	// Projects Nous is aware of
	Projects []ProjectConfig `json:"projects"`
}

// MatrixConfig holds Matrix connection settings.
type MatrixConfig struct {
	Homeserver   string   `json:"homeserver"`    // e.g., http://synapse:8008
	UserID       string   `json:"user_id"`       // e.g., @bot:matrix.example.com
	Password     string   `json:"password"`      // bot password
	ServerName   string   `json:"server_name"`   // e.g., matrix.example.com
	AllowedUsers []string `json:"allowed_users"` // who can talk to Nous
	DataDir      string   `json:"data_dir"`      // persistent state for nio store
}

// LLMConfig holds LLM provider settings.
type LLMConfig struct {
	// Deep tier — complex reasoning (Claude)
	Deep ProviderConfig `json:"deep"`
	// Mid tier — balanced quality/cost (Copilot/GPT)
	Mid ProviderConfig `json:"mid"`
	// Fast tier — quick tasks (Kimi K2.5)
	Fast ProviderConfig `json:"fast"`
}

// ProviderConfig holds settings for a single LLM provider.
type ProviderConfig struct {
	Provider      string  `json:"provider"`                // "anthropic", "kimi"
	Model         string  `json:"model"`                   // e.g., "claude-opus-4-6", "kimi-k2.5"
	APIKey        string  `json:"api_key"`                 // can use env var reference: "$ANTHROPIC_API_KEY"
	BaseURL       string  `json:"base_url,omitempty"`      // optional override
	ContextWindow int     `json:"context_window,omitempty"` // max input tokens (e.g., 1000000 for Opus 4.6)
	MaxOutput     int     `json:"max_output,omitempty"`     // max output tokens per request
	Temperature   float64 `json:"temperature,omitempty"`    // sampling temperature (0.0-1.0)
}

// OpenCodeConfig holds OpenCode integration settings.
type OpenCodeConfig struct {
	BinaryPath string `json:"binary_path"` // path to opencode binary
	APIUrl     string `json:"api_url"`     // for serve mode: http://opencode:4096
	Username   string `json:"username"`
	Password   string `json:"password"`
}


// EmbeddingsConfig holds semantic memory settings.
type EmbeddingsConfig struct {
	Enabled      bool   `json:"enabled"`                // enable semantic memory
	PostgresURL  string `json:"postgres_url,omitempty"`  // postgres://user:pass@host:5432/db
	TEIURL       string `json:"tei_url,omitempty"`       // http://tei-embeddings:80
	SyncInterval string `json:"sync_interval,omitempty"` // e.g. "30s"
	BatchSize    int    `json:"batch_size,omitempty"`    // batch size for embedding (default 50)
}

// DreamConfig holds dream worker (autonomous memory maintenance) settings.
type DreamConfig struct {
	Disabled bool   `json:"disabled,omitempty"` // disable dream worker entirely
	Interval string `json:"interval,omitempty"` // e.g. "6h" (default)
}

// ProjectConfig describes a project Nous is aware of.
type ProjectConfig struct {
	Name  string `json:"name"`            // e.g., "my-project"
	Path  string `json:"path"`            // e.g., "/projects/my-project"
	Scope string `json:"scope"`           // brain scope for this project
	URL   string `json:"url,omitempty"`   // OpenCode serve API URL (for dispatch)
}

// LoadConfig reads config from a file path or environment.
// If path is empty, uses defaults suitable for container deployment.
func LoadConfig(path string) (*Config, error) {
	if path == "" {
		return defaultConfig(), nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}

	// Resolve env var references in all $-prefixed values
	cfg.Matrix.Homeserver = resolveEnv(cfg.Matrix.Homeserver)
	cfg.Matrix.UserID = resolveEnv(cfg.Matrix.UserID)
	cfg.Matrix.Password = resolveEnv(cfg.Matrix.Password)
	cfg.Matrix.ServerName = resolveEnv(cfg.Matrix.ServerName)
	cfg.LLM.Deep.APIKey = resolveEnv(cfg.LLM.Deep.APIKey)
	cfg.LLM.Fast.APIKey = resolveEnv(cfg.LLM.Fast.APIKey)
	cfg.LLM.Mid.APIKey = resolveEnv(cfg.LLM.Mid.APIKey)
	cfg.OpenCode.APIUrl = resolveEnv(cfg.OpenCode.APIUrl)
	cfg.OpenCode.Password = resolveEnv(cfg.OpenCode.Password)
	cfg.AuthJSONPath = resolveEnv(cfg.AuthJSONPath)
	cfg.Embeddings.PostgresURL = resolveEnv(cfg.Embeddings.PostgresURL)
	cfg.Embeddings.TEIURL = resolveEnv(cfg.Embeddings.TEIURL)

	return &cfg, nil
}

// resolveEnv replaces $ENV_VAR references with actual values.
func resolveEnv(s string) string {
	if len(s) > 1 && s[0] == '$' {
		if v := os.Getenv(s[1:]); v != "" {
			return v
		}
	}
	return s
}

// defaultConfig returns a config using environment variables,
// suitable for the existing Docker Compose setup.
func defaultConfig() *Config {
	return &Config{
		Name: "nous",
		Matrix: MatrixConfig{
			Homeserver:   envOr("MATRIX_HOMESERVER", "http://synapse:8008"),
			UserID:       envOr("MATRIX_BOT_USER", "nous"),
			Password:     envOr("MATRIX_BOT_PASSWORD", ""),
			ServerName:   envOr("MATRIX_SERVER_NAME", "matrix.example.com"),
			AllowedUsers: []string{envOr("ALLOWED_USERS", "@admin:matrix.example.com")},
			DataDir:      envOr("NOUS_DATA_DIR", "/data"),
		},
		LLM: LLMConfig{
			Deep: ProviderConfig{
				Provider:      "anthropic",
				Model:         "claude-opus-4-6",
				APIKey:        os.Getenv("ANTHROPIC_API_KEY"),
				ContextWindow: 1_000_000, // 1M context via beta header
				MaxOutput:     16384,     // reasonable chat output limit
				Temperature:   0.7,
			},
			Mid: ProviderConfig{
				Provider:      "copilot",
				Model:         "gpt-5.2",
				ContextWindow: 128_000, // GPT-5 context window
				MaxOutput:     16384,
				Temperature:   0.7,
			},
			Fast: ProviderConfig{
				Provider:      "kimi",
				Model:         "k2p5",
				APIKey:        os.Getenv("KIMI_API_KEY"),
				BaseURL:       "https://api.kimi.com/coding",
				ContextWindow: 256_000, // K2.5 supports 256K
				MaxOutput:     8192,
				Temperature:   0.7,
			},
		},
		AuthJSONPath: envOr("OPENCODE_AUTH_JSON", ""),
		OpenCode: OpenCodeConfig{
			APIUrl:   envOr("OPENCODE_API_URL", "http://opencode:4096"),
			Username: envOr("OPENCODE_SERVER_USERNAME", "opencode"),
			Password: envOr("OPENCODE_SERVER_PASSWORD", ""),
		},
		Embeddings: EmbeddingsConfig{
			Enabled:      envOr("NOUS_EMBEDDINGS_ENABLED", "") != "",
			PostgresURL:  envOr("NOUS_PG_URL", ""),
			TEIURL:       envOr("NOUS_TEI_URL", ""),
			SyncInterval: envOr("NOUS_EMBED_SYNC_INTERVAL", "30s"),
			BatchSize:    32,
		},
		Workspace: WorkspaceConfig{
			Enabled:    envOr("NOUS_WORKSPACE_ENABLED", "1") != "",
			SocketPath: envOr("NOUS_WORKSPACE_SOCKET", "/tmp/nous.sock"),
		},
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
