package daemon

import (
	"encoding/json"
	"fmt"
	"os"
)

type Config struct {
	Name       string                     `json:"name"`
	BrainPath  string                     `json:"brain_path,omitempty"`
	HTTPAddr   string                     `json:"http_addr,omitempty"`
	Workspace  WorkspaceConfig            `json:"workspace,omitempty"`
	Embeddings EmbeddingsConfig           `json:"embeddings,omitempty"`
	Dream      DreamConfig                `json:"dream,omitempty"`
	Modules    map[string]json.RawMessage `json:"modules,omitempty"`
}

type WorkspaceConfig struct {
	Enabled    bool   `json:"enabled"`
	SocketPath string `json:"socket_path,omitempty"`
	TCPAddr    string `json:"tcp_addr,omitempty"`
}

type EmbeddingsConfig struct {
	Enabled      bool   `json:"enabled"`
	PostgresURL  string `json:"postgres_url,omitempty"`
	TEIURL       string `json:"tei_url,omitempty"`
	SyncInterval string `json:"sync_interval,omitempty"`
	BatchSize    int    `json:"batch_size,omitempty"`
}

type DreamConfig struct {
	Disabled bool   `json:"disabled,omitempty"`
	Interval string `json:"interval,omitempty"`
}

func LoadConfig(path string) (*Config, error) {
	base := defaultConfig()
	baseJSON, err := json.Marshal(base)
	if err != nil {
		return nil, fmt.Errorf("marshal default config: %w", err)
	}

	merged := baseJSON
	if path != "" {
		fileData, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read config %s: %w", path, err)
		}
		merged, err = deepMergeJSON(merged, fileData)
		if err != nil {
			return nil, fmt.Errorf("merge config %s: %w", path, err)
		}
	}

	if overlay := os.Getenv("NOUS_PRIVATE_CONFIG"); overlay != "" {
		overlayData, err := os.ReadFile(overlay)
		if err != nil {
			return nil, fmt.Errorf("read private config %s: %w", overlay, err)
		}
		merged, err = deepMergeJSON(merged, overlayData)
		if err != nil {
			return nil, fmt.Errorf("merge private config %s: %w", overlay, err)
		}
	}

	var cfg Config
	if err := json.Unmarshal(merged, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	cfg.Name = resolveEnv(cfg.Name)
	cfg.BrainPath = resolveEnv(cfg.BrainPath)
	cfg.HTTPAddr = resolveEnv(cfg.HTTPAddr)
	cfg.Workspace.SocketPath = resolveEnv(cfg.Workspace.SocketPath)
	cfg.Workspace.TCPAddr = resolveEnv(cfg.Workspace.TCPAddr)
	cfg.Embeddings.PostgresURL = resolveEnv(cfg.Embeddings.PostgresURL)
	cfg.Embeddings.TEIURL = resolveEnv(cfg.Embeddings.TEIURL)

	if cfg.Name == "" {
		cfg.Name = "nous"
	}
	if cfg.HTTPAddr == "" {
		cfg.HTTPAddr = ":8080"
	}
	if cfg.Modules == nil {
		cfg.Modules = map[string]json.RawMessage{}
	}

	return &cfg, nil
}

func deepMergeJSON(base, overlay []byte) ([]byte, error) {
	var baseMap map[string]interface{}
	if len(base) > 0 {
		if err := json.Unmarshal(base, &baseMap); err != nil {
			return nil, err
		}
	}
	if baseMap == nil {
		baseMap = map[string]interface{}{}
	}

	var overlayMap map[string]interface{}
	if len(overlay) > 0 {
		if err := json.Unmarshal(overlay, &overlayMap); err != nil {
			return nil, err
		}
	}
	mergeMap(baseMap, overlayMap)
	return json.Marshal(baseMap)
}

func mergeMap(dst, src map[string]interface{}) {
	for k, v := range src {
		dstObj, dstIsObj := dst[k].(map[string]interface{})
		srcObj, srcIsObj := v.(map[string]interface{})
		if dstIsObj && srcIsObj {
			mergeMap(dstObj, srcObj)
			dst[k] = dstObj
			continue
		}
		dst[k] = v
	}
}

func resolveEnv(s string) string {
	if len(s) > 1 && s[0] == '$' {
		if v := os.Getenv(s[1:]); v != "" {
			return v
		}
	}
	return s
}

func defaultConfig() *Config {
	return &Config{
		Name:      "nous",
		BrainPath: envOr("NOUS_BRAIN_PATH", "brain"),
		HTTPAddr:  envOr("NOUS_HTTP_ADDR", ":8080"),
		Workspace: WorkspaceConfig{
			Enabled:    envOr("NOUS_WORKSPACE_ENABLED", "1") != "",
			SocketPath: envOr("NOUS_WORKSPACE_SOCKET", "/tmp/nous.sock"),
			TCPAddr:    envOr("NOUS_WORKSPACE_TCP", ":8090"),
		},
		Embeddings: EmbeddingsConfig{
			Enabled:      envOr("NOUS_EMBEDDINGS_ENABLED", "") != "",
			PostgresURL:  envOr("NOUS_PG_URL", ""),
			TEIURL:       envOr("NOUS_TEI_URL", ""),
			SyncInterval: envOr("NOUS_EMBED_SYNC_INTERVAL", "30s"),
			BatchSize:    32,
		},
		Dream: DreamConfig{
			Disabled: envOr("NOUS_DREAM_DISABLED", "") != "",
			Interval: envOr("NOUS_DREAM_INTERVAL", "6h"),
		},
		Modules: map[string]json.RawMessage{},
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
