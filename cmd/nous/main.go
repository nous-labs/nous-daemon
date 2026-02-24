package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/nous-labs/nous/internal/daemon"
	"github.com/nous-labs/nous/pkg/brain"
	"log/slog"
	_ "modernc.org/sqlite"
	"os"
	"os/signal"
	"syscall"
)

var (
	version = "dev"
	commit  = "unknown"
)

func main() {
	// Flags
	brainPath := flag.String("brain", "", "Path to brain directory (contains state.db, knowledge/)")
	configPath := flag.String("config", "", "Path to config file")
	showVersion := flag.Bool("version", false, "Show version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Printf("nous %s (%s)\n", version, commit)
		os.Exit(0)
	}

	// Logger
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	// Resolve brain path
	bp := *brainPath
	if bp == "" {
		bp = os.Getenv("NOUS_BRAIN_PATH")
	}
	if bp == "" {
		bp = "brain" // default: relative to cwd
	}

	// Open brain
	b, err := brain.Open(bp)
	if err != nil {
		slog.Error("failed to open brain", "path", bp, "error", err)
		os.Exit(1)
	}
	defer b.Close()

	slog.Info("nous starting",
		"version", version,
		"brain", bp,
		"memories", b.Stats().Memories,
	)

	// Load config
	cp := *configPath
	if cp == "" {
		cp = os.Getenv("NOUS_CONFIG_PATH")
	}

	cfg, err := daemon.LoadConfig(cp)
	if err != nil {
		slog.Error("failed to load config", "path", cp, "error", err)
		os.Exit(1)
	}

	// Create and start daemon
	d, err := daemon.New(b, cfg)
	if err != nil {
		slog.Error("failed to create daemon", "error", err)
		os.Exit(1)
	}

	// Graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigCh
		slog.Info("received signal, shutting down", "signal", sig)
		cancel()
	}()

	if err := d.Run(ctx); err != nil && ctx.Err() == nil {
		slog.Error("daemon error", "error", err)
		os.Exit(1)
	}

	slog.Info("nous stopped")
}
