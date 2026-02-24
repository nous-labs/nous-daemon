package brain

import (
	"os"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"
)

// testBrainPath returns the path to the real brain, or skips the test.
func testBrainPath(t *testing.T) string {
	// Check NOUS_BRAIN_PATH env var first
	if p := os.Getenv("NOUS_BRAIN_PATH"); p != "" {
		return p
	}
	// Check common locations
	for _, p := range []string{"/brain", "../../brain", "../../../brain"} {
		if _, err := os.Stat(filepath.Join(p, "state.db")); err == nil {
			return p
		}
	}
	t.Skip("brain/state.db not found — set NOUS_BRAIN_PATH or run from project root")
	return ""
}

func TestOpen(t *testing.T) {
	path := testBrainPath(t)
	b, err := Open(path)
	if err != nil {
		t.Fatalf("Open(%q): %v", path, err)
	}
	defer b.Close()

	if b.Path() != path {
		t.Errorf("Path() = %q, want %q", b.Path(), path)
	}
}

func TestStats(t *testing.T) {
	path := testBrainPath(t)
	b, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer b.Close()

	s := b.Stats()
	if s.Memories == 0 {
		t.Error("Stats.Memories = 0, expected > 0")
	}
	if s.Tasks == 0 {
		t.Error("Stats.Tasks = 0, expected > 0")
	}
	t.Logf("Stats: memories=%d tasks=%d entities=%d kv=%d sessions=%d",
		s.Memories, s.Tasks, s.Entities, s.KVEntries, s.SessionRefs)
}

func TestRecall(t *testing.T) {
	path := testBrainPath(t)
	b, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer b.Close()

	// Recall all (no filter)
	memories, err := b.Recall("", RecallOptions{Limit: 5})
	if err != nil {
		t.Fatalf("Recall (all): %v", err)
	}
	if len(memories) == 0 {
		t.Fatal("Recall (all): got 0 memories")
	}
	t.Logf("Recall (all): got %d memories", len(memories))
	for _, m := range memories {
		t.Logf("  [%d] %s/%s: %.80s", m.ID, m.Type, m.Scope, m.Content)
	}

	// Recall by type
	memories, err = b.Recall("", RecallOptions{Type: "decision", Limit: 3})
	if err != nil {
		t.Fatalf("Recall (type=decision): %v", err)
	}
	for _, m := range memories {
		if m.Type != "decision" {
			t.Errorf("expected type=decision, got %q", m.Type)
		}
	}
	t.Logf("Recall (type=decision): got %d memories", len(memories))

	// Recall by scope
	memories, err = b.Recall("", RecallOptions{Scope: "omo-config", Limit: 3})
	if err != nil {
		t.Fatalf("Recall (scope=omo-config): %v", err)
	}
	for _, m := range memories {
		if m.Scope != "omo-config" {
			t.Errorf("expected scope=omo-config, got %q", m.Scope)
		}
	}
	t.Logf("Recall (scope=omo-config): got %d memories", len(memories))
}

func TestBootstrap(t *testing.T) {
	path := testBrainPath(t)
	b, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer b.Close()

	result, err := b.Bootstrap("omo-config")
	if err != nil {
		t.Fatalf("Bootstrap: %v", err)
	}
	if len(result) == 0 {
		t.Fatal("Bootstrap returned empty string")
	}
	t.Logf("Bootstrap output (%d chars):\n%s", len(result), result)
}

func TestCapture(t *testing.T) {
	path := testBrainPath(t)
	b, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer b.Close()

	// Capture a test memory
	id, err := b.Capture("observation", "test", "brain_test.go integration test", "test,integration", "brain_test")
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}
	if id == 0 {
		t.Error("Capture returned id=0")
	}
	t.Logf("Captured memory id=%d", id)

	// Verify we can recall it
	memories, err := b.Recall("", RecallOptions{Scope: "test", Limit: 1})
	if err != nil {
		t.Fatalf("Recall after capture: %v", err)
	}
	if len(memories) == 0 {
		t.Fatal("Recall after capture: got 0 memories")
	}
	if memories[0].Content != "brain_test.go integration test" {
		t.Errorf("unexpected content: %q", memories[0].Content)
	}

	// Clean up — soft delete
	_, err = b.db.Exec("UPDATE memories SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?", id)
	if err != nil {
		t.Fatalf("cleanup: %v", err)
	}
}

func TestKV(t *testing.T) {
	path := testBrainPath(t)
	b, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer b.Close()

	// Set a test key
	err = b.KVSet("test:brain_test", "hello from Go")
	if err != nil {
		t.Fatalf("KVSet: %v", err)
	}

	// Get it back
	val, err := b.KVGet("test:brain_test")
	if err != nil {
		t.Fatalf("KVGet: %v", err)
	}
	if val != "hello from Go" {
		t.Errorf("KVGet = %q, want %q", val, "hello from Go")
	}

	// Clean up
	_, err = b.db.Exec("DELETE FROM kv WHERE key = ?", "test:brain_test")
	if err != nil {
		t.Fatalf("cleanup: %v", err)
	}
}
