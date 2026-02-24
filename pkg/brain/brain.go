// Package brain provides access to Nous's persistent memory system.
//
// It reads and writes the same SQLite database (state.db) used by the
// Python omo-memory CLI, ensuring full backward compatibility.
// The brain is Nous's identity — memories, knowledge, tasks, entities
// all survive across sessions and runtimes.
package brain

import (
	"crypto/md5"
	"database/sql"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Brain provides access to Nous's persistent memory and knowledge.
type Brain struct {
	db   *sql.DB
	path string // root brain directory
}

// Stats holds brain statistics.
type Stats struct {
	Memories    int
	Tasks       int
	Entities    int
	KVEntries   int
	SessionRefs int
}

// Memory represents a single memory entry.
type Memory struct {
	ID              int
	Type            string // decision, preference, observation, failure, pattern, fact
	Scope           string // global, project name
	Content         string
	Tags            string
	Source          string
	CreatedAt       time.Time
	UpdatedAt       time.Time
	VerifiedAt      *time.Time
	TTLDays         *int
	StalenessPolicy *string
	ExpiresAt       *time.Time
	SupersededBy    *int
	TopicKey        *string
	DeletedAt       *time.Time
	RevisionCount   int
	DuplicateCount  int
	ValidAt         *time.Time
	InvalidAt       *time.Time
}

// Task represents a brain task.
type Task struct {
	ID          int
	Title       string
	Description string
	Status      string // pending, in_progress, completed, cancelled
	Priority    string // low, medium, high
	DueDate     *string
	RepeatRule  *string
	EntityID    *int
	Tags        string
	CreatedAt   time.Time
	CompletedAt *time.Time
}

// Episode represents a timeline episode (session of work).
type Episode struct {
	ID              string
	Scope           string
	StartedAt       time.Time
	EndedAt         *time.Time
	Intent          string
	Summary         string
	Status          string
	ParentEpisodeID *string
}

// AccessStats holds frequency data for a memory.
type AccessStats struct {
	MemoryID       int
	AccessCount    int
	LastAccessedAt *time.Time
	AccessScore    float64
}

// Entity represents a tracked entity (project, person, service, etc).
type Entity struct {
	ID        int
	Type      string
	Name      string
	Metadata  string // JSON
	CreatedAt time.Time
	UpdatedAt time.Time
}

// Open opens an existing brain at the given directory path.
// The directory must contain state.db.
func Open(path string) (*Brain, error) {
	dbPath := filepath.Join(path, "state.db")
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("brain database not found at %s", dbPath)
	}

	// Open with WAL mode for concurrent reads, foreign keys enabled
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=foreign_keys(1)&_pragma=busy_timeout(5000)", dbPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open brain db: %w", err)
	}

	// Verify connection
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping brain db: %w", err)
	}

	b := &Brain{db: db, path: path}

	stats := b.Stats()
	slog.Info("brain opened",
		"path", path,
		"memories", stats.Memories,
		"tasks", stats.Tasks,
		"entities", stats.Entities,
	)

	return b, nil
}

// Close closes the brain database.
func (b *Brain) Close() error {
	return b.db.Close()
}

// Path returns the brain root directory.
func (b *Brain) Path() string {
	return b.path
}

// Stats returns counts for all brain tables.
func (b *Brain) Stats() Stats {
	var s Stats
	b.db.QueryRow("SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL").Scan(&s.Memories)
	b.db.QueryRow("SELECT COUNT(*) FROM tasks").Scan(&s.Tasks)
	b.db.QueryRow("SELECT COUNT(*) FROM entities").Scan(&s.Entities)
	b.db.QueryRow("SELECT COUNT(*) FROM kv").Scan(&s.KVEntries)
	b.db.QueryRow("SELECT COUNT(*) FROM session_refs").Scan(&s.SessionRefs)
	return s
}

// --- Memory Operations ---

// Capture stores a new memory. This is the primary write operation.
func (b *Brain) Capture(typ, scope, content, tags, source string) (int64, error) {
	now := time.Now().UTC().Format("2006-01-02 15:04:05")

	result, err := b.db.Exec(
		`INSERT INTO memories (type, scope, content, tags, source, created_at, updated_at, valid_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		typ, scope, content, tags, source, now, now, now,
	)
	if err != nil {
		return 0, fmt.Errorf("capture memory: %w", err)
	}

	id, _ := result.LastInsertId()
	slog.Debug("memory captured", "id", id, "type", typ, "scope", scope)
	return id, nil
}

// Recall retrieves memories matching a keyword search using FTS5.
func (b *Brain) Recall(query string, opts RecallOptions) ([]Memory, error) {
	var conditions []string
	var args []interface{}

	// Base query — join with FTS for search
	baseQuery := `SELECT m.id, m.type, m.scope, m.content, m.tags, m.source,
		m.created_at, m.updated_at, m.verified_at, m.ttl_days, m.staleness_policy,
		m.revision_count, m.duplicate_count
		FROM memories m`

	if query != "" {
		baseQuery += ` JOIN memories_fts fts ON m.id = fts.rowid`
		conditions = append(conditions, `memories_fts MATCH ?`)
		args = append(args, query)
	}

	// Always exclude soft-deleted
	conditions = append(conditions, `m.deleted_at IS NULL`)

	if opts.Type != "" {
		conditions = append(conditions, `m.type = ?`)
		args = append(args, opts.Type)
	}
	if opts.Scope != "" {
		conditions = append(conditions, `m.scope = ?`)
		args = append(args, opts.Scope)
	}
	if opts.Tags != "" {
		// Tag matching: any of the provided tags
		for _, tag := range strings.Split(opts.Tags, ",") {
			tag = strings.TrimSpace(tag)
			if tag != "" {
				conditions = append(conditions, `m.tags LIKE ?`)
				args = append(args, "%"+tag+"%")
			}
		}
	}

	if len(conditions) > 0 {
		baseQuery += " WHERE " + strings.Join(conditions, " AND ")
	}

	baseQuery += " ORDER BY m.id DESC"

	limit := opts.Limit
	if limit <= 0 {
		limit = 20
	}
	baseQuery += fmt.Sprintf(" LIMIT %d", limit)

	rows, err := b.db.Query(baseQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("recall memories: %w", err)
	}
	defer rows.Close()

	var memories []Memory
	for rows.Next() {
		var m Memory
		var createdAt, updatedAt, verifiedAt sql.NullString
		var tags, source, scope, stalenessPolicy sql.NullString
		var ttlDays sql.NullInt64
		var revisionCount, duplicateCount sql.NullInt64
		err := rows.Scan(
			&m.ID, &m.Type, &scope, &m.Content, &tags, &source,
			&createdAt, &updatedAt, &verifiedAt, &ttlDays, &stalenessPolicy,
			&revisionCount, &duplicateCount,
		)
		if err != nil {
			return nil, fmt.Errorf("scan memory: %w", err)
		}
		m.Scope = scope.String
		m.Tags = tags.String
		m.Source = source.String
		m.RevisionCount = int(revisionCount.Int64)
		m.DuplicateCount = int(duplicateCount.Int64)
		if createdAt.Valid {
			m.CreatedAt = parseTime(createdAt.String)
		}
		if updatedAt.Valid {
			m.UpdatedAt = parseTime(updatedAt.String)
		}
		if verifiedAt.Valid {
			t := parseTime(verifiedAt.String)
			m.VerifiedAt = &t
		}
		if ttlDays.Valid {
			ttl := int(ttlDays.Int64)
			m.TTLDays = &ttl
		}
		if stalenessPolicy.Valid {
			p := stalenessPolicy.String
			m.StalenessPolicy = &p
		}
		memories = append(memories, m)
	}

	return memories, rows.Err()
}

// RecallOptions controls memory retrieval filtering.
type RecallOptions struct {
	Type  string
	Scope string
	Tags  string
	Limit int
}

// Bootstrap returns a compact context string suitable for session initialization.
// This replicates the Python `omo-memory bootstrap` command output.
func (b *Brain) Bootstrap(scope string) (string, error) {
	var parts []string

	// 1. Hard constraints (always)
	constraints, err := b.Recall("", RecallOptions{
		Type:  "preference",
		Tags:  "hard-constraint",
		Limit: 20,
	})
	if err != nil {
		return "", fmt.Errorf("bootstrap constraints: %w", err)
	}
	if len(constraints) > 0 {
		parts = append(parts, "## Constraints")
		for _, m := range constraints {
			summary := m.Content
			if len(summary) > 120 {
				summary = summary[:120] + "..."
			}
			parts = append(parts, fmt.Sprintf("- [%d] %s", m.ID, summary))
		}
	}

	// 2. Recent decisions/observations for scope
	recent, err := b.Recall("", RecallOptions{
		Scope: scope,
		Limit: 5,
	})
	if err != nil {
		return "", fmt.Errorf("bootstrap recent: %w", err)
	}
	if len(recent) > 0 {
		parts = append(parts, "\n## Recent")
		for _, m := range recent {
			summary := m.Content
			if len(summary) > 120 {
				summary = summary[:120] + "..."
			}
			ago := formatTimeAgo(m.CreatedAt)
			parts = append(parts, fmt.Sprintf("- [%d] %s: %s (%s)", m.ID, m.Type, summary, ago))
		}
	}

	// 3. Recent failures and patterns
	patterns, err := b.Recall("", RecallOptions{
		Type:  "failure",
		Scope: scope,
		Limit: 3,
	})
	if err != nil {
		return "", fmt.Errorf("bootstrap patterns: %w", err)
	}
	patternsP, err := b.Recall("", RecallOptions{
		Type:  "pattern",
		Scope: scope,
		Limit: 3,
	})
	if err != nil {
		return "", fmt.Errorf("bootstrap patterns: %w", err)
	}
	patterns = append(patterns, patternsP...)

	if len(patterns) > 0 {
		parts = append(parts, "\n## Patterns")
		for _, m := range patterns {
			summary := m.Content
			if len(summary) > 100 {
				summary = summary[:100] + "..."
			}
			symbol := "✗"
			if m.Type == "pattern" {
				symbol = "✓"
			}
			parts = append(parts, fmt.Sprintf("- %s [%d] %s", symbol, m.ID, summary))
		}
	}

	return strings.Join(parts, "\n"), nil
}

// --- Knowledge Operations ---

// ReadKnowledge reads a markdown file from the knowledge directory.
func (b *Brain) ReadKnowledge(filename string) (string, error) {
	path := filepath.Join(b.path, "knowledge", filename)
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read knowledge %s: %w", filename, err)
	}
	return string(data), nil
}

// ListKnowledge returns all knowledge files.
func (b *Brain) ListKnowledge() ([]string, error) {
	dir := filepath.Join(b.path, "knowledge")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("list knowledge: %w", err)
	}
	var files []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".md") {
			files = append(files, e.Name())
		}
	}
	return files, nil
}

// --- KV Operations ---

// KVGet retrieves a value from the key-value store.
func (b *Brain) KVGet(key string) (string, error) {
	var value string
	err := b.db.QueryRow("SELECT value FROM kv WHERE key = ?", key).Scan(&value)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return value, err
}

// KVSet stores a value in the key-value store.
func (b *Brain) KVSet(key, value string) error {
	now := time.Now().UTC().Format("2006-01-02 15:04:05")
	_, err := b.db.Exec(
		`INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?)
		 ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`,
		key, value, now,
	)
	return err
}

// --- Embedding Support ---

// MemoryRef is a lightweight reference to a memory for embedding sync.
type MemoryRef struct {
	ID          int
	ContentHash string // MD5 of content for staleness detection
}

// GetAllMemoryRefs returns all active memory IDs with content hashes.
// Used by the embedding sync worker to detect un-embedded or stale memories.
func (b *Brain) GetAllMemoryRefs() ([]MemoryRef, error) {
	rows, err := b.db.Query(`
		SELECT id, content FROM memories WHERE deleted_at IS NULL
	`)
	if err != nil {
		return nil, fmt.Errorf("get memory refs: %w", err)
	}
	defer rows.Close()

	var refs []MemoryRef
	for rows.Next() {
		var id int
		var content string
		if err := rows.Scan(&id, &content); err != nil {
			return nil, fmt.Errorf("scan memory ref: %w", err)
		}
		refs = append(refs, MemoryRef{
			ID:          id,
			ContentHash: fmt.Sprintf("%x", md5Hash([]byte(content))),
		})
	}
	return refs, rows.Err()
}

// GetMemoriesByIDs fetches full memories for a list of IDs.
// Returns memories in the order they were found (not necessarily input order).
func (b *Brain) GetMemoriesByIDs(ids []int64) ([]Memory, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	// Build placeholder string: ?,?,?,...
	placeholders := make([]byte, 0, len(ids)*2)
	args := make([]interface{}, len(ids))
	for i, id := range ids {
		if i > 0 {
			placeholders = append(placeholders, ',')
		}
		placeholders = append(placeholders, '?')
		args[i] = id
	}

	query := fmt.Sprintf(`SELECT id, type, scope, content, tags, source,
		created_at, updated_at, verified_at, ttl_days, staleness_policy,
		revision_count, duplicate_count
		FROM memories WHERE id IN (%s) AND deleted_at IS NULL`, string(placeholders))

	rows, err := b.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("get memories by ids: %w", err)
	}
	defer rows.Close()

	var memories []Memory
	for rows.Next() {
		var m Memory
		var createdAt, updatedAt, verifiedAt sql.NullString
		var tags, source, scope, stalenessPolicy sql.NullString
		var ttlDays sql.NullInt64
		var revisionCount, duplicateCount sql.NullInt64
		err := rows.Scan(
			&m.ID, &m.Type, &scope, &m.Content, &tags, &source,
			&createdAt, &updatedAt, &verifiedAt, &ttlDays, &stalenessPolicy,
			&revisionCount, &duplicateCount,
		)
		if err != nil {
			return nil, fmt.Errorf("scan memory: %w", err)
		}
		m.Scope = scope.String
		m.Tags = tags.String
		m.Source = source.String
		m.RevisionCount = int(revisionCount.Int64)
		m.DuplicateCount = int(duplicateCount.Int64)
		if createdAt.Valid {
			m.CreatedAt = parseTime(createdAt.String)
		}
		if updatedAt.Valid {
			m.UpdatedAt = parseTime(updatedAt.String)
		}
		if verifiedAt.Valid {
			t := parseTime(verifiedAt.String)
			m.VerifiedAt = &t
		}
		if ttlDays.Valid {
			ttl := int(ttlDays.Int64)
			m.TTLDays = &ttl
		}
		if stalenessPolicy.Valid {
			p := stalenessPolicy.String
			m.StalenessPolicy = &p
		}
		memories = append(memories, m)
	}
	return memories, rows.Err()
}

// ActiveEpisode returns the currently active episode for a scope, or nil.
func (b *Brain) ActiveEpisode(scope string) (*Episode, error) {
	var e Episode
	var endedAt, intent, summary, parentEpisodeID sql.NullString
	var startedAt sql.NullString

	err := b.db.QueryRow(`
		SELECT id, scope, started_at, ended_at, intent, summary, status, parent_episode_id
		FROM episodes
		WHERE scope = ? AND status = 'active'
		ORDER BY started_at DESC
		LIMIT 1
	`, scope).Scan(
		&e.ID, &e.Scope, &startedAt, &endedAt, &intent, &summary, &e.Status, &parentEpisodeID,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get active episode: %w", err)
	}

	if startedAt.Valid {
		e.StartedAt = parseTime(startedAt.String)
	}
	if endedAt.Valid {
		t := parseTime(endedAt.String)
		e.EndedAt = &t
	}
	if intent.Valid {
		e.Intent = intent.String
	}
	if summary.Valid {
		e.Summary = summary.String
	}
	if parentEpisodeID.Valid {
		p := parentEpisodeID.String
		e.ParentEpisodeID = &p
	}

	return &e, nil
}

// GetAccessStats returns access stats for a list of memory IDs.
// Returns a map of memory_id -> AccessStats.
func (b *Brain) GetAccessStats(ids []int) (map[int]AccessStats, error) {
	stats := make(map[int]AccessStats, len(ids))
	if len(ids) == 0 {
		return stats, nil
	}

	placeholders := make([]byte, 0, len(ids)*2)
	args := make([]interface{}, len(ids))
	for i, id := range ids {
		if i > 0 {
			placeholders = append(placeholders, ',')
		}
		placeholders = append(placeholders, '?')
		args[i] = id
		stats[id] = AccessStats{MemoryID: id}
	}

	query := fmt.Sprintf(`
		SELECT memory_id, access_count, last_accessed_at, access_score
		FROM memory_access_stats
		WHERE memory_id IN (%s)
	`, string(placeholders))

	rows, err := b.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("get access stats: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var s AccessStats
		var lastAccessedAt sql.NullString
		if err := rows.Scan(&s.MemoryID, &s.AccessCount, &lastAccessedAt, &s.AccessScore); err != nil {
			return nil, fmt.Errorf("scan access stats: %w", err)
		}
		if lastAccessedAt.Valid {
			t := parseTime(lastAccessedAt.String)
			s.LastAccessedAt = &t
		}
		stats[s.MemoryID] = s
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	return stats, nil
}

// RecordAccess logs an access event and updates stats with decayed scoring.
// tau = 20 days in seconds (1728000). Formula: new_score = old * exp(-dt/tau) + 1.0
func (b *Brain) RecordAccess(memoryID int, episodeID string, accessKind string, query string) error {
	now := time.Now().UTC()
	nowStr := now.Format("2006-01-02 15:04:05")

	tx, err := b.db.Begin()
	if err != nil {
		return fmt.Errorf("begin record access tx: %w", err)
	}
	defer tx.Rollback()

	var episodeArg interface{}
	if episodeID != "" {
		episodeArg = episodeID
	}

	if _, err := tx.Exec(`
		INSERT INTO memory_access_events(memory_id, episode_id, accessed_at, access_kind, query)
		VALUES (?, ?, ?, ?, ?)
	`, memoryID, episodeArg, nowStr, accessKind, query); err != nil {
		return fmt.Errorf("insert access event: %w", err)
	}

	var oldScore float64
	var lastAccessedAt sql.NullString
	err = tx.QueryRow(`
		SELECT access_score, last_accessed_at
		FROM memory_access_stats
		WHERE memory_id = ?
	`, memoryID).Scan(&oldScore, &lastAccessedAt)
	if err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("query existing access stats: %w", err)
	}

	deltaSeconds := 0.0
	if err == nil && lastAccessedAt.Valid {
		last := parseTime(lastAccessedAt.String)
		if !last.IsZero() {
			deltaSeconds = now.Sub(last).Seconds()
			if deltaSeconds < 0 {
				deltaSeconds = 0
			}
		}
	}

	newScore := oldScore*math.Exp(-deltaSeconds/1728000.0) + 1.0

	if _, err := tx.Exec(`
		INSERT INTO memory_access_stats(memory_id, access_count, last_accessed_at, access_score)
		VALUES (?, 1, ?, ?)
		ON CONFLICT(memory_id) DO UPDATE SET
			access_count = memory_access_stats.access_count + 1,
			last_accessed_at = excluded.last_accessed_at,
			access_score = excluded.access_score
	`, memoryID, nowStr, newScore); err != nil {
		return fmt.Errorf("upsert access stats: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit record access tx: %w", err)
	}

	return nil
}

// ListTasks returns all non-completed tasks, ordered by priority then creation.
func (b *Brain) ListTasks() ([]Task, error) {
	rows, err := b.db.Query(`
		SELECT id, title, COALESCE(description,''), status, COALESCE(priority,'medium'),
		       due_date, repeat_rule, entity_id, COALESCE(tags,''), created_at, completed_at
		FROM tasks
		WHERE status NOT IN ('completed', 'cancelled')
		ORDER BY
		  CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
		  created_at ASC
	`)
	if err != nil {
		return nil, fmt.Errorf("list tasks: %w", err)
	}
	defer rows.Close()

	var tasks []Task
	for rows.Next() {
		var t Task
		var dueDate, repeatRule sql.NullString
		var entityID sql.NullInt64
		var createdAt, completedAt sql.NullString
		if err := rows.Scan(&t.ID, &t.Title, &t.Description, &t.Status, &t.Priority,
			&dueDate, &repeatRule, &entityID, &t.Tags, &createdAt, &completedAt); err != nil {
			return nil, fmt.Errorf("scan task: %w", err)
		}
		if dueDate.Valid {
			s := dueDate.String
			t.DueDate = &s
		}
		if repeatRule.Valid {
			s := repeatRule.String
			t.RepeatRule = &s
		}
		if entityID.Valid {
			i := int(entityID.Int64)
			t.EntityID = &i
		}
		if createdAt.Valid {
			t.CreatedAt = parseTime(createdAt.String)
		}
		if completedAt.Valid {
			ct := parseTime(completedAt.String)
			t.CompletedAt = &ct
		}
		tasks = append(tasks, t)
	}
	return tasks, rows.Err()
}

// --- Dream Operations (autonomous memory maintenance) ---

// StaleMemory is a memory with a computed staleness score.
type StaleMemory struct {
	Memory
	StalenessScore    float64
	DaysSinceVerified float64
}

// FindStaleMemories returns memories whose staleness score exceeds the threshold.
// Staleness is computed based on each memory's staleness_policy:
//   - none: always 0 (never stale)
//   - ttl: days_since_verified / ttl_days, clamped to 1.0
//   - half_life: 1 - exp(-days * ln2 / ttl_days)
func (b *Brain) FindStaleMemories(threshold float64) ([]StaleMemory, error) {
	rows, err := b.db.Query(`
		SELECT id, type, scope, content, tags, source, created_at, updated_at,
		       verified_at, ttl_days, staleness_policy
		FROM memories
		WHERE deleted_at IS NULL
		  AND staleness_policy IS NOT NULL
		  AND staleness_policy != 'none'
		  AND ttl_days IS NOT NULL
		ORDER BY updated_at ASC
	`)
	if err != nil {
		return nil, fmt.Errorf("find stale memories: %w", err)
	}
	defer rows.Close()

	now := time.Now().UTC()
	var results []StaleMemory

	for rows.Next() {
		var m Memory
		var createdAt, updatedAt, verifiedAt sql.NullString
		var tags, source, scope, policy sql.NullString
		var ttlDays sql.NullInt64
		err := rows.Scan(
			&m.ID, &m.Type, &scope, &m.Content, &tags, &source,
			&createdAt, &updatedAt, &verifiedAt, &ttlDays, &policy,
		)
		if err != nil {
			return nil, fmt.Errorf("scan stale memory: %w", err)
		}
		m.Scope = scope.String
		m.Tags = tags.String
		m.Source = source.String
		if createdAt.Valid {
			m.CreatedAt = parseTime(createdAt.String)
		}
		if updatedAt.Valid {
			m.UpdatedAt = parseTime(updatedAt.String)
		}
		if verifiedAt.Valid {
			t := parseTime(verifiedAt.String)
			m.VerifiedAt = &t
		}
		if ttlDays.Valid {
			ttl := int(ttlDays.Int64)
			m.TTLDays = &ttl
		}
		if policy.Valid {
			p := policy.String
			m.StalenessPolicy = &p
		}

		// Compute staleness score
		referenceTime := m.UpdatedAt
		if m.VerifiedAt != nil {
			referenceTime = *m.VerifiedAt
		}
		daysSince := now.Sub(referenceTime).Hours() / 24.0
		if daysSince < 0 {
			daysSince = 0
		}

		var score float64
		if m.TTLDays != nil && *m.TTLDays > 0 && m.StalenessPolicy != nil {
			switch *m.StalenessPolicy {
			case "ttl":
				score = daysSince / float64(*m.TTLDays)
				if score > 1.0 {
					score = 1.0
				}
			case "half_life":
				score = 1.0 - math.Exp(-daysSince*math.Ln2/float64(*m.TTLDays))
			}
		}

		if score >= threshold {
			results = append(results, StaleMemory{
				Memory:            m,
				StalenessScore:    score,
				DaysSinceVerified: daysSince,
			})
		}
	}

	return results, rows.Err()
}

// FindOrphanedEpisodes returns active episodes with no activity for the given duration.
func (b *Brain) FindOrphanedEpisodes(staleAfter time.Duration) ([]Episode, error) {
	cutoff := time.Now().UTC().Add(-staleAfter).Format("2006-01-02 15:04:05")
	rows, err := b.db.Query(`
		SELECT id, scope, started_at, ended_at, intent, summary, status, parent_episode_id
		FROM episodes
		WHERE status = 'active'
		  AND started_at < ?
	`, cutoff)
	if err != nil {
		return nil, fmt.Errorf("find orphaned episodes: %w", err)
	}
	defer rows.Close()

	var episodes []Episode
	for rows.Next() {
		var e Episode
		var startedAt, endedAt, intent, summary, parentID sql.NullString
		if err := rows.Scan(&e.ID, &e.Scope, &startedAt, &endedAt, &intent, &summary, &e.Status, &parentID); err != nil {
			return nil, fmt.Errorf("scan orphaned episode: %w", err)
		}
		if startedAt.Valid {
			e.StartedAt = parseTime(startedAt.String)
		}
		if endedAt.Valid {
			t := parseTime(endedAt.String)
			e.EndedAt = &t
		}
		if intent.Valid {
			e.Intent = intent.String
		}
		if summary.Valid {
			e.Summary = summary.String
		}
		if parentID.Valid {
			p := parentID.String
			e.ParentEpisodeID = &p
		}
		episodes = append(episodes, e)
	}
	return episodes, rows.Err()
}

// CloseEpisode marks an episode as done with a summary.
func (b *Brain) CloseEpisode(id string, summary string) error {
	now := time.Now().UTC().Format("2006-01-02 15:04:05")
	_, err := b.db.Exec(`
		UPDATE episodes SET status = 'done', ended_at = ?, summary = ?
		WHERE id = ? AND status = 'active'
	`, now, summary, id)
	if err != nil {
		return fmt.Errorf("close episode %s: %w", id, err)
	}
	return nil
}

// InvalidateMemory performs bi-temporal invalidation: sets invalid_at without deleting.
func (b *Brain) InvalidateMemory(id int, reason string) error {
	now := time.Now().UTC().Format("2006-01-02 15:04:05")
	_, err := b.db.Exec(`
		UPDATE memories SET invalid_at = ?, updated_at = ?
		WHERE id = ? AND invalid_at IS NULL
	`, now, now, id)
	if err != nil {
		return fmt.Errorf("invalidate memory %d: %w", id, err)
	}
	// Log the invalidation as a new memory for audit trail
	b.Capture("observation", "nous-daemon",
		fmt.Sprintf("Dream: invalidated memory [%d] \u2014 %s", id, reason),
		"dream,invalidation", "dream-worker")
	return nil
}

// DecayAllAccessScores recalculates all access scores with exponential decay.
// Returns the number of scores updated.
// Formula: new_score = old_score * exp(-dt/tau) where tau = 20 days in seconds.
func (b *Brain) DecayAllAccessScores() (int, error) {
	now := time.Now().UTC()
	const tau = 1728000.0 // 20 days in seconds

	rows, err := b.db.Query(`
		SELECT memory_id, access_score, last_accessed_at
		FROM memory_access_stats
		WHERE access_score > 0.01
	`)
	if err != nil {
		return 0, fmt.Errorf("query access stats for decay: %w", err)
	}
	defer rows.Close()

	type scoreUpdate struct {
		MemoryID int
		NewScore float64
	}
	var updates []scoreUpdate

	for rows.Next() {
		var memID int
		var oldScore float64
		var lastAccessed sql.NullString
		if err := rows.Scan(&memID, &oldScore, &lastAccessed); err != nil {
			return 0, fmt.Errorf("scan access stats: %w", err)
		}
		if !lastAccessed.Valid {
			continue
		}
		last := parseTime(lastAccessed.String)
		dt := now.Sub(last).Seconds()
		if dt <= 0 {
			continue
		}
		newScore := oldScore * math.Exp(-dt/tau)
		if math.Abs(newScore-oldScore) > 0.001 {
			updates = append(updates, scoreUpdate{MemoryID: memID, NewScore: newScore})
		}
	}
	if err := rows.Err(); err != nil {
		return 0, err
	}

	if len(updates) == 0 {
		return 0, nil
	}

	// Batch update in a transaction
	tx, err := b.db.Begin()
	if err != nil {
		return 0, fmt.Errorf("begin decay tx: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`UPDATE memory_access_stats SET access_score = ? WHERE memory_id = ?`)
	if err != nil {
		return 0, fmt.Errorf("prepare decay stmt: %w", err)
	}
	defer stmt.Close()

	for _, u := range updates {
		if _, err := stmt.Exec(u.NewScore, u.MemoryID); err != nil {
			return 0, fmt.Errorf("update score for memory %d: %w", u.MemoryID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("commit decay tx: %w", err)
	}

	return len(updates), nil
}

// MemoryStats returns counts by type for the dream report.
func (b *Brain) MemoryStats() map[string]int {
	stats := make(map[string]int)
	rows, err := b.db.Query(`
		SELECT type, COUNT(*) FROM memories
		WHERE deleted_at IS NULL
		GROUP BY type
	`)
	if err != nil {
		return stats
	}
	defer rows.Close()
	for rows.Next() {
		var typ string
		var count int
		if err := rows.Scan(&typ, &count); err == nil {
			stats[typ] = count
		}
	}
	return stats
}

// --- Helper ---

func formatTimeAgo(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	default:
		days := int(d.Hours() / 24)
		if days == 1 {
			return "1 day ago"
		}
		return fmt.Sprintf("%d days ago", days)
	}
}

// parseTime parses a datetime string from SQLite, handling multiple formats.
// SQLite stores DATETIME as text and different drivers/inserts may use different formats.
func parseTime(s string) time.Time {
	formats := []string{
		time.RFC3339,                // 2006-01-02T15:04:05Z (modernc/sqlite returns this)
		"2006-01-02T15:04:05Z07:00", // with explicit offset
		"2006-01-02 15:04:05",       // Python omo-memory writes this
		"2006-01-02T15:04:05",       // without timezone
	}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			return t
		}
	}
	return time.Time{} // zero value if unparseable
}

// md5Hash computes MD5 hash for content comparison.
func md5Hash(data []byte) [16]byte {
	return md5.Sum(data)
}
