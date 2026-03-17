package memory

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
)

func TestFileSessionOperator_SaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	state := SessionState{
		TenantID: "tenant-1",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "hello"),
			makeMessage("m2", ModelMessage, ai.RoleModel, "hi there"),
		},
	}

	if err := op.SaveState(ctx, "sess-1", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	loaded, err := op.LoadState(ctx, "sess-1", All, 0)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected state, got nil")
	}
	if loaded.TenantID != "tenant-1" {
		t.Errorf("expected tenant-1, got %q", loaded.TenantID)
	}
	if len(loaded.Messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(loaded.Messages))
	}
}

func TestFileSessionOperator_LoadNonexistent(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	state, err := op.LoadState(ctx, "does-not-exist", All, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if state != nil {
		t.Errorf("expected nil for nonexistent session, got %+v", state)
	}
}

func TestFileSessionOperator_Delete(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	state := SessionState{
		TenantID: "tenant-1",
		Messages: []SessionMessage{makeMessage("m1", UIMessage, ai.RoleUser, "hello")},
	}
	if err := op.SaveState(ctx, "sess-del", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	if err := op.DeleteSession(ctx, "sess-del"); err != nil {
		t.Fatalf("DeleteSession: %v", err)
	}

	loaded, err := op.LoadState(ctx, "sess-del", All, 0)
	if err != nil {
		t.Fatalf("LoadState after delete: %v", err)
	}
	if loaded != nil {
		t.Errorf("expected nil after delete, got %+v", loaded)
	}
}

func TestFileSessionOperator_DeleteNonexistent(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	if err := op.DeleteSession(ctx, "never-existed"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestFileSessionOperator_SlidingWindow(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	msgs := makeMessages(10)
	state := SessionState{TenantID: "t1", Messages: msgs}
	if err := op.SaveState(ctx, "sess-sw", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	loaded, err := op.LoadState(ctx, "sess-sw", SlidingWindow, 3)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if len(loaded.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(loaded.Messages))
	}
	if loaded.Messages[0].MessageID != msgs[7].MessageID {
		t.Errorf("expected first message to be msgs[7], got %q", loaded.Messages[0].MessageID)
	}
}

func TestFileSessionOperator_TailEndsPruning(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	msgs := makeMessages(10)
	state := SessionState{TenantID: "t1", Messages: msgs}
	if err := op.SaveState(ctx, "sess-te", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	loaded, err := op.LoadState(ctx, "sess-te", TailEndsPruning, 2)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if len(loaded.Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(loaded.Messages))
	}
	if loaded.Messages[0].MessageID != msgs[0].MessageID {
		t.Errorf("head[0]: expected %q, got %q", msgs[0].MessageID, loaded.Messages[0].MessageID)
	}
	if loaded.Messages[3].MessageID != msgs[9].MessageID {
		t.Errorf("tail[1]: expected %q, got %q", msgs[9].MessageID, loaded.Messages[3].MessageID)
	}
}

func TestFileSessionOperator_AtomicWrite(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{makeMessage("m1", UIMessage, ai.RoleUser, "hello")},
	}
	if err := op.SaveState(ctx, "sess-atomic", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	// Verify the state file exists and no temp files remain.
	sessDir := filepath.Join(dir, "sess-atomic")
	entries, err := os.ReadDir(sessDir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	for _, e := range entries {
		if e.Name() != "state.json" {
			t.Errorf("unexpected file in session dir: %q", e.Name())
		}
	}
}

func TestFileSessionOperator_AppendOnlyValidation(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	// Save with 3 messages.
	msgs := makeMessages(3)
	state := SessionState{TenantID: "t1", Messages: msgs}
	if err := op.SaveState(ctx, "sess-ao", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	// Try to save with fewer messages — should fail.
	shorter := SessionState{TenantID: "t1", Messages: msgs[:1]}
	err := op.SaveState(ctx, "sess-ao", shorter)
	if err == nil {
		t.Fatal("expected append-only violation error, got nil")
	}

	// Verify original state is preserved.
	loaded, err := op.LoadState(ctx, "sess-ao", All, 0)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if len(loaded.Messages) != 3 {
		t.Errorf("expected 3 messages preserved, got %d", len(loaded.Messages))
	}
}

func TestFileSessionOperator_AppendOnlyAllowsGrowth(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	msgs := makeMessages(3)
	state := SessionState{TenantID: "t1", Messages: msgs}
	if err := op.SaveState(ctx, "sess-grow", state); err != nil {
		t.Fatalf("SaveState 1: %v", err)
	}

	// Append more messages — should succeed.
	msgs = append(msgs, makeMessage("m-extra", UIMessage, ai.RoleUser, "extra"))
	state2 := SessionState{TenantID: "t1", Messages: msgs}
	if err := op.SaveState(ctx, "sess-grow", state2); err != nil {
		t.Fatalf("SaveState 2: %v", err)
	}

	loaded, err := op.LoadState(ctx, "sess-grow", All, 0)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if len(loaded.Messages) != 4 {
		t.Errorf("expected 4 messages, got %d", len(loaded.Messages))
	}
}

func TestFileSessionOperator_TenantMismatch(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	state := SessionState{
		TenantID: "tenant-A",
		Messages: []SessionMessage{makeMessage("m1", UIMessage, ai.RoleUser, "hello")},
	}
	if err := op.SaveState(ctx, "sess-tenant", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	// Attempt to save with a different tenant — should fail.
	state2 := SessionState{
		TenantID: "tenant-B",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "hello"),
			makeMessage("m2", UIMessage, ai.RoleUser, "intruder"),
		},
	}
	err := op.SaveState(ctx, "sess-tenant", state2)
	if err == nil {
		t.Fatal("expected tenant mismatch error, got nil")
	}
}

func TestFileSessionOperator_TenantConsistentSave(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	state := SessionState{
		TenantID: "tenant-A",
		Messages: []SessionMessage{makeMessage("m1", UIMessage, ai.RoleUser, "hello")},
	}
	if err := op.SaveState(ctx, "sess-same", state); err != nil {
		t.Fatalf("SaveState 1: %v", err)
	}

	// Same tenant should be fine.
	state2 := SessionState{
		TenantID: "tenant-A",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "hello"),
			makeMessage("m2", UIMessage, ai.RoleUser, "follow-up"),
		},
	}
	if err := op.SaveState(ctx, "sess-same", state2); err != nil {
		t.Fatalf("SaveState 2: %v", err)
	}
}

func TestFileSessionOperator_ContextCancelled(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{makeMessage("m1", UIMessage, ai.RoleUser, "hello")},
	}

	if err := op.SaveState(ctx, "sess-ctx", state); err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}

	if _, err := op.LoadState(ctx, "sess-ctx", All, 0); err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}

	if err := op.DeleteSession(ctx, "sess-ctx"); err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}

func TestFileSessionOperator_PreservesLastConsolidateAt(t *testing.T) {
	dir := t.TempDir()
	op := NewFileSessionOperator(dir)
	ctx := context.Background()

	ts := time.Date(2025, 6, 15, 12, 0, 0, 0, time.UTC)
	state := SessionState{
		TenantID:          "t1",
		Messages:          []SessionMessage{makeMessage("m1", UIMessage, ai.RoleUser, "hello")},
		LastConsolidateAt: ts,
	}
	if err := op.SaveState(ctx, "sess-lc", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	loaded, err := op.LoadState(ctx, "sess-lc", All, 0)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if !loaded.LastConsolidateAt.Equal(ts) {
		t.Errorf("expected LastConsolidateAt %v, got %v", ts, loaded.LastConsolidateAt)
	}
}

func TestAtomicWriteFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	data := []byte(`{"key":"value"}`)
	if err := atomicWriteFile(path, data, 0644); err != nil {
		t.Fatalf("atomicWriteFile: %v", err)
	}

	read, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(read) != string(data) {
		t.Errorf("expected %q, got %q", data, read)
	}

	// Verify no temp files remain.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 {
		t.Errorf("expected 1 file, got %d", len(entries))
	}
}
