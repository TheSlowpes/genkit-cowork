// Copyright 2026 Kevin Lopes
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

// --- mock vector backend ---

type mockVectorBackend struct {
	mu       sync.Mutex
	indexed  map[string][]*ai.Document
	indexErr error
	deleted  []string
	delErr   error
}

func newMockVectorBackend() *mockVectorBackend {
	return &mockVectorBackend{
		indexed: make(map[string][]*ai.Document),
	}
}

func (m *mockVectorBackend) Index(ctx context.Context, sessionID string, docs []*ai.Document) error {
	if m.indexErr != nil {
		return m.indexErr
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.indexed[sessionID] = append(m.indexed[sessionID], docs...)
	return nil
}

func (m *mockVectorBackend) Retrieve(ctx context.Context, sessionID, query string, topK int) ([]*ai.Document, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	docs := m.indexed[sessionID]
	if len(docs) == 0 {
		return nil, nil
	}
	if topK > len(docs) {
		topK = len(docs)
	}
	return docs[:topK], nil
}

func (m *mockVectorBackend) Delete(ctx context.Context, sessionID string) error {
	if m.delErr != nil {
		return m.delErr
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.deleted = append(m.deleted, sessionID)
	delete(m.indexed, sessionID)
	return nil
}

func (m *mockVectorBackend) indexedCount(sessionID string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.indexed[sessionID])
}

// --- VectorOperator tests ---

func TestVectorOperator_SaveState_IndexesNewMessages(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "hello world"),
			makeMessage("m2", ModelMessage, ai.RoleModel, "hi there"),
		},
	}

	if err := vop.SaveState(ctx, tenantID, "sess-1", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	if c := backend.indexedCount("sess-1"); c != 2 {
		t.Errorf("expected 2 indexed docs, got %d", c)
	}
}

func TestVectorOperator_SaveState_SkipsAlreadyIndexed(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "hello world"),
		},
	}
	if err := vop.SaveState(ctx, tenantID, "sess-1", state); err != nil {
		t.Fatalf("SaveState 1: %v", err)
	}

	// Save again with an additional message.
	state.Messages = append(state.Messages, makeMessage("m2", ModelMessage, ai.RoleModel, "reply"))
	if err := vop.SaveState(ctx, tenantID, "sess-1", state); err != nil {
		t.Fatalf("SaveState 2: %v", err)
	}

	// Only m2 should have been newly indexed.
	if c := backend.indexedCount("sess-1"); c != 2 {
		t.Errorf("expected 2 total indexed docs (1 + 1), got %d", c)
	}
}

func TestVectorOperator_SaveState_SkipsEmptyMessages(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{
			{
				MessageID: "m-empty",
				Origin:    UIMessage,
				Kind:      KindEpisodic,
				Content:   ai.Message{Role: ai.RoleUser, Content: []*ai.Part{}},
			},
		},
	}
	if err := vop.SaveState(ctx, tenantID, "sess-empty", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	if c := backend.indexedCount("sess-empty"); c != 0 {
		t.Errorf("expected 0 indexed docs for empty content, got %d", c)
	}
}

func TestVectorOperator_SaveState_SkipsNoMessageID(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{
			{
				// No MessageID
				Origin:  UIMessage,
				Kind:    KindEpisodic,
				Content: ai.Message{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("text")}},
			},
		},
	}
	if err := vop.SaveState(ctx, tenantID, "sess-noid", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	if c := backend.indexedCount("sess-noid"); c != 0 {
		t.Errorf("expected 0 indexed docs for missing ID, got %d", c)
	}
}

func TestVectorOperator_SaveState_IndexFailureRetriesNextSave(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	backend.indexErr = errors.New("simulated index failure")
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "hello"),
		},
	}

	// First save: indexing fails but SaveState returns nil.
	if err := vop.SaveState(ctx, tenantID, "sess-retry", state); err != nil {
		t.Fatalf("SaveState should not error on index failure: %v", err)
	}

	// Verify no indexed IDs file was created (so retry happens).
	idsPath := filepath.Join(dir, tenantID, "sess-retry", "indexed_ids.json")
	if _, err := os.Stat(idsPath); !errors.Is(err, os.ErrNotExist) {
		t.Error("expected indexed_ids.json to not exist after index failure")
	}

	// Fix the backend and save again.
	backend.indexErr = nil
	if err := vop.SaveState(ctx, tenantID, "sess-retry", state); err != nil {
		t.Fatalf("SaveState retry: %v", err)
	}

	// Now it should be indexed.
	if c := backend.indexedCount("sess-retry"); c != 1 {
		t.Errorf("expected 1 indexed doc after retry, got %d", c)
	}
}

func TestVectorOperator_LoadState_DelegatesToBase(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	state := SessionState{
		TenantID: "t1",
		Messages: makeMessages(5),
	}
	if err := vop.SaveState(ctx, tenantID, "sess-ld", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	loaded, err := vop.LoadState(ctx, tenantID, "sess-ld", SlidingWindow, 2)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if len(loaded.Messages) != 2 {
		t.Errorf("expected 2 messages with SlidingWindow, got %d", len(loaded.Messages))
	}
}

func TestVectorOperator_DeleteSession_CleansUpBackend(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "hello"),
		},
	}
	if err := vop.SaveState(ctx, tenantID, "sess-del", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	if err := vop.DeleteSession(ctx, tenantID, "sess-del"); err != nil {
		t.Fatalf("DeleteSession: %v", err)
	}

	// Backend should have received Delete call.
	if len(backend.deleted) != 1 || backend.deleted[0] != "sess-del" {
		t.Errorf("expected backend.Delete called with sess-del, got %v", backend.deleted)
	}

	// Indexed docs should be gone.
	if c := backend.indexedCount("sess-del"); c != 0 {
		t.Errorf("expected 0 indexed docs after delete, got %d", c)
	}

	// Session should be gone from base.
	loaded, err := vop.LoadState(ctx, tenantID, "sess-del", All, 0)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if loaded != nil {
		t.Errorf("expected nil after delete, got %+v", loaded)
	}
}

func TestVectorOperator_DeleteSession_BackendError(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "hello"),
		},
	}
	if err := vop.SaveState(ctx, tenantID, "sess-delerr", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	backend.delErr = errors.New("simulated delete failure")
	err := vop.DeleteSession(ctx, tenantID, "sess-delerr")
	if err == nil {
		t.Fatal("expected error from backend delete failure, got nil")
	}
}

func TestVectorOperator_Search(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "the quick brown fox"),
			makeMessage("m2", ModelMessage, ai.RoleModel, "jumps over the lazy dog"),
			makeMessage("m3", UIMessage, ai.RoleUser, "something unrelated"),
		},
	}
	if err := vop.SaveState(ctx, tenantID, "sess-search", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	// Mock backend returns all indexed docs (up to topK).
	results, err := vop.Search(ctx, tenantID, "sess-search", "fox", 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func TestVectorOperator_SearchZeroTopK(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	results, err := vop.Search(ctx, tenantID, "sess-x", "query", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil for topK=0, got %v", results)
	}
}

func TestVectorOperator_SearchNonexistentSession(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx := context.Background()

	results, err := vop.Search(ctx, tenantID, "nope", "query", 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil for nonexistent session, got %v", results)
	}
}

func TestVectorOperator_ContextCancelled(t *testing.T) {
	dir := t.TempDir()
	tenantID := "t1"
	base := NewFileSessionOperator(dir, tenantID)
	backend := newMockVectorBackend()
	vop := NewVectorOperator(base, backend, dir, tenantID)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	state := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{makeMessage("m1", UIMessage, ai.RoleUser, "hello")},
	}

	if err := vop.SaveState(ctx, tenantID, "sess-ctx", state); err == nil {
		t.Fatal("expected error from cancelled context on SaveState")
	}

	if _, err := vop.LoadState(ctx, tenantID, "sess-ctx", All, 0); err == nil {
		t.Fatal("expected error from cancelled context on LoadState")
	}

	if err := vop.DeleteSession(ctx, tenantID, "sess-ctx"); err == nil {
		t.Fatal("expected error from cancelled context on DeleteSession")
	}

	if _, err := vop.Search(ctx, tenantID, "sess-ctx", "query", 5); err == nil {
		t.Fatal("expected error from cancelled context on Search")
	}
}

func TestMessageText(t *testing.T) {
	tests := []struct {
		name string
		msg  SessionMessage
		want string
	}{
		{
			name: "single text part",
			msg: SessionMessage{
				Content: ai.Message{
					Content: []*ai.Part{ai.NewTextPart("hello")},
				},
			},
			want: "hello",
		},
		{
			name: "multiple text parts",
			msg: SessionMessage{
				Content: ai.Message{
					Content: []*ai.Part{
						ai.NewTextPart("hello"),
						ai.NewTextPart("world"),
					},
				},
			},
			want: "hello world",
		},
		{
			name: "empty content",
			msg: SessionMessage{
				Content: ai.Message{
					Content: []*ai.Part{},
				},
			},
			want: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := messageText(&tc.msg)
			if got != tc.want {
				t.Errorf("expected %q, got %q", tc.want, got)
			}
		})
	}
}
