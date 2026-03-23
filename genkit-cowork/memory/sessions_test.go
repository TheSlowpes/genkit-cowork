// Copyright 2026 Kevin Lopes
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
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
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/x/session"
)

// --- helpers ---

func makeMessage(id string, origin MessageOrigin, role ai.Role, text string) SessionMessage {
	return SessionMessage{
		MessageID: id,
		Origin:    origin,
		Content: ai.Message{
			Role:    role,
			Content: []*ai.Part{ai.NewTextPart(text)},
		},
		Timestamp: time.Now(),
	}
}

func makeMessages(n int) []SessionMessage {
	msgs := make([]SessionMessage, n)
	for i := range n {
		msgs[i] = makeMessage(
			"msg-"+string(rune('a'+i)),
			UIMessage,
			ai.RoleUser,
			"message "+string(rune('a'+i)),
		)
	}
	return msgs
}

// --- defaultSessionOperator ---

func TestDefaultSessionOperator_SaveAndLoad(t *testing.T) {
	ctx := context.Background()
	op := &defaultSessionOperator{}

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

func TestDefaultSessionOperator_LoadNonexistent(t *testing.T) {
	ctx := context.Background()
	op := &defaultSessionOperator{}

	state, err := op.LoadState(ctx, "does-not-exist", All, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if state != nil {
		t.Errorf("expected nil for nonexistent session, got %+v", state)
	}
}

func TestDefaultSessionOperator_Delete(t *testing.T) {
	ctx := context.Background()
	op := &defaultSessionOperator{}

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

func TestDefaultSessionOperator_DeleteNonexistent(t *testing.T) {
	ctx := context.Background()
	op := &defaultSessionOperator{}

	// Should not error when deleting a session that doesn't exist.
	if err := op.DeleteSession(ctx, "never-existed"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDefaultSessionOperator_DeepCopyOnSave(t *testing.T) {
	ctx := context.Background()
	op := &defaultSessionOperator{}

	msgs := []SessionMessage{makeMessage("m1", UIMessage, ai.RoleUser, "original")}
	state := SessionState{TenantID: "t1", Messages: msgs}

	if err := op.SaveState(ctx, "sess-copy", state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	// Mutate the original slice after save.
	msgs[0].Origin = EmailMessage

	loaded, err := op.LoadState(ctx, "sess-copy", All, 0)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if loaded.Messages[0].Origin != UIMessage {
		t.Errorf("mutation leaked through save: expected origin %q, got %q", UIMessage, loaded.Messages[0].Origin)
	}
}

func TestDefaultSessionOperator_OverwriteOnSave(t *testing.T) {
	ctx := context.Background()
	op := &defaultSessionOperator{}

	state1 := SessionState{
		TenantID: "t1",
		Messages: []SessionMessage{makeMessage("m1", UIMessage, ai.RoleUser, "first")},
	}
	if err := op.SaveState(ctx, "sess-ow", state1); err != nil {
		t.Fatalf("SaveState 1: %v", err)
	}

	state2 := SessionState{
		TenantID: "t1-updated",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "first"),
			makeMessage("m2", ModelMessage, ai.RoleModel, "second"),
		},
	}
	if err := op.SaveState(ctx, "sess-ow", state2); err != nil {
		t.Fatalf("SaveState 2: %v", err)
	}

	loaded, err := op.LoadState(ctx, "sess-ow", All, 0)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if loaded.TenantID != "t1-updated" {
		t.Errorf("expected tenant t1-updated, got %q", loaded.TenantID)
	}
	if len(loaded.Messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(loaded.Messages))
	}
}

// --- filterMessages ---

func TestFilterMessages_AllMode(t *testing.T) {
	msgs := makeMessages(5)
	result := filterMessages(msgs, All, 0)
	if len(result) != 5 {
		t.Errorf("All mode: expected 5, got %d", len(result))
	}
}

func TestFilterMessages_EmptySlice(t *testing.T) {
	result := filterMessages(nil, SlidingWindow, 3)
	if result != nil {
		t.Errorf("expected nil for empty input, got %v", result)
	}
}

func TestFilterMessages_SlidingWindow(t *testing.T) {
	msgs := makeMessages(10)

	result := filterMessages(msgs, SlidingWindow, 3)
	if len(result) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(result))
	}
	// Should be the last 3.
	for i, msg := range result {
		expected := msgs[7+i].MessageID
		if msg.MessageID != expected {
			t.Errorf("msg[%d]: expected ID %q, got %q", i, expected, msg.MessageID)
		}
	}
}

func TestFilterMessages_SlidingWindowNExceedsTotal(t *testing.T) {
	msgs := makeMessages(3)
	result := filterMessages(msgs, SlidingWindow, 10)
	if len(result) != 3 {
		t.Errorf("expected all 3 messages when n > total, got %d", len(result))
	}
}

func TestFilterMessages_SlidingWindowNZero(t *testing.T) {
	msgs := makeMessages(3)
	result := filterMessages(msgs, SlidingWindow, 0)
	if len(result) != 3 {
		t.Errorf("expected all 3 messages when n=0, got %d", len(result))
	}
}

func TestFilterMessages_TailEndsPruning(t *testing.T) {
	msgs := makeMessages(10)

	result := filterMessages(msgs, TailEndsPruning, 2)
	if len(result) != 4 {
		t.Fatalf("expected 4 messages (first 2 + last 2), got %d", len(result))
	}

	// First 2 should be msgs[0], msgs[1].
	if result[0].MessageID != msgs[0].MessageID {
		t.Errorf("head[0]: expected %q, got %q", msgs[0].MessageID, result[0].MessageID)
	}
	if result[1].MessageID != msgs[1].MessageID {
		t.Errorf("head[1]: expected %q, got %q", msgs[1].MessageID, result[1].MessageID)
	}
	// Last 2 should be msgs[8], msgs[9].
	if result[2].MessageID != msgs[8].MessageID {
		t.Errorf("tail[0]: expected %q, got %q", msgs[8].MessageID, result[2].MessageID)
	}
	if result[3].MessageID != msgs[9].MessageID {
		t.Errorf("tail[1]: expected %q, got %q", msgs[9].MessageID, result[3].MessageID)
	}
}

func TestFilterMessages_TailEndsPruningNExceedsHalf(t *testing.T) {
	msgs := makeMessages(4)
	// 2*n = 4 >= total, so all messages should be returned.
	result := filterMessages(msgs, TailEndsPruning, 2)
	if len(result) != 4 {
		t.Errorf("expected all 4 messages when 2*n >= total, got %d", len(result))
	}
}

func TestFilterMessages_TailEndsPruningNZero(t *testing.T) {
	msgs := makeMessages(5)
	result := filterMessages(msgs, TailEndsPruning, 0)
	if len(result) != 5 {
		t.Errorf("expected all 5 messages when n=0, got %d", len(result))
	}
}

// --- copyMessages ---

func TestCopyMessages_Independence(t *testing.T) {
	original := makeMessages(3)
	copied := copyMessages(original)

	if len(copied) != len(original) {
		t.Fatalf("expected %d messages, got %d", len(original), len(copied))
	}

	// Mutate the copy, verify original is unaffected.
	copied[0].Origin = ZoomMessage
	if original[0].Origin == ZoomMessage {
		t.Error("mutation of copy affected original")
	}
}

// --- Session (public API) ---

func TestNewSession_Defaults(t *testing.T) {
	s := NewSession()
	if s.opts.mode != All {
		t.Errorf("expected default mode All, got %d", s.opts.mode)
	}
	if s.opts.operator == nil {
		t.Error("expected default operator, got nil")
	}
}

func TestNewSession_CustomOptions(t *testing.T) {
	s := NewSession(
		WithPersistenceMode(SlidingWindow, 10),
	)
	if s.opts.mode != SlidingWindow {
		t.Errorf("expected SlidingWindow, got %d", s.opts.mode)
	}
	if s.opts.nMessages != 10 {
		t.Errorf("expected nMessages 10, got %d", s.opts.nMessages)
	}
}

func TestNewSession_CustomOperator(t *testing.T) {
	op := &defaultSessionOperator{}
	s := NewSession(WithCustomSessionOperator(op))
	if s.opts.operator != op {
		t.Error("expected custom operator to be set")
	}
}

func TestSession_GetNonexistentReturnsNil(t *testing.T) {
	ctx := context.Background()
	s := NewSession()

	data, err := s.Get(ctx, "nope")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if data != nil {
		t.Errorf("expected nil for nonexistent session, got %+v", data)
	}
}

func TestSession_SaveAndGet(t *testing.T) {
	ctx := context.Background()
	s := NewSession()

	state := SessionState{
		TenantID: "tenant-1",
		Messages: []SessionMessage{
			makeMessage("m1", UIMessage, ai.RoleUser, "hello"),
		},
	}

	if err := s.Save(ctx, "sess-1", &session.Data[SessionState]{ID: "sess-1", State: state}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	data, err := s.Get(ctx, "sess-1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if data == nil {
		t.Fatal("expected data, got nil")
	}
	if data.ID != "sess-1" {
		t.Errorf("expected ID sess-1, got %q", data.ID)
	}
	if data.State.TenantID != "tenant-1" {
		t.Errorf("expected tenant-1, got %q", data.State.TenantID)
	}
}

func TestSession_SaveAutoAssignsMessageID(t *testing.T) {
	ctx := context.Background()
	s := NewSession()

	msg := SessionMessage{
		// MessageID is intentionally empty.
		Origin: UIMessage,
		Content: ai.Message{
			Role:    ai.RoleUser,
			Content: []*ai.Part{ai.NewTextPart("hello")},
		},
	}

	state := SessionState{TenantID: "t1", Messages: []SessionMessage{msg}}
	if err := s.Save(ctx, "sess-auto", &session.Data[SessionState]{ID: "sess-auto", State: state}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	data, err := s.Get(ctx, "sess-auto")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if data.State.Messages[0].MessageID == "" {
		t.Error("expected auto-assigned MessageID, got empty")
	}
}

func TestSession_SaveAutoAssignsTimestamp(t *testing.T) {
	ctx := context.Background()
	s := NewSession()

	msg := SessionMessage{
		MessageID: "m1",
		Origin:    UIMessage,
		Content: ai.Message{
			Role:    ai.RoleUser,
			Content: []*ai.Part{ai.NewTextPart("hello")},
		},
		// Timestamp is intentionally zero.
	}

	before := time.Now()
	state := SessionState{TenantID: "t1", Messages: []SessionMessage{msg}}
	if err := s.Save(ctx, "sess-ts", &session.Data[SessionState]{ID: "sess-ts", State: state}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	data, err := s.Get(ctx, "sess-ts")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	ts := data.State.Messages[0].Timestamp
	if ts.IsZero() {
		t.Error("expected auto-assigned Timestamp, got zero")
	}
	if ts.Before(before) {
		t.Errorf("auto-assigned timestamp %v is before test start %v", ts, before)
	}
}

func TestSession_SavePreservesExistingIDAndTimestamp(t *testing.T) {
	ctx := context.Background()
	s := NewSession()

	fixedTime := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	msg := SessionMessage{
		MessageID: "explicit-id",
		Origin:    UIMessage,
		Content: ai.Message{
			Role:    ai.RoleUser,
			Content: []*ai.Part{ai.NewTextPart("hello")},
		},
		Timestamp: fixedTime,
	}

	state := SessionState{TenantID: "t1", Messages: []SessionMessage{msg}}
	if err := s.Save(ctx, "sess-keep", &session.Data[SessionState]{ID: "sess-keep", State: state}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	data, err := s.Get(ctx, "sess-keep")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if data.State.Messages[0].MessageID != "explicit-id" {
		t.Errorf("expected preserved ID 'explicit-id', got %q", data.State.Messages[0].MessageID)
	}
	if !data.State.Messages[0].Timestamp.Equal(fixedTime) {
		t.Errorf("expected preserved timestamp %v, got %v", fixedTime, data.State.Messages[0].Timestamp)
	}
}

func TestSession_GetWithSlidingWindow(t *testing.T) {
	ctx := context.Background()
	s := NewSession(WithPersistenceMode(SlidingWindow, 2))

	msgs := makeMessages(5)
	state := SessionState{TenantID: "t1", Messages: msgs}
	if err := s.Save(ctx, "sess-sw", &session.Data[SessionState]{ID: "sess-sw", State: state}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	data, err := s.Get(ctx, "sess-sw")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(data.State.Messages) != 2 {
		t.Errorf("expected 2 messages with SlidingWindow, got %d", len(data.State.Messages))
	}
	// Should be the last 2.
	if data.State.Messages[0].MessageID != msgs[3].MessageID {
		t.Errorf("expected last-2 messages, got IDs %q and %q",
			data.State.Messages[0].MessageID, data.State.Messages[1].MessageID)
	}
}

func TestSession_GetWithTailEndsPruning(t *testing.T) {
	ctx := context.Background()
	s := NewSession(WithPersistenceMode(TailEndsPruning, 2))

	msgs := makeMessages(10)
	state := SessionState{TenantID: "t1", Messages: msgs}
	if err := s.Save(ctx, "sess-te", &session.Data[SessionState]{ID: "sess-te", State: state}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	data, err := s.Get(ctx, "sess-te")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(data.State.Messages) != 4 {
		t.Fatalf("expected 4 messages with TailEndsPruning(2), got %d", len(data.State.Messages))
	}
	// First 2 + last 2.
	if data.State.Messages[0].MessageID != msgs[0].MessageID {
		t.Errorf("head[0]: expected %q, got %q", msgs[0].MessageID, data.State.Messages[0].MessageID)
	}
	if data.State.Messages[3].MessageID != msgs[9].MessageID {
		t.Errorf("tail[1]: expected %q, got %q", msgs[9].MessageID, data.State.Messages[3].MessageID)
	}
}
