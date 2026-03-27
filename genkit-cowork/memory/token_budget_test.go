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
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/x/session"
)

func msgWithUsage(id string, total int) SessionMessage {
	m := ai.NewUserTextMessage(id)
	m.Metadata = map[string]any{
		"generationUsage": map[string]any{
			"totalTokens": total,
		},
	}
	return SessionMessage{MessageID: id, Content: *m}
}

func TestApplyTokenBudget_ExactFit(t *testing.T) {
	msgs := []SessionMessage{
		msgWithUsage("m1", 5),
		msgWithUsage("m2", 4),
		msgWithUsage("m3", 3),
	}

	got, err := applyTokenBudget(msgs, 7, nil)
	if err != nil {
		t.Fatalf("applyTokenBudget: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(got) = %d, want 2", len(got))
	}
	if got[0].MessageID != "m2" || got[1].MessageID != "m3" {
		t.Fatalf("unexpected messages: %+v", got)
	}
}

func TestApplyTokenBudget_OverflowByOne(t *testing.T) {
	msgs := []SessionMessage{
		msgWithUsage("m1", 4),
		msgWithUsage("m2", 4),
		msgWithUsage("m3", 4),
	}

	got, err := applyTokenBudget(msgs, 7, nil)
	if err != nil {
		t.Fatalf("applyTokenBudget: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("len(got) = %d, want 1", len(got))
	}
	if got[0].MessageID != "m3" {
		t.Fatalf("expected m3, got %q", got[0].MessageID)
	}
}

func TestApplyTokenBudget_EmptyAndHugeBudget(t *testing.T) {
	msgs := []SessionMessage{
		msgWithUsage("m1", 2),
		msgWithUsage("m2", 3),
	}

	gotEmpty, err := applyTokenBudget(msgs, 0, nil)
	if err != nil {
		t.Fatalf("applyTokenBudget empty: %v", err)
	}
	if len(gotEmpty) != 0 {
		t.Fatalf("empty budget should produce 0 messages, got %d", len(gotEmpty))
	}

	gotHuge, err := applyTokenBudget(msgs, 1000, nil)
	if err != nil {
		t.Fatalf("applyTokenBudget huge: %v", err)
	}
	if len(gotHuge) != 2 {
		t.Fatalf("huge budget should keep all, got %d", len(gotHuge))
	}
}

func TestSession_GetTokenBudgetMode(t *testing.T) {
	ctx := context.Background()
	s := NewSession(
		WithTenantID("tenant-1"),
		WithTokenBudget(7),
	)

	state := SessionState{
		TenantID: "tenant-1",
		Messages: []SessionMessage{
			msgWithUsage("m1", 5),
			msgWithUsage("m2", 4),
			msgWithUsage("m3", 3),
		},
	}
	if err := s.Save(ctx, "sess-token", &session.Data[SessionState]{ID: "sess-token", State: state}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	got, err := s.Get(ctx, "sess-token")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got == nil {
		t.Fatal("expected state")
	}
	if len(got.State.Messages) != 2 {
		t.Fatalf("token-budget messages = %d, want 2", len(got.State.Messages))
	}
	if got.State.Messages[0].MessageID != "m2" || got.State.Messages[1].MessageID != "m3" {
		t.Fatalf("unexpected pruned messages: %+v", got.State.Messages)
	}
}
