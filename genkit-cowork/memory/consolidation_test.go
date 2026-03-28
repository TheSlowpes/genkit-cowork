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
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/firebase/genkit/go/genkit"
)

type stubDeriver struct {
	drafts []InsightDraft
	err    error
}

func (s stubDeriver) Derive(ctx context.Context, input ConsolidationInput) ([]InsightDraft, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.drafts, nil
}

func TestConsolidationService_RunTenant_SavesInsightsAndRun(t *testing.T) {
	ctx := context.Background()
	sessionOp := &defaultSessionOperator{}
	sessions := NewSession(WithCustomSessionOperator(sessionOp), WithTenantID("tenant-1"))
	files := NewDefaultFileOperator()
	insights := NewDefaultInsightOperator()

	state := SessionState{
		TenantID: "tenant-1",
		Messages: []SessionMessage{{
			MessageID: "m1",
			Origin:    UIMessage,
			Kind:      KindEpisodic,
			Content:   *ai.NewUserTextMessage("customer prefers monthly invoices"),
			Timestamp: time.Now().UTC(),
			Sequence:  1,
		}},
	}
	if err := sessions.ForTenant("tenant-1").Save(ctx, "session-1", &session.Data[SessionState]{
		ID:    "session-1",
		State: state,
	}); err != nil {
		t.Fatalf("session save error = %v", err)
	}

	now := time.Now().UTC()
	if err := files.SaveFileRecord(ctx, "tenant-1", FileRecord{
		FileID:       "file-1",
		TenantID:     "tenant-1",
		Name:         "policy.md",
		MimeType:     "text/markdown",
		UploadedAt:   now,
		IngestStatus: FileIngestCompleted,
	}); err != nil {
		t.Fatalf("save file record error = %v", err)
	}
	if err := files.SaveFileChunks(ctx, "tenant-1", "file-1", []FileChunkRecord{{
		ChunkID:       "file-1:0",
		FileID:        "file-1",
		TenantID:      "tenant-1",
		Text:          "billing policies and invoice disputes",
		UploadedAt:    now,
		Index:         0,
		TokenEstimate: 5,
	}}); err != nil {
		t.Fatalf("save file chunks error = %v", err)
	}

	service := NewConsolidationService(
		sessionOp,
		files,
		insights,
		NewDefaultPreferenceOperator(),
		stubDeriver{drafts: []InsightDraft{{
			Kind:       InsightKindFact,
			Title:      "Invoice cadence",
			Summary:    "Tenant expects monthly invoice cadence",
			SessionIDs: []string{"session-1"},
			FileIDs:    []string{"file-1"},
			ChunkIDs:   []string{"file-1:0"},
			Confidence: 0.9,
		}}},
		nil,
		ConsolidationConfig{Model: "test/unused", PromptVersion: "v1"},
	)

	run, err := service.RunTenant(ctx, "tenant-1")
	if err != nil {
		t.Fatalf("RunTenant() error = %v", err)
	}
	if run.Status != ConsolidationRunSucceeded {
		t.Fatalf("run status = %q, want %q", run.Status, ConsolidationRunSucceeded)
	}
	if run.InsightCount != 1 {
		t.Fatalf("run insight count = %d, want 1", run.InsightCount)
	}

	storedInsights, err := insights.ListInsights(ctx, "tenant-1")
	if err != nil {
		t.Fatalf("ListInsights() error = %v", err)
	}
	if len(storedInsights) != 1 {
		t.Fatalf("len(storedInsights) = %d, want 1", len(storedInsights))
	}
	if storedInsights[0].Title != "Invoice cadence" {
		t.Fatalf("insight title = %q, want %q", storedInsights[0].Title, "Invoice cadence")
	}
}

func TestConsolidationService_RunTenant_Idempotent(t *testing.T) {
	ctx := context.Background()
	sessionOp := &defaultSessionOperator{}
	sessions := NewSession(WithCustomSessionOperator(sessionOp), WithTenantID("tenant-1"))
	files := NewDefaultFileOperator()
	insights := NewDefaultInsightOperator()

	if err := sessions.ForTenant("tenant-1").Save(ctx, "session-1", &session.Data[SessionState]{
		ID: "session-1",
		State: SessionState{
			TenantID: "tenant-1",
			Messages: []SessionMessage{{
				MessageID: "m1",
				Origin:    UIMessage,
				Kind:      KindEpisodic,
				Content:   *ai.NewUserTextMessage("hello"),
				Timestamp: time.Now().UTC(),
			}},
		},
	}); err != nil {
		t.Fatalf("session save error = %v", err)
	}

	service := NewConsolidationService(
		sessionOp,
		files,
		insights,
		NewDefaultPreferenceOperator(),
		stubDeriver{drafts: []InsightDraft{{
			Kind:       InsightKindFact,
			Title:      "Greeting",
			Summary:    "User greeted",
			SessionIDs: []string{"session-1"},
			Confidence: 0.8,
		}}},
		nil,
		ConsolidationConfig{Model: "test/unused", PromptVersion: "v1", Window: 10 * time.Minute},
	)

	fixedNow := time.Now().UTC().Truncate(time.Second)
	service.now = func() time.Time { return fixedNow }

	run1, err := service.RunTenant(ctx, "tenant-1")
	if err != nil {
		t.Fatalf("RunTenant first error = %v", err)
	}
	run2, err := service.RunTenant(ctx, "tenant-1")
	if err != nil {
		t.Fatalf("RunTenant second error = %v", err)
	}
	if run1.RunID != run2.RunID {
		t.Fatalf("expected idempotent run reuse, run1=%q run2=%q", run1.RunID, run2.RunID)
	}
}

func TestConsolidationService_RunTenant_PromotesPreferenceCandidates(t *testing.T) {
	ctx := context.Background()
	sessionOp := &defaultSessionOperator{}
	sessions := NewSession(WithCustomSessionOperator(sessionOp), WithTenantID("tenant-1"))
	files := NewDefaultFileOperator()
	insights := NewDefaultInsightOperator()
	prefs := NewDefaultPreferenceOperator()

	if err := sessions.ForTenant("tenant-1").Save(ctx, "session-1", &session.Data[SessionState]{
		ID: "session-1",
		State: SessionState{
			TenantID: "tenant-1",
			Messages: []SessionMessage{{
				MessageID: "m1",
				Origin:    UIMessage,
				Kind:      KindEpisodic,
				Content:   *ai.NewUserTextMessage("Keep responses concise"),
				Timestamp: time.Now().UTC(),
			}},
		},
	}); err != nil {
		t.Fatalf("session save error = %v", err)
	}

	service := NewConsolidationService(
		sessionOp,
		files,
		insights,
		prefs,
		stubDeriver{drafts: []InsightDraft{
			{
				Kind:       InsightKindPreferenceCandidate,
				Title:      "response_style: concise",
				Summary:    "User prefers concise responses",
				SessionIDs: []string{"session-1"},
				Confidence: 0.92,
			},
			{
				Kind:       InsightKindPreferenceCandidate,
				Title:      "verbosity",
				Summary:    "User likes detailed explanations",
				SessionIDs: []string{"session-1"},
				Confidence: 0.55,
			},
		}},
		nil,
		ConsolidationConfig{Model: "test/unused", PromptVersion: "v1", PreferencePromotionConfidence: 0.8},
	)

	if _, err := service.RunTenant(ctx, "tenant-1"); err != nil {
		t.Fatalf("RunTenant() error = %v", err)
	}

	implicit, err := prefs.ListPreferences(ctx, "tenant-1", PreferenceFilter{Source: PreferenceSourceImplicit})
	if err != nil {
		t.Fatalf("ListPreferences(implicit) error = %v", err)
	}
	if len(implicit) != 1 {
		t.Fatalf("len(implicit) = %d, want 1", len(implicit))
	}
	if implicit[0].Key != "response_style" || implicit[0].Value != "concise" {
		t.Fatalf("promoted preference = %+v, want key=response_style value=concise", implicit[0])
	}
	if implicit[0].Confidence < 0.8 {
		t.Fatalf("promoted preference confidence = %v, want >=0.8", implicit[0].Confidence)
	}
}

func TestLLMInsightDeriver_Derive(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)

	genkit.DefineModel(
		g,
		"test/consolidation-deriver",
		&ai.ModelOptions{Supports: &ai.ModelSupports{SystemRole: true, Multiturn: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				FinishReason: ai.FinishReasonStop,
				Message:      &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart(`{"insights":[{"kind":"fact","title":"Billing preference","summary":"Customer prefers monthly invoices","confidence":0.91,"sessionIDs":["session-1"],"fileIDs":["file-1"],"chunkIDs":["file-1:0"],"evidence":["monthly invoices"]}]}`)}},
			}, nil
		},
	)

	deriver := NewLLMInsightDeriver(g, "test/consolidation-deriver")
	insights, err := deriver.Derive(ctx, ConsolidationInput{TenantID: "tenant-1"})
	if err != nil {
		t.Fatalf("Derive() error = %v", err)
	}
	if len(insights) != 1 {
		t.Fatalf("len(insights) = %d, want 1", len(insights))
	}
	if insights[0].Title != "Billing preference" {
		t.Fatalf("insight title = %q, want %q", insights[0].Title, "Billing preference")
	}
}

func TestVectorInsightIndexer_SearchFiltersRecordType(t *testing.T) {
	ctx := context.Background()
	backend := newMockVectorBackend()
	indexer := NewVectorInsightIndexer(backend)

	insight := InsightRecord{InsightID: "i-1", RunID: "run-1", Kind: InsightKindFact, Summary: "monthly invoices"}
	if err := indexer.IndexInsights(ctx, "tenant-1", []InsightRecord{insight}); err != nil {
		t.Fatalf("IndexInsights() error = %v", err)
	}

	if err := backend.Index(ctx, "tenant-1", "session-1", []*ai.Document{ai.DocumentFromText("session message", map[string]any{"recordType": "session_message", "messageID": "m1"})}); err != nil {
		t.Fatalf("backend.Index() session doc error = %v", err)
	}

	results, err := indexer.SearchInsights(ctx, "tenant-1", "invoice", 5)
	if err != nil {
		t.Fatalf("SearchInsights() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	if results[0].InsightID != "i-1" {
		t.Fatalf("insightID = %q, want %q", results[0].InsightID, "i-1")
	}
}

func TestNormalizeInsightDraft(t *testing.T) {
	draft := normalizeInsightDraft(InsightDraft{
		Kind:       "",
		Title:      "  X  ",
		Summary:    "  Y  ",
		Confidence: 2,
		SessionIDs: []string{"s2", "", "s1", "s2"},
	})

	if draft.Kind != InsightKindFact {
		t.Fatalf("Kind = %q, want %q", draft.Kind, InsightKindFact)
	}
	if draft.Title != "X" || draft.Summary != "Y" {
		t.Fatalf("trimmed values = (%q,%q), want (X,Y)", draft.Title, draft.Summary)
	}
	if draft.Confidence != 1 {
		t.Fatalf("Confidence = %v, want 1", draft.Confidence)
	}
	if len(draft.SessionIDs) != 2 || draft.SessionIDs[0] != "s1" || draft.SessionIDs[1] != "s2" {
		t.Fatalf("SessionIDs = %v, want [s1 s2]", draft.SessionIDs)
	}
}

func TestComputeConsolidationIdempotencyKeyStable(t *testing.T) {
	t0 := time.Now().UTC().Truncate(time.Second)
	a := computeConsolidationIdempotencyKey("tenant-1", t0, t0.Add(time.Minute), "v1", "m1")
	b := computeConsolidationIdempotencyKey("tenant-1", t0, t0.Add(time.Minute), "v1", "m1")
	if a != b {
		t.Fatalf("expected stable keys, got %q and %q", a, b)
	}

	c := computeConsolidationIdempotencyKey("tenant-1", t0, t0.Add(2*time.Minute), "v1", "m1")
	if a == c {
		t.Fatalf("expected different key for different window")
	}
}

func TestConsolidationService_SearchTenantInsightsRequiresIndexer(t *testing.T) {
	service := NewConsolidationService(&defaultSessionOperator{}, NewDefaultFileOperator(), NewDefaultInsightOperator(), NewDefaultPreferenceOperator(), stubDeriver{}, nil, ConsolidationConfig{})
	_, err := service.SearchTenantInsights(context.Background(), "tenant-1", "query", 3)
	if err == nil {
		t.Fatal("expected missing indexer error")
	}
}

func TestLLMInsightDeriver_MissingModel(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	deriver := NewLLMInsightDeriver(g, "")
	_, err := deriver.Derive(ctx, ConsolidationInput{TenantID: "tenant-1"})
	if err == nil {
		t.Fatal("expected missing model error")
	}
}

func TestLLMInsightDeriver_InvalidJSON(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)

	genkit.DefineModel(
		g,
		"test/consolidation-invalid-json",
		&ai.ModelOptions{Supports: &ai.ModelSupports{SystemRole: true, Multiturn: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				FinishReason: ai.FinishReasonStop,
				Message:      &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("not json")}},
			}, nil
		},
	)

	deriver := NewLLMInsightDeriver(g, "test/consolidation-invalid-json")
	_, err := deriver.Derive(ctx, ConsolidationInput{TenantID: "tenant-1"})
	if err == nil {
		t.Fatal("expected parse error")
	}
}

func TestFilterMessagesByTimestamp(t *testing.T) {
	now := time.Now().UTC()
	messages := []SessionMessage{
		{MessageID: "m1", Timestamp: now.Add(-2 * time.Hour)},
		{MessageID: "m2", Timestamp: now.Add(-30 * time.Minute)},
		{MessageID: "m3", Timestamp: now.Add(30 * time.Minute)},
	}

	got := filterMessagesByTimestamp(messages, now.Add(-time.Hour), now)
	if len(got) != 1 || got[0].MessageID != "m2" {
		t.Fatalf("unexpected filtered messages: %+v", got)
	}
}
