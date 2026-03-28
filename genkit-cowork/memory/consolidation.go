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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/uuid"
)

const (
	recordTypeInsight = "insight"
)

// ConsolidationConfig controls tenant consolidation behavior.
type ConsolidationConfig struct {
	Model              string
	PromptVersion      string
	MaxSessionMessages int
	MaxFileChunks      int
	MinConfidence      float64
	Cadence            time.Duration
	Window             time.Duration
}

// ConsolidationInput is the source payload passed to an InsightDeriver.
type ConsolidationInput struct {
	TenantID    string
	WindowStart time.Time
	WindowEnd   time.Time
	Sessions    []ConsolidationSession
	Files       []ConsolidationFile
}

// ConsolidationSession is a session slice used by consolidation.
type ConsolidationSession struct {
	SessionID string
	Messages  []SessionMessage
	Turns     []TurnRecord
}

// ConsolidationFile is a tenant-global file slice used by consolidation.
type ConsolidationFile struct {
	Record FileRecord
	Chunks []FileChunkRecord
}

// InsightDraft is one candidate insight produced by a deriver.
type InsightDraft struct {
	Kind       InsightKind `json:"kind"`
	Title      string      `json:"title"`
	Summary    string      `json:"summary"`
	Evidence   []string    `json:"evidence,omitempty"`
	SessionIDs []string    `json:"sessionIDs,omitempty"`
	TurnIDs    []string    `json:"turnIDs,omitempty"`
	FileIDs    []string    `json:"fileIDs,omitempty"`
	ChunkIDs   []string    `json:"chunkIDs,omitempty"`
	Confidence float64     `json:"confidence"`
}

// InsightDeriver derives structured insights from consolidation input.
type InsightDeriver interface {
	Derive(ctx context.Context, input ConsolidationInput) ([]InsightDraft, error)
}

// LLMInsightDeriver derives insights using a configured registered Genkit model.
type LLMInsightDeriver struct {
	g      *genkit.Genkit
	model  string
	prompt string
}

type llmInsightDeriveOutput struct {
	Insights []InsightDraft `json:"insights"`
}

// NewLLMInsightDeriver returns an LLM-backed InsightDeriver.
func NewLLMInsightDeriver(g *genkit.Genkit, model string) *LLMInsightDeriver {
	return &LLMInsightDeriver{
		g:      g,
		model:  model,
		prompt: defaultConsolidationPrompt(),
	}
}

// Derive requests JSON insights from the configured model and validates output.
func (d *LLMInsightDeriver) Derive(ctx context.Context, input ConsolidationInput) ([]InsightDraft, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if d.g == nil {
		return nil, fmt.Errorf("llm deriver: genkit instance is nil")
	}
	if strings.TrimSpace(d.model) == "" {
		return nil, fmt.Errorf("llm deriver: model is required")
	}

	model := genkit.LookupModel(d.g, d.model)
	if model == nil {
		return nil, fmt.Errorf("llm deriver: model %q not found", d.model)
	}

	payload, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("llm deriver: marshal input: %w", err)
	}

	resp, err := genkit.Generate(
		ctx,
		d.g,
		ai.WithModel(model),
		ai.WithSystem(d.prompt),
		ai.WithOutputType(llmInsightDeriveOutput{}),
		ai.WithMessages(ai.NewUserTextMessage(string(payload))),
	)
	if err != nil {
		return nil, fmt.Errorf("llm deriver: generate: %w", err)
	}
	if resp == nil || resp.Message == nil {
		return nil, fmt.Errorf("llm deriver: empty model response")
	}

	var envelope llmInsightDeriveOutput
	if err := resp.Output(&envelope); err != nil {
		return nil, fmt.Errorf("llm deriver: decode output: %w", err)
	}

	validated := make([]InsightDraft, 0, len(envelope.Insights))
	for _, draft := range envelope.Insights {
		draft = normalizeInsightDraft(draft)
		if draft.Title == "" || draft.Summary == "" {
			continue
		}
		validated = append(validated, draft)
	}

	return validated, nil
}

// InsightIndexer indexes and retrieves tenant insight records.
type InsightIndexer interface {
	IndexInsights(ctx context.Context, tenantID string, insights []InsightRecord) error
	SearchInsights(ctx context.Context, tenantID, query string, topK int) ([]InsightRecord, error)
}

type vectorInsightIndexer struct {
	backend VectorBackend
}

// NewVectorInsightIndexer adapts a VectorBackend for insight indexing/search.
func NewVectorInsightIndexer(backend VectorBackend) InsightIndexer {
	if backend == nil {
		return nil
	}
	return &vectorInsightIndexer{backend: backend}
}

func (i *vectorInsightIndexer) IndexInsights(ctx context.Context, tenantID string, insights []InsightRecord) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(insights) == 0 {
		return nil
	}

	docs := make([]*ai.Document, 0, len(insights))
	for _, insight := range insights {
		docs = append(docs, ai.DocumentFromText(insight.Summary, map[string]any{
			"tenantID":        tenantID,
			"recordType":      recordTypeInsight,
			"insightID":       insight.InsightID,
			"runID":           insight.RunID,
			"kind":            string(insight.Kind),
			"confidence":      insight.Confidence,
			"promptVersion":   insight.PromptVersion,
			"deriverModelRef": insight.DeriverModelRef,
		}))
	}

	return i.backend.Index(ctx, tenantID, "insights", docs)
}

func (i *vectorInsightIndexer) SearchInsights(ctx context.Context, tenantID, query string, topK int) ([]InsightRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if topK <= 0 {
		return nil, nil
	}

	docs, err := i.backend.RetrieveTenant(ctx, tenantID, query, topK*3)
	if err != nil {
		return nil, err
	}

	results := make([]InsightRecord, 0, topK)
	for _, doc := range docs {
		rt, _ := doc.Metadata["recordType"].(string)
		if rt != recordTypeInsight {
			continue
		}
		id, _ := doc.Metadata["insightID"].(string)
		if id == "" {
			continue
		}
		runID, _ := doc.Metadata["runID"].(string)
		kind, _ := doc.Metadata["kind"].(string)
		results = append(results, InsightRecord{
			InsightID: id,
			TenantID:  tenantID,
			RunID:     runID,
			Kind:      InsightKind(kind),
			Summary:   documentText(doc),
		})
		if len(results) >= topK {
			break
		}
	}

	return results, nil
}

// ConsolidationService runs tenant-scoped insight derivation and persistence.
type ConsolidationService struct {
	sessions SessionOperator
	files    FileOperator
	insights InsightOperator
	deriver  InsightDeriver
	indexer  InsightIndexer
	cfg      ConsolidationConfig
	now      func() time.Time
}

// NewConsolidationService constructs a consolidation service.
func NewConsolidationService(
	sessions SessionOperator,
	files FileOperator,
	insights InsightOperator,
	deriver InsightDeriver,
	indexer InsightIndexer,
	cfg ConsolidationConfig,
) *ConsolidationService {
	if insights == nil {
		insights = NewDefaultInsightOperator()
	}
	if cfg.PromptVersion == "" {
		cfg.PromptVersion = "v1"
	}
	if cfg.MinConfidence <= 0 {
		cfg.MinConfidence = 0.5
	}
	if cfg.MaxSessionMessages <= 0 {
		cfg.MaxSessionMessages = 200
	}
	if cfg.MaxFileChunks <= 0 {
		cfg.MaxFileChunks = 200
	}
	if cfg.Cadence <= 0 {
		cfg.Cadence = 24 * time.Hour
	}

	return &ConsolidationService{
		sessions: sessions,
		files:    files,
		insights: insights,
		deriver:  deriver,
		indexer:  indexer,
		cfg:      cfg,
		now:      func() time.Time { return time.Now().UTC() },
	}
}

// RunTenant executes one consolidation run for a tenant.
func (s *ConsolidationService) RunTenant(ctx context.Context, tenantID string) (*ConsolidationRunRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(tenantID) == "" {
		return nil, fmt.Errorf("consolidation: tenantID is required")
	}
	if s.sessions == nil {
		return nil, fmt.Errorf("consolidation: session operator is required")
	}
	if s.files == nil {
		return nil, fmt.Errorf("consolidation: file operator is required")
	}
	if s.deriver == nil {
		return nil, fmt.Errorf("consolidation: deriver is required")
	}

	now := s.now()
	windowStart := s.computeWindowStart(ctx, tenantID, now)
	windowEnd := now

	input, sessionIDs, fileIDs, err := s.buildInput(ctx, tenantID, windowStart, windowEnd)
	if err != nil {
		return nil, err
	}

	idempotencyKey := computeConsolidationIdempotencyKey(tenantID, windowStart, windowEnd, s.cfg.PromptVersion, s.cfg.Model)
	if prior, err := s.insights.GetRunByIdempotencyKey(ctx, tenantID, idempotencyKey); err != nil {
		return nil, fmt.Errorf("consolidation: load prior run: %w", err)
	} else if prior != nil && prior.Status == ConsolidationRunSucceeded {
		copy := *prior
		return &copy, nil
	}

	run := ConsolidationRunRecord{
		RunID:          uuid.New().String(),
		TenantID:       tenantID,
		WindowStart:    windowStart,
		WindowEnd:      windowEnd,
		Model:          s.cfg.Model,
		PromptVersion:  s.cfg.PromptVersion,
		Status:         ConsolidationRunSucceeded,
		SessionCount:   len(sessionIDs),
		FileCount:      len(fileIDs),
		IdempotencyKey: idempotencyKey,
		CreatedAt:      now,
	}

	drafts, err := s.deriver.Derive(ctx, input)
	if err != nil {
		run.Status = ConsolidationRunFailed
		run.Error = err.Error()
		_ = s.insights.SaveRun(ctx, tenantID, run)
		return nil, fmt.Errorf("consolidation: derive insights: %w", err)
	}

	insightRecords := s.buildInsightRecords(tenantID, run, drafts)
	if err := s.insights.SaveInsights(ctx, tenantID, insightRecords); err != nil {
		run.Status = ConsolidationRunFailed
		run.Error = err.Error()
		_ = s.insights.SaveRun(ctx, tenantID, run)
		return nil, fmt.Errorf("consolidation: save insights: %w", err)
	}

	if s.indexer != nil && len(insightRecords) > 0 {
		if err := s.indexer.IndexInsights(ctx, tenantID, insightRecords); err != nil {
			run.Status = ConsolidationRunFailed
			run.Error = err.Error()
			_ = s.insights.SaveRun(ctx, tenantID, run)
			return nil, fmt.Errorf("consolidation: index insights: %w", err)
		}
	}

	if err := s.updateSessionConsolidationTime(ctx, tenantID, sessionIDs, now); err != nil {
		run.Status = ConsolidationRunFailed
		run.Error = err.Error()
		_ = s.insights.SaveRun(ctx, tenantID, run)
		return nil, fmt.Errorf("consolidation: update sessions: %w", err)
	}

	run.InsightCount = len(insightRecords)
	if err := s.insights.SaveRun(ctx, tenantID, run); err != nil {
		return nil, fmt.Errorf("consolidation: save run: %w", err)
	}

	return &run, nil
}

// SearchTenantInsights retrieves semantically similar insight records.
func (s *ConsolidationService) SearchTenantInsights(ctx context.Context, tenantID, query string, topK int) ([]InsightRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.indexer == nil {
		return nil, fmt.Errorf("consolidation: insight indexer is not configured")
	}
	if strings.TrimSpace(tenantID) == "" {
		return nil, fmt.Errorf("consolidation: tenantID is required")
	}
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("consolidation: query is required")
	}
	if topK <= 0 {
		topK = 5
	}
	return s.indexer.SearchInsights(ctx, tenantID, query, topK)
}

func (s *ConsolidationService) computeWindowStart(ctx context.Context, tenantID string, now time.Time) time.Time {
	if s.cfg.Window > 0 {
		return now.Add(-s.cfg.Window)
	}

	runs, err := s.insights.ListRuns(ctx, tenantID)
	if err == nil {
		for i := len(runs) - 1; i >= 0; i-- {
			if runs[i].Status == ConsolidationRunSucceeded {
				return runs[i].WindowEnd
			}
		}
	}

	return now.Add(-s.cfg.Cadence)
}

func (s *ConsolidationService) buildInput(ctx context.Context, tenantID string, windowStart, windowEnd time.Time) (ConsolidationInput, []string, []string, error) {
	sessionIDs, err := s.sessions.ListSessions(ctx, tenantID)
	if err != nil {
		return ConsolidationInput{}, nil, nil, fmt.Errorf("consolidation: list sessions: %w", err)
	}

	files, err := s.files.ListFiles(ctx, tenantID)
	if err != nil {
		return ConsolidationInput{}, nil, nil, fmt.Errorf("consolidation: list files: %w", err)
	}

	input := ConsolidationInput{TenantID: tenantID, WindowStart: windowStart, WindowEnd: windowEnd}

	outSessionIDs := make([]string, 0, len(sessionIDs))
	for _, sessionID := range sessionIDs {
		state, err := s.sessions.LoadState(ctx, tenantID, sessionID, All, 0)
		if err != nil {
			return ConsolidationInput{}, nil, nil, fmt.Errorf("consolidation: load session %q: %w", sessionID, err)
		}
		if state == nil {
			continue
		}

		messages := filterMessagesByTimestamp(state.Messages, windowStart, windowEnd)
		if len(messages) == 0 {
			continue
		}
		if len(messages) > s.cfg.MaxSessionMessages {
			messages = messages[len(messages)-s.cfg.MaxSessionMessages:]
		}

		input.Sessions = append(input.Sessions, ConsolidationSession{
			SessionID: sessionID,
			Messages:  messages,
			Turns:     state.Turns,
		})
		outSessionIDs = append(outSessionIDs, sessionID)
	}

	outFileIDs := make([]string, 0, len(files))
	for _, file := range files {
		if file.UploadedAt.Before(windowStart) || file.UploadedAt.After(windowEnd) {
			continue
		}
		chunks, err := s.files.ListFileChunks(ctx, tenantID, file.FileID)
		if err != nil {
			return ConsolidationInput{}, nil, nil, fmt.Errorf("consolidation: list file chunks %q: %w", file.FileID, err)
		}
		if len(chunks) == 0 {
			continue
		}
		if len(chunks) > s.cfg.MaxFileChunks {
			chunks = chunks[:s.cfg.MaxFileChunks]
		}

		input.Files = append(input.Files, ConsolidationFile{Record: file, Chunks: chunks})
		outFileIDs = append(outFileIDs, file.FileID)
	}

	return input, outSessionIDs, outFileIDs, nil
}

func filterMessagesByTimestamp(messages []SessionMessage, start, end time.Time) []SessionMessage {
	if start.IsZero() && end.IsZero() {
		return copyMessages(messages)
	}
	out := make([]SessionMessage, 0, len(messages))
	for _, msg := range messages {
		ts := msg.Timestamp
		if ts.IsZero() {
			out = append(out, msg)
			continue
		}
		if !start.IsZero() && ts.Before(start) {
			continue
		}
		if !end.IsZero() && ts.After(end) {
			continue
		}
		out = append(out, msg)
	}
	return out
}

func (s *ConsolidationService) buildInsightRecords(tenantID string, run ConsolidationRunRecord, drafts []InsightDraft) []InsightRecord {
	if len(drafts) == 0 {
		return nil
	}

	byIdempotency := make(map[string]InsightRecord)
	for _, draft := range drafts {
		draft = normalizeInsightDraft(draft)
		if draft.Confidence < s.cfg.MinConfidence {
			continue
		}
		key := computeInsightIdempotencyKey(tenantID, run.RunID, draft)
		if _, exists := byIdempotency[key]; exists {
			continue
		}

		now := s.now()
		rec := InsightRecord{
			InsightID:       uuid.New().String(),
			TenantID:        tenantID,
			RunID:           run.RunID,
			Version:         1,
			Kind:            draft.Kind,
			Title:           draft.Title,
			Summary:         draft.Summary,
			Evidence:        append([]string(nil), draft.Evidence...),
			SessionIDs:      append([]string(nil), draft.SessionIDs...),
			TurnIDs:         append([]string(nil), draft.TurnIDs...),
			FileIDs:         append([]string(nil), draft.FileIDs...),
			ChunkIDs:        append([]string(nil), draft.ChunkIDs...),
			Confidence:      draft.Confidence,
			CreatedAt:       now,
			IdempotencyKey:  key,
			PromptVersion:   s.cfg.PromptVersion,
			DeriverModelRef: s.cfg.Model,
		}
		byIdempotency[key] = rec
	}

	keys := make([]string, 0, len(byIdempotency))
	for key := range byIdempotency {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]InsightRecord, 0, len(keys))
	for _, key := range keys {
		out = append(out, byIdempotency[key])
	}
	return out
}

func (s *ConsolidationService) updateSessionConsolidationTime(ctx context.Context, tenantID string, sessionIDs []string, at time.Time) error {
	for _, sessionID := range sessionIDs {
		state, err := s.sessions.LoadState(ctx, tenantID, sessionID, All, 0)
		if err != nil {
			return fmt.Errorf("load session %q for consolidation update: %w", sessionID, err)
		}
		if state == nil {
			continue
		}
		if !state.LastConsolidateAt.Before(at) {
			continue
		}
		state.LastConsolidateAt = at
		if err := s.sessions.SaveState(ctx, tenantID, sessionID, *state); err != nil {
			return fmt.Errorf("save session %q consolidation timestamp: %w", sessionID, err)
		}
	}
	return nil
}

func computeConsolidationIdempotencyKey(tenantID string, start, end time.Time, promptVersion, model string) string {
	raw := fmt.Sprintf("tenant=%s|start=%s|end=%s|prompt=%s|model=%s", tenantID, start.UTC().Format(time.RFC3339Nano), end.UTC().Format(time.RFC3339Nano), promptVersion, model)
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:])
}

func computeInsightIdempotencyKey(tenantID, runID string, draft InsightDraft) string {
	raw := fmt.Sprintf(
		"tenant=%s|run=%s|kind=%s|title=%s|summary=%s|sessions=%s|turns=%s|files=%s|chunks=%s",
		tenantID,
		runID,
		draft.Kind,
		draft.Title,
		draft.Summary,
		strings.Join(draft.SessionIDs, ","),
		strings.Join(draft.TurnIDs, ","),
		strings.Join(draft.FileIDs, ","),
		strings.Join(draft.ChunkIDs, ","),
	)
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:])
}

func normalizeInsightDraft(in InsightDraft) InsightDraft {
	in.Title = strings.TrimSpace(in.Title)
	in.Summary = strings.TrimSpace(in.Summary)
	if in.Confidence < 0 {
		in.Confidence = 0
	}
	if in.Confidence > 1 {
		in.Confidence = 1
	}
	if in.Kind == "" {
		in.Kind = InsightKindFact
	}
	in.Evidence = normalizeStringList(in.Evidence)
	in.SessionIDs = normalizeStringList(in.SessionIDs)
	in.TurnIDs = normalizeStringList(in.TurnIDs)
	in.FileIDs = normalizeStringList(in.FileIDs)
	in.ChunkIDs = normalizeStringList(in.ChunkIDs)
	return in
}

func normalizeStringList(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]bool)
	out := make([]string, 0, len(in))
	for _, value := range in {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func defaultConsolidationPrompt() string {
	return strings.TrimSpace(`You derive durable, tenant-scoped memory insights from structured conversation and file history.

Rules:
- Focus on high-signal, reusable insights.
- Keep titles and summaries concise and factual.
- Include provenance IDs only when present in the input.
- Confidence must be between 0 and 1.
- Avoid duplicates and near-duplicates.`)
}
