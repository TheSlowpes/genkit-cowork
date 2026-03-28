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
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"sync"
	"time"
)

// InsightKind classifies one derived insight.
type InsightKind string

const (
	InsightKindFact                InsightKind = "fact"
	InsightKindTask                InsightKind = "task"
	InsightKindRisk                InsightKind = "risk"
	InsightKindPreferenceCandidate InsightKind = "preference-candidate"
)

// ConsolidationRunStatus is the outcome status for a consolidation run.
type ConsolidationRunStatus string

const (
	ConsolidationRunSucceeded ConsolidationRunStatus = "succeeded"
	ConsolidationRunFailed    ConsolidationRunStatus = "failed"
	ConsolidationRunSkipped   ConsolidationRunStatus = "skipped"
)

// InsightRecord is an immutable derived memory record.
type InsightRecord struct {
	InsightID       string      `json:"insightID"`
	TenantID        string      `json:"tenantID"`
	RunID           string      `json:"runID"`
	Version         int         `json:"version"`
	Kind            InsightKind `json:"kind"`
	Title           string      `json:"title"`
	Summary         string      `json:"summary"`
	Evidence        []string    `json:"evidence,omitempty"`
	SessionIDs      []string    `json:"sessionIDs,omitempty"`
	TurnIDs         []string    `json:"turnIDs,omitempty"`
	FileIDs         []string    `json:"fileIDs,omitempty"`
	ChunkIDs        []string    `json:"chunkIDs,omitempty"`
	Confidence      float64     `json:"confidence"`
	CreatedAt       time.Time   `json:"createdAt"`
	IdempotencyKey  string      `json:"idempotencyKey"`
	PromptVersion   string      `json:"promptVersion"`
	DeriverModelRef string      `json:"deriverModelRef"`
}

// ConsolidationRunRecord tracks one consolidation attempt for a tenant.
type ConsolidationRunRecord struct {
	RunID          string                 `json:"runID"`
	TenantID       string                 `json:"tenantID"`
	WindowStart    time.Time              `json:"windowStart"`
	WindowEnd      time.Time              `json:"windowEnd"`
	Model          string                 `json:"model"`
	PromptVersion  string                 `json:"promptVersion"`
	Status         ConsolidationRunStatus `json:"status"`
	Error          string                 `json:"error,omitempty"`
	InsightCount   int                    `json:"insightCount"`
	SessionCount   int                    `json:"sessionCount"`
	FileCount      int                    `json:"fileCount"`
	IdempotencyKey string                 `json:"idempotencyKey"`
	CreatedAt      time.Time              `json:"createdAt"`
}

// InsightOperator persists and queries derived insights and run ledger entries.
type InsightOperator interface {
	SaveInsights(ctx context.Context, tenantID string, insights []InsightRecord) error
	ListInsights(ctx context.Context, tenantID string) ([]InsightRecord, error)
	SaveRun(ctx context.Context, tenantID string, run ConsolidationRunRecord) error
	ListRuns(ctx context.Context, tenantID string) ([]ConsolidationRunRecord, error)
	GetRunByIdempotencyKey(ctx context.Context, tenantID, idempotencyKey string) (*ConsolidationRunRecord, error)
}

// DefaultInsightOperator is the in-memory InsightOperator implementation.
type DefaultInsightOperator struct {
	mu       sync.Mutex
	insights map[string]map[string]InsightRecord
	runs     map[string][]ConsolidationRunRecord
}

// NewDefaultInsightOperator creates an in-memory InsightOperator.
func NewDefaultInsightOperator() *DefaultInsightOperator {
	return &DefaultInsightOperator{
		insights: make(map[string]map[string]InsightRecord),
		runs:     make(map[string][]ConsolidationRunRecord),
	}
}

var _ InsightOperator = (*DefaultInsightOperator)(nil)

// SaveInsights appends new immutable insights for a tenant.
func (o *DefaultInsightOperator) SaveInsights(ctx context.Context, tenantID string, insights []InsightRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("insight operator: context cancelled: %w", err)
	}
	if tenantID == "" {
		return fmt.Errorf("insight operator: tenantID is required")
	}
	if len(insights) == 0 {
		return nil
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	if _, ok := o.insights[tenantID]; !ok {
		o.insights[tenantID] = make(map[string]InsightRecord)
	}

	for _, insight := range insights {
		if insight.InsightID == "" {
			return fmt.Errorf("insight operator: insightID is required")
		}
		if insight.TenantID != "" && insight.TenantID != tenantID {
			return fmt.Errorf("insight operator: tenant mismatch: got %q want %q", insight.TenantID, tenantID)
		}
		if _, exists := o.insights[tenantID][insight.InsightID]; exists {
			continue
		}
		insight.TenantID = tenantID
		o.insights[tenantID][insight.InsightID] = copyInsightRecord(insight)
	}

	return nil
}

// ListInsights lists all tenant insights in deterministic order.
func (o *DefaultInsightOperator) ListInsights(ctx context.Context, tenantID string) ([]InsightRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("insight operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	tenantInsights, ok := o.insights[tenantID]
	if !ok {
		return []InsightRecord{}, nil
	}

	keys := slices.Collect(maps.Keys(tenantInsights))
	sort.Strings(keys)
	out := make([]InsightRecord, 0, len(keys))
	for _, key := range keys {
		out = append(out, copyInsightRecord(tenantInsights[key]))
	}
	return out, nil
}

// SaveRun appends one run record to the tenant run ledger.
func (o *DefaultInsightOperator) SaveRun(ctx context.Context, tenantID string, run ConsolidationRunRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("insight operator: context cancelled: %w", err)
	}
	if tenantID == "" {
		return fmt.Errorf("insight operator: tenantID is required")
	}
	if run.RunID == "" {
		return fmt.Errorf("insight operator: runID is required")
	}
	if run.TenantID != "" && run.TenantID != tenantID {
		return fmt.Errorf("insight operator: tenant mismatch: got %q want %q", run.TenantID, tenantID)
	}
	run.TenantID = tenantID

	o.mu.Lock()
	defer o.mu.Unlock()

	for _, existing := range o.runs[tenantID] {
		if existing.RunID == run.RunID {
			return nil
		}
	}
	o.runs[tenantID] = append(o.runs[tenantID], copyConsolidationRunRecord(run))
	return nil
}

// ListRuns lists all tenant run records.
func (o *DefaultInsightOperator) ListRuns(ctx context.Context, tenantID string) ([]ConsolidationRunRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("insight operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	runs := o.runs[tenantID]
	out := make([]ConsolidationRunRecord, len(runs))
	for i := range runs {
		out[i] = copyConsolidationRunRecord(runs[i])
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].CreatedAt.Before(out[j].CreatedAt) })
	return out, nil
}

// GetRunByIdempotencyKey returns the newest run for the idempotency key.
func (o *DefaultInsightOperator) GetRunByIdempotencyKey(ctx context.Context, tenantID, idempotencyKey string) (*ConsolidationRunRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("insight operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	var match *ConsolidationRunRecord
	for i := range o.runs[tenantID] {
		run := o.runs[tenantID][i]
		if run.IdempotencyKey != idempotencyKey {
			continue
		}
		copy := copyConsolidationRunRecord(run)
		match = &copy
	}
	return match, nil
}

type fileInsightOperator struct {
	rootDir string
	mu      sync.Mutex
}

// NewFileInsightOperator returns a filesystem-backed InsightOperator.
func NewFileInsightOperator(rootDir string) InsightOperator {
	return &fileInsightOperator{rootDir: rootDir}
}

var _ InsightOperator = (*fileInsightOperator)(nil)

func (o *fileInsightOperator) SaveInsights(ctx context.Context, tenantID string, insights []InsightRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("file insight operator: context cancelled: %w", err)
	}
	if tenantID == "" {
		return fmt.Errorf("file insight operator: tenantID is required")
	}
	if len(insights) == 0 {
		return nil
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	existing, err := o.readInsights(tenantID)
	if err != nil {
		return err
	}
	byID := make(map[string]InsightRecord, len(existing))
	for _, item := range existing {
		byID[item.InsightID] = item
	}

	for _, insight := range insights {
		if insight.InsightID == "" {
			return fmt.Errorf("file insight operator: insightID is required")
		}
		if insight.TenantID != "" && insight.TenantID != tenantID {
			return fmt.Errorf("file insight operator: tenant mismatch: got %q want %q", insight.TenantID, tenantID)
		}
		if _, ok := byID[insight.InsightID]; ok {
			continue
		}
		insight.TenantID = tenantID
		existing = append(existing, copyInsightRecord(insight))
		byID[insight.InsightID] = insight
	}

	sort.SliceStable(existing, func(i, j int) bool {
		return existing[i].CreatedAt.Before(existing[j].CreatedAt)
	})

	return o.writeInsights(tenantID, existing)
}

func (o *fileInsightOperator) ListInsights(ctx context.Context, tenantID string) ([]InsightRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file insight operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	insights, err := o.readInsights(tenantID)
	if err != nil {
		return nil, err
	}
	out := make([]InsightRecord, len(insights))
	for i := range insights {
		out[i] = copyInsightRecord(insights[i])
	}
	return out, nil
}

func (o *fileInsightOperator) SaveRun(ctx context.Context, tenantID string, run ConsolidationRunRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("file insight operator: context cancelled: %w", err)
	}
	if tenantID == "" {
		return fmt.Errorf("file insight operator: tenantID is required")
	}
	if run.RunID == "" {
		return fmt.Errorf("file insight operator: runID is required")
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	runs, err := o.readRuns(tenantID)
	if err != nil {
		return err
	}
	for _, existing := range runs {
		if existing.RunID == run.RunID {
			return nil
		}
	}
	run.TenantID = tenantID
	runs = append(runs, copyConsolidationRunRecord(run))
	sort.SliceStable(runs, func(i, j int) bool { return runs[i].CreatedAt.Before(runs[j].CreatedAt) })
	return o.writeRuns(tenantID, runs)
}

func (o *fileInsightOperator) ListRuns(ctx context.Context, tenantID string) ([]ConsolidationRunRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file insight operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	runs, err := o.readRuns(tenantID)
	if err != nil {
		return nil, err
	}
	out := make([]ConsolidationRunRecord, len(runs))
	for i := range runs {
		out[i] = copyConsolidationRunRecord(runs[i])
	}
	return out, nil
}

func (o *fileInsightOperator) GetRunByIdempotencyKey(ctx context.Context, tenantID, idempotencyKey string) (*ConsolidationRunRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file insight operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	runs, err := o.readRuns(tenantID)
	if err != nil {
		return nil, err
	}
	var match *ConsolidationRunRecord
	for i := range runs {
		run := runs[i]
		if run.IdempotencyKey != idempotencyKey {
			continue
		}
		copy := copyConsolidationRunRecord(run)
		match = &copy
	}
	return match, nil
}

func (o *fileInsightOperator) insightsPath(tenantID string) string {
	return filepath.Join(o.rootDir, tenantID, "insights", "records.json")
}

func (o *fileInsightOperator) runsPath(tenantID string) string {
	return filepath.Join(o.rootDir, tenantID, "insights", "runs.json")
}

func (o *fileInsightOperator) readInsights(tenantID string) ([]InsightRecord, error) {
	data, err := os.ReadFile(o.insightsPath(tenantID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []InsightRecord{}, nil
		}
		return nil, fmt.Errorf("file insight operator: read insights file: %w", err)
	}

	var insights []InsightRecord
	if err := json.Unmarshal(data, &insights); err != nil {
		return nil, fmt.Errorf("file insight operator: unmarshal insights file: %w", err)
	}
	return insights, nil
}

func (o *fileInsightOperator) writeInsights(tenantID string, insights []InsightRecord) error {
	data, err := json.Marshal(insights)
	if err != nil {
		return fmt.Errorf("file insight operator: marshal insights file: %w", err)
	}
	path := o.insightsPath(tenantID)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("file insight operator: create insights dir: %w", err)
	}
	if err := atomicWriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("file insight operator: write insights file: %w", err)
	}
	return nil
}

func (o *fileInsightOperator) readRuns(tenantID string) ([]ConsolidationRunRecord, error) {
	data, err := os.ReadFile(o.runsPath(tenantID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []ConsolidationRunRecord{}, nil
		}
		return nil, fmt.Errorf("file insight operator: read runs file: %w", err)
	}

	var runs []ConsolidationRunRecord
	if err := json.Unmarshal(data, &runs); err != nil {
		return nil, fmt.Errorf("file insight operator: unmarshal runs file: %w", err)
	}
	return runs, nil
}

func (o *fileInsightOperator) writeRuns(tenantID string, runs []ConsolidationRunRecord) error {
	data, err := json.Marshal(runs)
	if err != nil {
		return fmt.Errorf("file insight operator: marshal runs file: %w", err)
	}
	path := o.runsPath(tenantID)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("file insight operator: create runs dir: %w", err)
	}
	if err := atomicWriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("file insight operator: write runs file: %w", err)
	}
	return nil
}

func copyInsightRecord(in InsightRecord) InsightRecord {
	in.Evidence = append([]string(nil), in.Evidence...)
	in.SessionIDs = append([]string(nil), in.SessionIDs...)
	in.TurnIDs = append([]string(nil), in.TurnIDs...)
	in.FileIDs = append([]string(nil), in.FileIDs...)
	in.ChunkIDs = append([]string(nil), in.ChunkIDs...)
	return in
}

func copyConsolidationRunRecord(in ConsolidationRunRecord) ConsolidationRunRecord {
	return in
}
