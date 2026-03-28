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
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// PreferenceSource identifies how a preference was created.
type PreferenceSource string

const (
	PreferenceSourceExplicit PreferenceSource = "explicit"
	PreferenceSourceImplicit PreferenceSource = "implicit"
)

// PreferenceStatus indicates whether a preference is active.
type PreferenceStatus string

const (
	PreferenceStatusActive   PreferenceStatus = "active"
	PreferenceStatusArchived PreferenceStatus = "archived"
)

// PreferenceRecord stores one tenant-scoped preference.
type PreferenceRecord struct {
	PreferenceID     string           `json:"preferenceID"`
	TenantID         string           `json:"tenantID"`
	Key              string           `json:"key"`
	Value            string           `json:"value"`
	Source           PreferenceSource `json:"source"`
	Confidence       float64          `json:"confidence"`
	Evidence         []string         `json:"evidence,omitempty"`
	SessionIDs       []string         `json:"sessionIDs,omitempty"`
	TurnIDs          []string         `json:"turnIDs,omitempty"`
	FileIDs          []string         `json:"fileIDs,omitempty"`
	ChunkIDs         []string         `json:"chunkIDs,omitempty"`
	SourceInsightID  string           `json:"sourceInsightID,omitempty"`
	SourceInsightKey string           `json:"sourceInsightKey,omitempty"`
	Status           PreferenceStatus `json:"status"`
	CreatedAt        time.Time        `json:"createdAt"`
	UpdatedAt        time.Time        `json:"updatedAt"`
}

// PreferenceFilter defines tenant preference query filters.
type PreferenceFilter struct {
	Source PreferenceSource
	Status PreferenceStatus
	Key    string
}

// PreferenceOperator persists and queries tenant-scoped preferences.
type PreferenceOperator interface {
	SavePreference(ctx context.Context, tenantID string, preference PreferenceRecord) (PreferenceRecord, error)
	LoadPreference(ctx context.Context, tenantID, preferenceID string) (*PreferenceRecord, error)
	ListPreferences(ctx context.Context, tenantID string, filter PreferenceFilter) ([]PreferenceRecord, error)
	DeletePreference(ctx context.Context, tenantID, preferenceID string) error
}

// DefaultPreferenceOperator is the in-memory PreferenceOperator implementation.
type DefaultPreferenceOperator struct {
	mu          sync.Mutex
	preferences map[string]map[string]PreferenceRecord
}

// NewDefaultPreferenceOperator creates an in-memory preference operator.
func NewDefaultPreferenceOperator() *DefaultPreferenceOperator {
	return &DefaultPreferenceOperator{preferences: make(map[string]map[string]PreferenceRecord)}
}

var _ PreferenceOperator = (*DefaultPreferenceOperator)(nil)

// SavePreference inserts or updates a tenant preference record.
func (o *DefaultPreferenceOperator) SavePreference(ctx context.Context, tenantID string, preference PreferenceRecord) (PreferenceRecord, error) {
	if err := ctx.Err(); err != nil {
		return PreferenceRecord{}, fmt.Errorf("preference operator: context cancelled: %w", err)
	}
	if strings.TrimSpace(tenantID) == "" {
		return PreferenceRecord{}, fmt.Errorf("preference operator: tenantID is required")
	}

	normalized := normalizePreferenceRecord(preference)
	if normalized.PreferenceID == "" {
		normalized.PreferenceID = uuid.New().String()
	}
	if normalized.TenantID != "" && normalized.TenantID != tenantID {
		return PreferenceRecord{}, fmt.Errorf("preference operator: tenant mismatch: got %q want %q", normalized.TenantID, tenantID)
	}
	normalized.TenantID = tenantID
	if normalized.CreatedAt.IsZero() {
		normalized.CreatedAt = time.Now().UTC()
	}
	normalized.UpdatedAt = time.Now().UTC()

	o.mu.Lock()
	defer o.mu.Unlock()

	if _, ok := o.preferences[tenantID]; !ok {
		o.preferences[tenantID] = make(map[string]PreferenceRecord)
	}
	existing, exists := o.preferences[tenantID][normalized.PreferenceID]
	if exists {
		normalized.CreatedAt = existing.CreatedAt
	}
	o.preferences[tenantID][normalized.PreferenceID] = normalized

	return copyPreferenceRecord(normalized), nil
}

// LoadPreference loads one tenant preference by ID.
func (o *DefaultPreferenceOperator) LoadPreference(ctx context.Context, tenantID, preferenceID string) (*PreferenceRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("preference operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	tenantPrefs, ok := o.preferences[tenantID]
	if !ok {
		return nil, nil
	}
	record, ok := tenantPrefs[preferenceID]
	if !ok {
		return nil, nil
	}
	copy := copyPreferenceRecord(record)
	return &copy, nil
}

// ListPreferences lists tenant preferences with optional filters.
func (o *DefaultPreferenceOperator) ListPreferences(ctx context.Context, tenantID string, filter PreferenceFilter) ([]PreferenceRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("preference operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	tenantPrefs, ok := o.preferences[tenantID]
	if !ok {
		return []PreferenceRecord{}, nil
	}

	keys := slices.Collect(maps.Keys(tenantPrefs))
	sort.Strings(keys)

	out := make([]PreferenceRecord, 0, len(keys))
	for _, id := range keys {
		record := tenantPrefs[id]
		if !matchesPreferenceFilter(record, filter) {
			continue
		}
		out = append(out, copyPreferenceRecord(record))
	}
	return out, nil
}

// DeletePreference removes one tenant preference.
func (o *DefaultPreferenceOperator) DeletePreference(ctx context.Context, tenantID, preferenceID string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("preference operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	delete(o.preferences[tenantID], preferenceID)
	return nil
}

type filePreferenceOperator struct {
	rootDir string
	mu      sync.Mutex
}

// NewFilePreferenceOperator creates a file-backed preference operator.
func NewFilePreferenceOperator(rootDir string) PreferenceOperator {
	return &filePreferenceOperator{rootDir: rootDir}
}

var _ PreferenceOperator = (*filePreferenceOperator)(nil)

func (o *filePreferenceOperator) SavePreference(ctx context.Context, tenantID string, preference PreferenceRecord) (PreferenceRecord, error) {
	if err := ctx.Err(); err != nil {
		return PreferenceRecord{}, fmt.Errorf("file preference operator: context cancelled: %w", err)
	}
	if strings.TrimSpace(tenantID) == "" {
		return PreferenceRecord{}, fmt.Errorf("file preference operator: tenantID is required")
	}

	normalized := normalizePreferenceRecord(preference)
	if normalized.PreferenceID == "" {
		normalized.PreferenceID = uuid.New().String()
	}
	if normalized.TenantID != "" && normalized.TenantID != tenantID {
		return PreferenceRecord{}, fmt.Errorf("file preference operator: tenant mismatch: got %q want %q", normalized.TenantID, tenantID)
	}
	normalized.TenantID = tenantID

	o.mu.Lock()
	defer o.mu.Unlock()

	all, err := o.readPreferences(tenantID)
	if err != nil {
		return PreferenceRecord{}, err
	}

	if normalized.CreatedAt.IsZero() {
		normalized.CreatedAt = time.Now().UTC()
	}
	normalized.UpdatedAt = time.Now().UTC()

	for i := range all {
		if all[i].PreferenceID != normalized.PreferenceID {
			continue
		}
		normalized.CreatedAt = all[i].CreatedAt
		all[i] = normalized
		if err := o.writePreferences(tenantID, all); err != nil {
			return PreferenceRecord{}, err
		}
		return copyPreferenceRecord(normalized), nil
	}

	all = append(all, normalized)
	sort.SliceStable(all, func(i, j int) bool {
		return all[i].CreatedAt.Before(all[j].CreatedAt)
	})
	if err := o.writePreferences(tenantID, all); err != nil {
		return PreferenceRecord{}, err
	}
	return copyPreferenceRecord(normalized), nil
}

func (o *filePreferenceOperator) LoadPreference(ctx context.Context, tenantID, preferenceID string) (*PreferenceRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file preference operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	all, err := o.readPreferences(tenantID)
	if err != nil {
		return nil, err
	}
	for i := range all {
		if all[i].PreferenceID == preferenceID {
			copy := copyPreferenceRecord(all[i])
			return &copy, nil
		}
	}
	return nil, nil
}

func (o *filePreferenceOperator) ListPreferences(ctx context.Context, tenantID string, filter PreferenceFilter) ([]PreferenceRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file preference operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	all, err := o.readPreferences(tenantID)
	if err != nil {
		return nil, err
	}
	out := make([]PreferenceRecord, 0, len(all))
	for _, record := range all {
		if !matchesPreferenceFilter(record, filter) {
			continue
		}
		out = append(out, copyPreferenceRecord(record))
	}
	return out, nil
}

func (o *filePreferenceOperator) DeletePreference(ctx context.Context, tenantID, preferenceID string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("file preference operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	all, err := o.readPreferences(tenantID)
	if err != nil {
		return err
	}
	out := make([]PreferenceRecord, 0, len(all))
	for _, record := range all {
		if record.PreferenceID == preferenceID {
			continue
		}
		out = append(out, record)
	}
	return o.writePreferences(tenantID, out)
}

func (o *filePreferenceOperator) prefsPath(tenantID string) string {
	return filepath.Join(o.rootDir, tenantID, "preferences", "records.json")
}

func (o *filePreferenceOperator) readPreferences(tenantID string) ([]PreferenceRecord, error) {
	data, err := os.ReadFile(o.prefsPath(tenantID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []PreferenceRecord{}, nil
		}
		return nil, fmt.Errorf("file preference operator: read preferences file: %w", err)
	}

	var records []PreferenceRecord
	if err := json.Unmarshal(data, &records); err != nil {
		return nil, fmt.Errorf("file preference operator: unmarshal preferences file: %w", err)
	}
	return records, nil
}

func (o *filePreferenceOperator) writePreferences(tenantID string, records []PreferenceRecord) error {
	data, err := json.Marshal(records)
	if err != nil {
		return fmt.Errorf("file preference operator: marshal preferences file: %w", err)
	}
	path := o.prefsPath(tenantID)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("file preference operator: create preferences dir: %w", err)
	}
	if err := atomicWriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("file preference operator: write preferences file: %w", err)
	}
	return nil
}

func normalizePreferenceRecord(in PreferenceRecord) PreferenceRecord {
	in.Key = strings.TrimSpace(in.Key)
	in.Value = strings.TrimSpace(in.Value)
	if in.Source == "" {
		in.Source = PreferenceSourceExplicit
	}
	if in.Status == "" {
		in.Status = PreferenceStatusActive
	}
	if in.Confidence < 0 {
		in.Confidence = 0
	}
	if in.Confidence > 1 {
		in.Confidence = 1
	}
	in.Evidence = normalizeStringList(in.Evidence)
	in.SessionIDs = normalizeStringList(in.SessionIDs)
	in.TurnIDs = normalizeStringList(in.TurnIDs)
	in.FileIDs = normalizeStringList(in.FileIDs)
	in.ChunkIDs = normalizeStringList(in.ChunkIDs)
	return in
}

func matchesPreferenceFilter(record PreferenceRecord, filter PreferenceFilter) bool {
	if filter.Source != "" && record.Source != filter.Source {
		return false
	}
	if filter.Status != "" && record.Status != filter.Status {
		return false
	}
	if strings.TrimSpace(filter.Key) != "" && !strings.EqualFold(record.Key, strings.TrimSpace(filter.Key)) {
		return false
	}
	return true
}

func copyPreferenceRecord(in PreferenceRecord) PreferenceRecord {
	in.Evidence = append([]string(nil), in.Evidence...)
	in.SessionIDs = append([]string(nil), in.SessionIDs...)
	in.TurnIDs = append([]string(nil), in.TurnIDs...)
	in.FileIDs = append([]string(nil), in.FileIDs...)
	in.ChunkIDs = append([]string(nil), in.ChunkIDs...)
	return in
}
