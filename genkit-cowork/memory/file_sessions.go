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
	"os"
	"path/filepath"
	"sort"
	"sync"
)

type fileSessionOperator struct {
	rootDir  string
	tenantID string
	mu       sync.Mutex
	locks    map[string]*sync.RWMutex
}

// NewFileSessionOperator returns a SessionOperator that persists session state
// as JSON files under rootDir/{tenantID}/{sessionID}/state.json.
//
// The directory is created lazily on first write. The tenantID argument is
// retained for constructor parity but persistence methods are tenant-scoped by
// their tenantID parameter.
func NewFileSessionOperator(rootDir, tenantID string) SessionOperator {
	return &fileSessionOperator{
		rootDir:  rootDir,
		tenantID: tenantID,
		locks:    make(map[string]*sync.RWMutex),
	}
}

// sessionLock returns the per-session RWMutex, creating one if needed.
func (f *fileSessionOperator) sessionLock(sessionID string) *sync.RWMutex {
	f.mu.Lock()
	defer f.mu.Unlock()

	lk, ok := f.locks[sessionID]
	if !ok {
		lk = &sync.RWMutex{}
		f.locks[sessionID] = lk
	}
	return lk
}

func (f *fileSessionOperator) statePath(tenantID, sessionID string) string {
	return filepath.Join(f.rootDir, tenantID, sessionID, "state.json")
}

// SaveState writes complete session state to disk for a tenant session.
func (f *fileSessionOperator) SaveState(ctx context.Context, tenantID, sessionID string, state SessionState) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("file operator: context cancelled: %w", err)
	}

	lk := f.sessionLock(sessionID)
	lk.Lock()
	defer lk.Unlock()

	dir := filepath.Join(f.rootDir, tenantID, sessionID)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("file operator: create session dir: %w", err)
	}

	// Append-only validation: if a state file already exists, verify that
	// the new state does not drop existing messages.
	existing, err := f.readStateFile(tenantID, sessionID)
	if err != nil {
		return fmt.Errorf("file operator: read existing state: %w", err)
	}
	if existing != nil {
		if err := validateAppendOnlyState(*existing, state); err != nil {
			return fmt.Errorf("file operator: %w", err)
		}
		// Tenant consistency: reject writes that would change the tenant owner.
		if existing.TenantID != "" && state.TenantID != existing.TenantID {
			return fmt.Errorf(
				"file operator: tenant mismatch: session owned by %q, attempted write from %q",
				existing.TenantID, state.TenantID,
			)
		}
	}

	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("file operator: marshal state: %w", err)
	}

	if err := atomicWriteFile(f.statePath(tenantID, sessionID), data, 0644); err != nil {
		return fmt.Errorf("file operator: write state file: %w", err)
	}

	return nil
}

// LoadState reads tenant session state from disk and applies load-time message
// filtering based on persistence mode.
func (f *fileSessionOperator) LoadState(ctx context.Context, tenantID, sessionID string, mode PersistenceMode, nMessages int) (*SessionState, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file operator: context cancelled: %w", err)
	}

	lk := f.sessionLock(sessionID)
	lk.RLock()
	defer lk.RUnlock()

	state, err := f.readStateFile(tenantID, sessionID)
	if err != nil {
		return nil, fmt.Errorf("file operator: read state file: %w", err)
	}
	if state == nil {
		return nil, nil
	}

	filtered := filterMessages(state.Messages, mode, nMessages)
	return &SessionState{
		TenantID:          state.TenantID,
		Messages:          filtered,
		Turns:             copyTurnRecords(state.Turns),
		Snapshots:         copyStateSnapshots(state.Snapshots),
		Assets:            copySessionAssets(state.Assets),
		LastConsolidateAt: state.LastConsolidateAt,
	}, nil
}

// DeleteSession removes all persisted files for a tenant session.
func (f *fileSessionOperator) DeleteSession(ctx context.Context, tenantID, sessionID string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("file operator: context cancelled: %w", err)
	}

	lk := f.sessionLock(sessionID)
	lk.Lock()
	defer lk.Unlock()

	dir := filepath.Join(f.rootDir, tenantID, sessionID)
	if err := os.RemoveAll(dir); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("file operator: delete session dir: %w", err)
	}

	// Clean up the per-session lock entry.
	f.mu.Lock()
	delete(f.locks, sessionID)
	f.mu.Unlock()

	return nil
}

// ListSessions lists all session IDs for tenantID.
//
// If tenantID has no persisted sessions yet, ListSessions returns an empty
// list and a nil error.
func (f *fileSessionOperator) ListSessions(ctx context.Context, tenantID string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file operator: context cancelled: %w", err)
	}

	dir := filepath.Join(f.rootDir, tenantID)
	sessionDirs, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []string{}, nil
		}
		return nil, fmt.Errorf("file operator: read tenant dir: %w", err)
	}
	sessionIDs := make([]string, 0, len(sessionDirs))
	for _, entry := range sessionDirs {
		if entry.IsDir() {
			sessionIDs = append(sessionIDs, entry.Name())
		}
	}
	sort.Strings(sessionIDs)
	return sessionIDs, nil
}

// readStateFile reads and unmarshals the state file without locking.
// Caller must hold the appropriate lock.
func (f *fileSessionOperator) readStateFile(tenantID, sessionID string) (*SessionState, error) {
	data, err := os.ReadFile(f.statePath(tenantID, sessionID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}

	var state SessionState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("unmarshal state: %w", err)
	}
	return &state, nil
}

// atomicWriteFile writes data to a temporary file in the same directory then
// renames it to the target path, ensuring the file is never left in a partial state.
func atomicWriteFile(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	tmpName := tmp.Name()

	// Clean up on any failure path.
	defer func() {
		if tmpName != "" {
			os.Remove(tmpName)
		}
	}()

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write temp file: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("sync temp file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close temp file: %w", err)
	}

	if err := os.Chmod(tmpName, perm); err != nil {
		return fmt.Errorf("chmod temp file: %w", err)
	}

	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("rename temp file: %w", err)
	}

	// Rename succeeded — prevent deferred cleanup from removing the target.
	tmpName = ""
	return nil
}
