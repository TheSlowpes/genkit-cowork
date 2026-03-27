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
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
)

// VectorOperator composes a base SessionOperator with vector indexing and
// semantic retrieval capabilities.
type VectorOperator struct {
	base    SessionOperator
	backend VectorBackend
	rootDir string
	mu      sync.Mutex
}

// NewVectorOperator creates a SessionOperator that composes a base session
// operator with a vector indexing backend.
//
// The tenantID argument is retained for constructor parity but tenant scoping
// is enforced by per-method tenantID parameters.
func NewVectorOperator(base SessionOperator, backend VectorBackend, rootDir, tenantID string) *VectorOperator {
	return &VectorOperator{
		base:    base,
		backend: backend,
		rootDir: rootDir,
	}
}

var _ SessionOperator = (*VectorOperator)(nil)

// SaveState persists tenant session state via the base operator, then indexes
// newly appended message text into the vector backend.
func (v *VectorOperator) SaveState(ctx context.Context, tenantID, sessionID string, state SessionState) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("vector operator: context cancelled: %w", err)
	}

	if err := v.base.SaveState(ctx, tenantID, sessionID, state); err != nil {
		return err
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	indexed, err := v.loadIndexedIDs(tenantID, sessionID)
	if err != nil {
		return err
	}

	var newDocs []*ai.Document
	var newIDs []string

	for i := range state.Messages {
		msg := &state.Messages[i]
		if msg.MessageID == "" {
			continue
		}
		if indexed[msg.MessageID] {
			continue
		}

		text := messageText(msg)
		if text == "" {
			continue
		}

		doc := ai.DocumentFromText(text, map[string]any{
			"messageID": msg.MessageID,
			"sessionID": sessionID,
			"kind":      string(msg.Kind),
			"origin":    string(msg.Origin),
		})
		newDocs = append(newDocs, doc)
		newIDs = append(newIDs, msg.MessageID)
	}

	if len(newDocs) == 0 {
		return nil
	}

	if err := v.backend.Index(ctx, sessionID, newDocs); err != nil {
		slog.WarnContext(ctx, "vector indexing failed, messages will be retried on next save",
			"sessionID", sessionID,
			"messageCount", len(newDocs),
			"error", err,
		)
		return nil
	}

	for _, id := range newIDs {
		indexed[id] = true
	}
	if err := v.saveIndexedIDs(tenantID, sessionID, indexed); err != nil {
		return fmt.Errorf("save indexed IDs: %w", err)
	}

	return nil
}

// LoadState delegates session loading to the composed base operator.
func (v *VectorOperator) LoadState(ctx context.Context, tenantID, sessionID string, mode PersistenceMode, nMessages int) (*SessionState, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("vector operator: context cancelled: %w", err)
	}

	return v.base.LoadState(ctx, tenantID, sessionID, mode, nMessages)
}

// DeleteSession removes canonical session state, vector entries, and local
// indexing metadata for the session.
func (v *VectorOperator) DeleteSession(ctx context.Context, tenantID, sessionID string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("vector operator: context cancelled: %w", err)
	}

	if err := v.base.DeleteSession(ctx, tenantID, sessionID); err != nil {
		return err
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	if err := v.backend.Delete(ctx, sessionID); err != nil {
		return fmt.Errorf("vector backend delete: %w", err)
	}

	path := v.indexedIDsPath(tenantID, sessionID)
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove indexed IDs file: %w", err)
	}
	return nil
}

// Search retrieves semantically similar messages for a tenant session.
func (v *VectorOperator) Search(ctx context.Context, tenantID, sessionID, query string, topK int) ([]SessionMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("vector operator: context cancelled: %w", err)
	}

	if topK <= 0 {
		return nil, nil
	}

	docs, err := v.backend.Retrieve(ctx, sessionID, query, topK)
	if err != nil {
		return nil, fmt.Errorf("vector backend retrieve: %w", err)
	}
	if len(docs) == 0 {
		return nil, nil
	}

	state, err := v.base.LoadState(ctx, tenantID, sessionID, All, 0)
	if err != nil {
		return nil, fmt.Errorf("load session state: %w", err)
	}
	if state == nil {
		return nil, nil
	}

	msgMap := make(map[string]SessionMessage, len(state.Messages))
	for _, msg := range state.Messages {
		msgMap[msg.MessageID] = msg
	}

	var results []SessionMessage
	for _, doc := range docs {
		messageID, _ := doc.Metadata["messageID"].(string)
		msg, ok := msgMap[messageID]
		if !ok {
			continue
		}
		results = append(results, msg)
	}
	return results, nil
}

// ListSessions delegates tenant session listing to the composed base operator.
func (v *VectorOperator) ListSessions(ctx context.Context, tenantID string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("vector operator: context cancelled: %w", err)
	}
	return v.base.ListSessions(ctx, tenantID)
}

func (v *VectorOperator) indexedIDsPath(tenantID, sessionID string) string {
	return filepath.Join(v.rootDir, tenantID, sessionID, "indexed_ids.json")
}

func (v *VectorOperator) loadIndexedIDs(tenantID, sessionID string) (map[string]bool, error) {
	data, err := os.ReadFile(v.indexedIDsPath(tenantID, sessionID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return make(map[string]bool), nil
		}
		return nil, fmt.Errorf("read indexed_ids.json: %w", err)
	}

	var ids []string
	if err := json.Unmarshal(data, &ids); err != nil {
		return nil, fmt.Errorf("unmarshal indexed_ids.json: %w", err)
	}

	result := make(map[string]bool, len(ids))
	for _, id := range ids {
		result[id] = true
	}
	return result, nil
}

func (v *VectorOperator) saveIndexedIDs(tenantID, sessionID string, indexed map[string]bool) error {
	ids := make([]string, 0, len(indexed))
	for id := range indexed {
		ids = append(ids, id)
	}

	data, err := json.Marshal(ids)
	if err != nil {
		return fmt.Errorf("marshal indexed_ids.json: %w", err)
	}

	dir := filepath.Join(v.rootDir, tenantID, sessionID)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create session dir: %w", err)
	}

	if err := atomicWriteFile(v.indexedIDsPath(tenantID, sessionID), data, 0644); err != nil {
		return fmt.Errorf("write indexed_ids.json: %w", err)
	}
	return nil
}

func messageText(msg *SessionMessage) string {
	var b strings.Builder
	for _, part := range msg.Content.Content {
		if part.IsText() {
			if b.Len() > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(part.Text)
		}
	}
	return b.String()
}
