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
	jobs    chan vectorIndexJob
	pending map[string]map[string]bool
}

type vectorIndexJob struct {
	ctx       context.Context
	tenantID  string
	sessionID string
	docs      []*ai.Document
	messageID []string
}

const defaultVectorIndexQueueSize = 64

// NewVectorOperator creates a SessionOperator that composes a base session
// operator with a vector indexing backend.
func NewVectorOperator(base SessionOperator, backend VectorBackend, rootDir string) *VectorOperator {
	v := &VectorOperator{
		base:    base,
		backend: backend,
		rootDir: rootDir,
		jobs:    make(chan vectorIndexJob, defaultVectorIndexQueueSize),
		pending: make(map[string]map[string]bool),
	}
	go v.processVectorIndexJobs()
	return v
}

var _ SessionOperator = (*VectorOperator)(nil)

// SaveState persists tenant session state via the base operator, then indexes
// newly appended message text into the vector backend via a background queue.
func (v *VectorOperator) SaveState(ctx context.Context, tenantID, sessionID string, state SessionState) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("vector operator: context cancelled: %w", err)
	}

	if err := v.base.SaveState(ctx, tenantID, sessionID, state); err != nil {
		return err
	}

	v.mu.Lock()

	indexed, err := v.loadIndexedIDs(tenantID, sessionID)
	if err != nil {
		v.mu.Unlock()
		return err
	}
	pending := v.pendingMessageIDs(tenantID, sessionID)

	var newDocs []*ai.Document
	var newIDs []string

	for i := range state.Messages {
		msg := &state.Messages[i]
		if msg.MessageID == "" {
			continue
		}
		if indexed[msg.MessageID] || pending[msg.MessageID] {
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
		v.mu.Unlock()
		return nil
	}

	for _, id := range newIDs {
		pending[id] = true
	}
	v.mu.Unlock()

	job := vectorIndexJob{
		ctx:       context.WithoutCancel(ctx),
		tenantID:  tenantID,
		sessionID: sessionID,
		docs:      newDocs,
		messageID: newIDs,
	}

	select {
	case v.jobs <- job:
		return nil
	default:
		v.clearPending(job.tenantID, job.sessionID, job.messageID)
		slog.WarnContext(ctx, "vector indexing queue is full, messages will be retried on next save",
			"sessionID", sessionID,
			"messageCount", len(newDocs),
		)
		return nil
	}

}

func (v *VectorOperator) processVectorIndexJobs() {
	for job := range v.jobs {
		if err := v.backend.Index(job.ctx, job.tenantID, job.sessionID, job.docs); err != nil {
			slog.WarnContext(job.ctx, "vector indexing failed, messages will be retried on next save",
				"sessionID", job.sessionID,
				"messageCount", len(job.docs),
				"error", err,
			)
			v.clearPending(job.tenantID, job.sessionID, job.messageID)
			continue
		}

		v.mu.Lock()
		indexed, err := v.loadIndexedIDs(job.tenantID, job.sessionID)
		if err != nil {
			v.mu.Unlock()
			slog.WarnContext(job.ctx, "load indexed IDs after vector indexing failed",
				"sessionID", job.sessionID,
				"error", err,
			)
			v.clearPending(job.tenantID, job.sessionID, job.messageID)
			continue
		}

		for _, id := range job.messageID {
			indexed[id] = true
		}
		err = v.saveIndexedIDs(job.tenantID, job.sessionID, indexed)
		v.mu.Unlock()
		if err != nil {
			slog.WarnContext(job.ctx, "save indexed IDs after vector indexing failed",
				"sessionID", job.sessionID,
				"error", err,
			)
			v.clearPending(job.tenantID, job.sessionID, job.messageID)
			continue
		}

		v.clearPending(job.tenantID, job.sessionID, job.messageID)
	}
}

func (v *VectorOperator) pendingMessageIDs(tenantID, sessionID string) map[string]bool {
	key := v.pendingSessionKey(tenantID, sessionID)
	ids, ok := v.pending[key]
	if !ok {
		ids = make(map[string]bool)
		v.pending[key] = ids
	}
	return ids
}

func (v *VectorOperator) clearPending(tenantID, sessionID string, messageIDs []string) {
	v.mu.Lock()
	defer v.mu.Unlock()

	key := v.pendingSessionKey(tenantID, sessionID)
	ids, ok := v.pending[key]
	if !ok {
		return
	}
	for _, id := range messageIDs {
		delete(ids, id)
	}
	if len(ids) == 0 {
		delete(v.pending, key)
	}
}

func (v *VectorOperator) pendingSessionKey(tenantID, sessionID string) string {
	return tenantID + "/" + sessionID
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

	if err := v.backend.Delete(ctx, tenantID, sessionID); err != nil {
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
	return v.SearchSession(ctx, tenantID, sessionID, query, topK)
}

// SearchSession retrieves semantically similar messages scoped to one tenant
// session.
func (v *VectorOperator) SearchSession(ctx context.Context, tenantID, sessionID, query string, topK int) ([]SessionMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("vector operator: context cancelled: %w", err)
	}

	if topK <= 0 {
		return nil, nil
	}

	docs, err := v.backend.RetrieveSession(ctx, tenantID, sessionID, query, topK)
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

// SearchTenant retrieves semantically similar messages across all sessions in a
// tenant.
func (v *VectorOperator) SearchTenant(ctx context.Context, tenantID, query string, topK int) ([]SessionMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("vector operator: context cancelled: %w", err)
	}

	if topK <= 0 {
		return nil, nil
	}

	docs, err := v.backend.RetrieveTenant(ctx, tenantID, query, topK)
	if err != nil {
		return nil, fmt.Errorf("vector backend retrieve tenant: %w", err)
	}
	if len(docs) == 0 {
		return nil, nil
	}

	sessionIDs, err := v.base.ListSessions(ctx, tenantID)
	if err != nil {
		return nil, fmt.Errorf("list tenant sessions: %w", err)
	}

	msgByID := make(map[string]SessionMessage)
	for _, sid := range sessionIDs {
		state, err := v.base.LoadState(ctx, tenantID, sid, All, 0)
		if err != nil {
			return nil, fmt.Errorf("load session state %q: %w", sid, err)
		}
		if state == nil {
			continue
		}
		for _, msg := range state.Messages {
			msgByID[msg.MessageID] = msg
		}
	}

	var results []SessionMessage
	for _, doc := range docs {
		messageID, _ := doc.Metadata["messageID"].(string)
		msg, ok := msgByID[messageID]
		if !ok {
			continue
		}
		results = append(results, msg)
		if len(results) >= topK {
			break
		}
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
