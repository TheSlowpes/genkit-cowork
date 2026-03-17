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

type VectorOperator struct {
	base    SessionOperator
	backend VectorBackend
	rootDir string
	mu      sync.Mutex
}

func NewVectorOperator(base SessionOperator, backend VectorBackend, rootDir string) *VectorOperator {
	return &VectorOperator{
		base:    base,
		backend: backend,
		rootDir: rootDir,
	}
}

var _ SessionOperator = (*VectorOperator)(nil)

func (v *VectorOperator) SaveState(ctx context.Context, sessionID string, state SessionState) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("vector operator: context cancelled: %w", err)
	}

	if err := v.base.SaveState(ctx, sessionID, state); err != nil {
		return err
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	indexed, err := v.loadIndexedIDs(sessionID)
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
	if err := v.saveIndexedIDs(sessionID, indexed); err != nil {
		return fmt.Errorf("save indexed IDs: %w", err)
	}

	return nil
}

func (v *VectorOperator) LoadState(ctx context.Context, sessionID string, mode PersistenceMode, nMessages int) (*SessionState, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("vector operator: context cancelled: %w", err)
	}

	return v.base.LoadState(ctx, sessionID, mode, nMessages)
}

func (v *VectorOperator) DeleteSession(ctx context.Context, sessionID string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("vector operator: context cancelled: %w", err)
	}

	if err := v.base.DeleteSession(ctx, sessionID); err != nil {
		return err
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	if err := v.backend.Delete(ctx, sessionID); err != nil {
		return fmt.Errorf("vector backend delete: %w", err)
	}

	path := v.indexedIDsPath(sessionID)
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove indexed IDs file: %w", err)
	}
	return nil
}

func (v *VectorOperator) Search(ctx context.Context, sessionID, query string, topK int) ([]SessionMessage, error) {
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

	state, err := v.base.LoadState(ctx, sessionID, All, 0)
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

func (v *VectorOperator) indexedIDsPath(sessionID string) string {
	return filepath.Join(v.rootDir, sessionID, "indexed_ids.json")
}

func (v *VectorOperator) loadIndexedIDs(sessionID string) (map[string]bool, error) {
	data, err := os.ReadFile(v.indexedIDsPath(sessionID))
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

func (v *VectorOperator) saveIndexedIDs(sessionID string, indexed map[string]bool) error {
	ids := make([]string, 0, len(indexed))
	for id := range indexed {
		ids = append(ids, id)
	}

	data, err := json.Marshal(ids)
	if err != nil {
		return fmt.Errorf("marshal indexed_ids.json: %w", err)
	}

	dir := filepath.Join(v.rootDir, sessionID)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create session dir: %w", err)
	}

	if err := atomicWriteFile(v.indexedIDsPath(sessionID), data, 0644); err != nil {
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
