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
	"encoding/base64"
	"fmt"
	"maps"
	"net/url"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/google/uuid"
)

// SessionMessage is a stored conversation message with provenance metadata.
type SessionMessage struct {
	// MessageID is the unique message identifier.
	MessageID string `json:"messageID"`
	// Sequence is the monotonic position of this message within the session.
	Sequence int64 `json:"sequence,omitempty"`
	// Origin identifies where the message came from.
	Origin MessageOrigin `json:"origin"`
	Kind   MessageKind   `json:"kind"`
	// Content is the full Genkit message payload.
	Content ai.Message `json:"content"`
	// Timestamp is when the message was created.
	Timestamp time.Time `json:"timestamp"`
}

// SessionState is the persisted state stored per session.
type SessionState struct {
	TenantID          string           `json:"tenantID"`
	Messages          []SessionMessage `json:"messages"`
	Turns             []TurnRecord     `json:"turns,omitempty"`
	Snapshots         []StateSnapshot  `json:"snapshots,omitempty"`
	Assets            []SessionAsset   `json:"assets,omitempty"`
	LastConsolidateAt time.Time        `json:"lastConsolidateAt"`
}

// SessionOperator abstracts the storage backend.
type SessionOperator interface {
	// SaveState persists the full session state. Always called, regardless of mode.
	SaveState(ctx context.Context, tenantID, sessionID string, state SessionState) error

	// LoadState retrieves full session state.
	// The mode and nMessages parameters control pruning at load time.
	LoadState(ctx context.Context, tenantID, sessionID string, mode PersistenceMode, nMessages int) (*SessionState, error)

	// DeleteSession removes all messages for a session.
	DeleteSession(ctx context.Context, tenantID, sessionID string) error

	// ListSessions lists all session IDs for tenantID.
	//
	// Implementations should return an empty list and a nil error when the
	// tenant has no sessions yet.
	ListSessions(ctx context.Context, tenantID string) ([]string, error)
}

var _ SessionOperator = (*defaultSessionOperator)(nil)

type defaultSessionOperator struct {
	mu    sync.Mutex
	store map[string]map[string]SessionState
}

func (o *defaultSessionOperator) SaveState(ctx context.Context, tenantID, sessionID string, state SessionState) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("default session operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	if o.store == nil {
		o.store = make(map[string]map[string]SessionState)
	}

	// Deep copy the messages slice to avoid shared references with the caller.
	msgs := make([]SessionMessage, len(state.Messages))
	copy(msgs, state.Messages)
	assets := make([]SessionAsset, len(state.Assets))
	copy(assets, state.Assets)
	turns := make([]TurnRecord, len(state.Turns))
	copy(turns, state.Turns)
	snapshots := make([]StateSnapshot, len(state.Snapshots))
	copy(snapshots, state.Snapshots)

	// Create tenant entry if it doesn't exist.
	if _, ok := o.store[tenantID]; !ok {
		o.store[tenantID] = make(map[string]SessionState)
	}

	if existing, ok := o.store[tenantID][sessionID]; ok {
		if err := validateAppendOnlyState(existing, state); err != nil {
			return fmt.Errorf("default session operator: %w", err)
		}
	}

	o.store[tenantID][sessionID] = SessionState{
		TenantID:          tenantID,
		Messages:          msgs,
		Turns:             turns,
		Snapshots:         snapshots,
		Assets:            assets,
		LastConsolidateAt: state.LastConsolidateAt,
	}
	return nil
}

func (o *defaultSessionOperator) LoadState(ctx context.Context, tenantID, sessionID string, mode PersistenceMode, nMessages int) (*SessionState, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("default session operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	state, ok := o.store[tenantID][sessionID]
	if !ok {
		return nil, nil
	}

	filtered := filterMessages(state.Messages, mode, nMessages)

	return &SessionState{
		TenantID:          tenantID,
		Messages:          filtered,
		Turns:             copyTurnRecords(state.Turns),
		Snapshots:         copyStateSnapshots(state.Snapshots),
		Assets:            copySessionAssets(state.Assets),
		LastConsolidateAt: state.LastConsolidateAt,
	}, nil
}

// filterMessages applies the persistence mode to select which messages to return.
// The underlying store always holds all messages; filtering happens only at load time.
func filterMessages(msgs []SessionMessage, mode PersistenceMode, n int) []SessionMessage {
	total := len(msgs)
	if total == 0 {
		return nil
	}

	switch mode {
	case SlidingWindow:
		if n <= 0 || n >= total {
			return copyMessages(msgs)
		}
		return copyMessages(msgs[total-n:])

	case TailEndsPruning:
		if n <= 0 || 2*n >= total {
			return copyMessages(msgs)
		}
		head := msgs[:n]
		tail := msgs[total-n:]
		result := make([]SessionMessage, 0, 2*n)
		result = append(result, head...)
		result = append(result, tail...)
		return result

	default: // All
		return copyMessages(msgs)
	}
}

func copyMessages(msgs []SessionMessage) []SessionMessage {
	out := make([]SessionMessage, len(msgs))
	copy(out, msgs)
	return out
}

func copySessionAssets(assets []SessionAsset) []SessionAsset {
	out := make([]SessionAsset, len(assets))
	copy(out, assets)
	return out
}

func copyTurnRecords(turns []TurnRecord) []TurnRecord {
	out := make([]TurnRecord, len(turns))
	copy(out, turns)
	return out
}

func copyStateSnapshots(snaps []StateSnapshot) []StateSnapshot {
	out := make([]StateSnapshot, len(snaps))
	copy(out, snaps)
	return out
}

func (o *defaultSessionOperator) DeleteSession(ctx context.Context, tenantID, sessionID string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("default session operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	delete(o.store[tenantID], sessionID)
	return nil
}

func (o *defaultSessionOperator) ListSessions(ctx context.Context, tenantID string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("default session operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	sessions, ok := o.store[tenantID]
	if !ok {
		return []string{}, nil
	}

	sessionIDs := slices.Collect(maps.Keys(sessions))
	sort.Strings(sessionIDs)
	return sessionIDs, nil
}

// MessageOrigin identifies the source channel for a message.
type MessageOrigin string

const (
	// ZoomMessage indicates a Zoom-originated user message.
	ZoomMessage MessageOrigin = "zoom"
	// UIMessage indicates an in-app UI-originated message.
	UIMessage MessageOrigin = "ui"
	// WhatsAppMessage indicates a WhatsApp-originated message.
	WhatsAppMessage MessageOrigin = "whatsapp"
	// EmailMessage indicates an email-originated message.
	EmailMessage MessageOrigin = "email"
	// ModelMessage indicates an assistant/model message.
	ModelMessage MessageOrigin = "model"
	// ToolMessage indicates a tool response message.
	ToolMessage MessageOrigin = "tool"
	// HeartbeatMessage indicates a heartbeat-originated message.
	HeartbeatMessage MessageOrigin = "heartbeat"
)

type MessageKind string

const (
	// KindEpisodic covers raw conversation turns: user input, model replies,
	// and heartbeat-initiated exchanges. These are the ground-truth record of
	// what was said and when.
	KindEpisodic MessageKind = "episodic"

	// KindSemantic covers consolidated insights written by the ConsolidationFlow.
	KindSemantic MessageKind = "semantic"

	// KindProcedural covers task execution patterns: sequences that succeeded,
	// sequences that failed, and what distinguished them. Written explicitly
	// by procedural logging paths.
	KindProcedural MessageKind = "procedural"

	// KindInstrumental covers tool call records: the tool name, input, output,
	// and success/failure outcome. Written by tool-result paths.
	KindInstrumental MessageKind = "instrumental"
)

// KindForMessage derives a default memory kind from message role.
func KindForMessage(role ai.Role) MessageKind {
	if role == ai.RoleTool {
		return KindInstrumental
	}
	return KindEpisodic
}

// SessionOption configures session store behavior.
type SessionOption func(*sessionOptions)

type sessionOptions struct {
	mode        PersistenceMode
	nMessages   int // used for SlidingWindow and TailEndsPruning modes
	tokenBudget int

	operator       SessionOperator
	assetStore     MediaAssetStore
	tenantID       string
	tokenEstimator TokenEstimator
}

// WithPersistenceMode sets the filtering mode used when loading messages from
// storage.
func WithPersistenceMode(mode PersistenceMode, n int) SessionOption {
	return func(opts *sessionOptions) {
		opts.mode = mode
		opts.nMessages = n
	}
}

// WithCustomSessionOperator injects a custom persistence backend.
func WithCustomSessionOperator(operator SessionOperator) SessionOption {
	return func(opts *sessionOptions) {
		opts.operator = operator
	}
}

// WithMediaAssetStore configures a media asset store used to persist data URI
// media parts and replace them with absolute file paths.
func WithMediaAssetStore(store MediaAssetStore) SessionOption {
	return func(opts *sessionOptions) {
		opts.assetStore = store
	}
}

// WithTenantID sets a fixed tenant ID for all session operations.
//
// Prefer ForTenant when tenant identity is request-scoped, and WithTenantID
// when a store instance is intentionally single-tenant.
func WithTenantID(tenantID string) SessionOption {
	return func(opts *sessionOptions) {
		opts.tenantID = tenantID
	}
}

// PersistenceMode controls how many messages are returned on load.
type PersistenceMode int

const (
	// All loads every message in the session.
	All PersistenceMode = iota
	// SlidingWindow loads only the last N messages.
	SlidingWindow
	// TailEndsPruning loads the first N and last N messages.
	TailEndsPruning
	// TokenBudget loads as much recent history as fits in a token budget.
	TokenBudget
)

// Session is a Genkit session store implementation backed by SessionOperator.
type Session struct {
	opts sessionOptions
}

// ForTenant returns a shallow copy of Session bound to tenantID.
//
// The returned store shares the same operator and options, but all
// Get/Save calls are scoped to the provided tenant.
func (s *Session) ForTenant(tenantID string) *Session {
	clone := *s
	clone.opts.tenantID = tenantID
	return &clone
}

// NewSession creates a new session store.
func NewSession(opts ...SessionOption) *Session {
	options := sessionOptions{
		mode:           All,
		operator:       &defaultSessionOperator{},
		tenantID:       "default",
		tokenEstimator: generationUsageTokenEstimator{},
	}
	for _, opt := range opts {
		opt(&options)
	}
	return &Session{opts: options}
}

var _ session.Store[SessionState] = (*Session)(nil)

// Get loads session state by ID.
func (s *Session) Get(ctx context.Context, sessionID string) (*session.Data[SessionState], error) {
	loadMode := s.opts.mode
	loadN := s.opts.nMessages
	if s.opts.mode == TokenBudget {
		loadMode = All
		loadN = 0
	}

	state, err := s.opts.operator.LoadState(ctx, s.opts.tenantID, sessionID, loadMode, loadN)
	if err != nil {
		return nil, err
	}
	if state == nil {
		return nil, nil
	}
	return &session.Data[SessionState]{
		ID:    sessionID,
		State: s.applyLoadPruning(*state),
	}, nil
}

func (s *Session) applyLoadPruning(state SessionState) SessionState {
	if s.opts.mode != TokenBudget {
		return state
	}

	pruned := state
	trimmed, err := applyTokenBudget(state.Messages, s.opts.tokenBudget, s.opts.tokenEstimator)
	if err != nil {
		return state
	}
	pruned.Messages = trimmed
	return pruned
}

// Save persists the provided session state, assigning IDs and timestamps when
// missing.
func (s *Session) Save(ctx context.Context, sessionID string, data *session.Data[SessionState]) error {
	if data == nil {
		return fmt.Errorf("save session: nil session data")
	}

	existing, err := s.opts.operator.LoadState(ctx, s.opts.tenantID, sessionID, All, 0)
	if err != nil {
		return fmt.Errorf("load existing state: %w", err)
	}

	existingMessages := 0
	existingTurns := 0
	existingSnapshots := 0
	lastMessageSequence := int64(0)
	lastTurnSequence := int64(0)
	lastSnapshotSequence := int64(0)
	if existing != nil {
		existingMessages = len(existing.Messages)
		existingTurns = len(existing.Turns)
		existingSnapshots = len(existing.Snapshots)
		if len(existing.Messages) > 0 {
			lastMessageSequence = existing.Messages[len(existing.Messages)-1].Sequence
		}
		if len(existing.Turns) > 0 {
			lastTurnSequence = existing.Turns[len(existing.Turns)-1].Sequence
		}
		if len(existing.Snapshots) > 0 {
			lastSnapshotSequence = existing.Snapshots[len(existing.Snapshots)-1].Sequence
		}

	}

	if len(data.State.Messages) < existingMessages {
		return fmt.Errorf("save session: append-only violation: new state has %d messages, existing has %d", len(data.State.Messages), existingMessages)
	}
	if len(data.State.Turns) < existingTurns {
		return fmt.Errorf("save session: append-only violation: new state has %d turns, existing has %d", len(data.State.Turns), existingTurns)
	}
	if len(data.State.Snapshots) < existingSnapshots {
		return fmt.Errorf("save session: append-only violation: new state has %d snapshots, existing has %d", len(data.State.Snapshots), existingSnapshots)
	}

	for i := range data.State.Messages {
		msg := &data.State.Messages[i]

		isNew := i >= existingMessages

		if isNew && msg.MessageID == "" {
			msg.MessageID = uuid.New().String()
		}
		if isNew && msg.Timestamp.IsZero() {
			msg.Timestamp = time.Now()
		}
		if isNew && msg.Kind == "" {
			msg.Kind = KindForMessage(msg.Content.Role)
		}
		if isNew && msg.Sequence == 0 {
			lastMessageSequence++
			msg.Sequence = lastMessageSequence
		}

		if s.opts.assetStore != nil {
			if err := s.normalizeMediaParts(ctx, data.State.TenantID, sessionID, msg, &data.State); err != nil {
				return fmt.Errorf("normalize media parts for message %q: %w", msg.MessageID, err)
			}
		}
	}

	newMessagesCount := len(data.State.Messages) - existingMessages
	newTurnsCount := len(data.State.Turns) - existingTurns
	if newMessagesCount > 0 {
		firstNew := data.State.Messages[existingMessages]
		lastNew := data.State.Messages[len(data.State.Messages)-1]

		if newTurnsCount == 0 {
			lastTurnSequence++
			data.State.Turns = append(data.State.Turns, TurnRecord{
				TurnID:               uuid.New().String(),
				Sequence:             lastTurnSequence,
				StartedAt:            firstNew.Timestamp,
				EndedAt:              lastNew.Timestamp,
				FirstMessageSequence: firstNew.Sequence,
				LastMessageSequence:  lastNew.Sequence,
				MessageCount:         newMessagesCount,
			})
		} else {
			newMessages := data.State.Messages[existingMessages:]
			msgCursor := 0
			for i := existingTurns; i < len(data.State.Turns); i++ {
				turn := &data.State.Turns[i]
				if turn.Sequence == 0 {
					lastTurnSequence++
					turn.Sequence = lastTurnSequence
				}

				if turn.MessageCount <= 0 {
					continue
				}

				if msgCursor+turn.MessageCount > len(newMessages) {
					return fmt.Errorf("save session: turn/message mismatch: turn %d claims %d messages, %d remaining", i-existingTurns+1, turn.MessageCount, len(newMessages)-msgCursor)
				}

				firstSeq := newMessages[msgCursor].Sequence
				lastSeq := newMessages[msgCursor+turn.MessageCount-1].Sequence
				if turn.FirstMessageSequence == 0 {
					turn.FirstMessageSequence = firstSeq
				}
				if turn.LastMessageSequence == 0 {
					turn.LastMessageSequence = lastSeq
				}
				msgCursor += turn.MessageCount
			}

			if msgCursor != len(newMessages) {
				return fmt.Errorf("save session: turn/message mismatch: %d new messages not referenced by new turns", len(newMessages)-msgCursor)
			}
		}

		if len(data.State.Snapshots) == existingSnapshots {
			lastSnapshotSequence++
			snapshot := StateSnapshot{
				SnapshotID:    uuid.New().String(),
				Sequence:      lastSnapshotSequence,
				CapturedAt:    time.Now(),
				TenantID:      s.opts.tenantID,
				SessionID:     sessionID,
				MessageCount:  len(data.State.Messages),
				TurnSequence:  data.State.Turns[len(data.State.Turns)-1].Sequence,
				StateVersion:  1,
				StateChecksum: stateChecksum(data.State),
			}
			data.State.Snapshots = append(data.State.Snapshots, snapshot)
		}
	}

	return s.opts.operator.SaveState(ctx, s.opts.tenantID, sessionID, data.State)
}

func (s *Session) normalizeMediaParts(ctx context.Context, tenantID, sessionID string, msg *SessionMessage, state *SessionState) error {
	for i, part := range msg.Content.Content {
		if part == nil || !part.IsMedia() {
			continue
		}

		if filepath.IsAbs(part.Text) {
			state.Assets = upsertSessionAsset(state.Assets, SessionAsset{
				AssetID:   filepath.Base(part.Text),
				MessageID: msg.MessageID,
				PartIndex: i,
				MimeType:  part.ContentType,
				Path:      part.Text,
			})
			continue
		}

		mimeType, payload, ok := parseDataURI(part.Text)
		if !ok {
			continue
		}

		assetID := uuid.New().String()
		absolutePath, err := s.opts.assetStore.Put(ctx, tenantID, sessionID, assetID, mimeType, payload)
		if err != nil {
			return fmt.Errorf("store media asset: %w", err)
		}

		part.Text = absolutePath
		part.ContentType = mimeType
		state.Assets = upsertSessionAsset(state.Assets, SessionAsset{
			AssetID:   assetID,
			MessageID: msg.MessageID,
			PartIndex: i,
			MimeType:  mimeType,
			SizeBytes: len(payload),
			Path:      absolutePath,
		})

	}
	return nil
}

func upsertSessionAsset(assets []SessionAsset, asset SessionAsset) []SessionAsset {
	for i := range assets {
		if assets[i].MessageID == asset.MessageID && assets[i].PartIndex == asset.PartIndex {
			assets[i] = asset
			return assets
		}
	}

	return append(assets, asset)
}

func parseDataURI(uri string) (mimeType string, data []byte, ok bool) {
	if !strings.HasPrefix(uri, "data:") {
		return "", nil, false
	}

	s := strings.TrimPrefix(uri, "data:")

	parts := strings.SplitN(s, ",", 2)
	if len(parts) != 2 {
		return "", nil, false
	}

	meta := parts[0]
	rawData := parts[1]

	isBase64 := strings.HasSuffix(meta, ";base64")
	if isBase64 {
		mimeType = strings.TrimSuffix(meta, ";base64")
		decoded, err := base64.StdEncoding.DecodeString(rawData)
		if err != nil {
			return "", nil, false
		}
		data = decoded
	} else {
		mimeType = meta
		decoded, err := url.PathUnescape(rawData)
		if err != nil {
			return "", nil, false
		}
		data = []byte(decoded)
	}

	if mimeType == "" {
		mimeType = "text/plain"
	}
	return mimeType, data, true
}

// MessagesForTurn reconstructs the persisted message window represented by a
// turn using sequence boundaries.
func MessagesForTurn(state SessionState, turn TurnRecord) ([]SessionMessage, error) {
	if turn.FirstMessageSequence == 0 || turn.LastMessageSequence == 0 {
		return nil, fmt.Errorf("messages for turn: turn %q has missing sequence bounds", turn.TurnID)
	}
	if turn.LastMessageSequence < turn.FirstMessageSequence {
		return nil, fmt.Errorf("messages for turn: invalid sequence bounds %d..%d", turn.FirstMessageSequence, turn.LastMessageSequence)
	}

	startIdx := -1
	endIdx := -1
	for i := range state.Messages {
		seq := state.Messages[i].Sequence
		if seq == turn.FirstMessageSequence {
			startIdx = i
		}
		if seq == turn.LastMessageSequence {
			endIdx = i
		}
	}
	if startIdx == -1 || endIdx == -1 || endIdx < startIdx {
		return nil, fmt.Errorf("messages for turn: sequences not found for turn %q", turn.TurnID)
	}

	window := copyMessages(state.Messages[startIdx : endIdx+1])
	if turn.MessageCount > 0 && len(window) != turn.MessageCount {
		return nil, fmt.Errorf("messages for turn: message count mismatch for turn %q: got %d, want %d", turn.TurnID, len(window), turn.MessageCount)
	}

	return window, nil
}

// WithTokenBudget configures load-time token-budget pruning.
//
// budget is interpreted by the configured TokenEstimator and only applies when
// mode is TokenBudget.
func WithTokenBudget(budget int) SessionOption {
	return func(opts *sessionOptions) {
		opts.mode = TokenBudget
		opts.nMessages = budget
		opts.tokenBudget = budget
	}
}

// WithTokenEstimator sets the token estimator used by TokenBudget mode.
func WithTokenEstimator(estimator TokenEstimator) SessionOption {
	return func(opts *sessionOptions) {
		opts.tokenEstimator = estimator
	}
}
