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
	// Origin identifies where the message came from.
	Origin MessageOrigin `json:"origin"`
	// Content is the full Genkit message payload.
	Content ai.Message `json:"content"`
	// Timestamp is when the message was created.
	Timestamp time.Time `json:"timestamp"`
}

// SessionState is the persisted state stored per session.
type SessionState struct {
	TenantID string           `json:"tenantID"`
	Messages []SessionMessage `json:"messages"`
}

// SessionOperator abstracts the storage backend.
type SessionOperator interface {
	// SaveState persists the full session state. Always called, regardless of mode.
	SaveState(ctx context.Context, sessionID string, state SessionState) error

	// LoadState retrieves full session state.
	// The mode and nMessages parameters control pruning at load time.
	LoadState(ctx context.Context, sessionID string, mode PersistenceMode, nMessages int) (*SessionState, error)

	// DeleteSession removes all messages for a session.
	DeleteSession(ctx context.Context, sessionID string) error
}

type defaultSessionOperator struct {
	mu    sync.Mutex
	store map[string]SessionState
}

func (o *defaultSessionOperator) SaveState(ctx context.Context, sessionID string, state SessionState) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.store == nil {
		o.store = make(map[string]SessionState)
	}

	// Deep copy the messages slice to avoid shared references with the caller.
	msgs := make([]SessionMessage, len(state.Messages))
	copy(msgs, state.Messages)

	o.store[sessionID] = SessionState{
		TenantID: state.TenantID,
		Messages: msgs,
	}
	return nil
}

func (o *defaultSessionOperator) LoadState(ctx context.Context, sessionID string, mode PersistenceMode, nMessages int) (*SessionState, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	state, ok := o.store[sessionID]
	if !ok {
		return nil, nil
	}

	filtered := filterMessages(state.Messages, mode, nMessages)

	return &SessionState{
		TenantID: state.TenantID,
		Messages: filtered,
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

func (o *defaultSessionOperator) DeleteSession(ctx context.Context, sessionID string) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	delete(o.store, sessionID)
	return nil
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

// SessionOption configures session store behavior.
type SessionOption func(*sessionOptions)

type sessionOptions struct {
	mode      PersistenceMode
	nMessages int // used for SlidingWindow and TailEndsPruning modes

	operator SessionOperator
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

// PersistenceMode controls how many messages are returned on load.
type PersistenceMode int

const (
	// All loads every message in the session.
	All PersistenceMode = iota
	// SlidingWindow loads only the last N messages.
	SlidingWindow
	// TailEndsPruning loads the first N and last N messages.
	TailEndsPruning
)

// Session is a Genkit session store implementation backed by SessionOperator.
type Session struct {
	opts sessionOptions
}

// NewSession creates a new session store.
func NewSession(opts ...SessionOption) *Session {
	options := sessionOptions{
		mode:     All,
		operator: &defaultSessionOperator{},
	}
	for _, opt := range opts {
		opt(&options)
	}
	return &Session{opts: options}
}

var _ session.Store[SessionState] = (*Session)(nil)

// Get loads session state by ID.
func (s *Session) Get(ctx context.Context, sessionID string) (*session.Data[SessionState], error) {
	state, err := s.opts.operator.LoadState(ctx, sessionID, s.opts.mode, s.opts.nMessages)
	if err != nil {
		return nil, err
	}
	if state == nil {
		return nil, nil
	}
	return &session.Data[SessionState]{
		ID:    sessionID,
		State: *state,
	}, nil
}

// Save persists the provided session state, assigning IDs and timestamps when
// missing.
func (s *Session) Save(ctx context.Context, sessionID string, data *session.Data[SessionState]) error {
	for i := range data.State.Messages {
		if data.State.Messages[i].MessageID == "" {
			data.State.Messages[i].MessageID = uuid.NewString()
		}
		if data.State.Messages[i].Timestamp.IsZero() {
			data.State.Messages[i].Timestamp = time.Now()
		}
	}

	return s.opts.operator.SaveState(ctx, sessionID, data.State)
}
