package memory

import (
	"context"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/google/uuid"
)

type SessionMessage struct {
	MessageID string        `json:"messageID"`
	Origin    MessageOrigin `json:"origin"`
	Content   ai.Message    `json:"content"`
	Timestamp time.Time     `json:"timestamp"`
}

type SessionState struct {
	TenantID string           `json:"tenantID"`
	Messages []SessionMessage `json:"messages"`
}

// SessionOperator abstracts the storage backend.
type SessionOperator interface {
	// SaveState persists the full session state. Always called, regardless of mode.
	SaveState(ctx context.Context, sessionID string, state SessionState) error

	// LoadState retrieves full session state
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

type MessageOrigin string

const (
	ZoomMessage     MessageOrigin = "zoom"
	UIMessage       MessageOrigin = "ui"
	WhatsAppMessage MessageOrigin = "whatsapp"
	EmailMessage    MessageOrigin = "email"
	ModelMessage    MessageOrigin = "model"
	ToolMessage     MessageOrigin = "tool"
)

type SessionOption func(*sessionOptions)

type sessionOptions struct {
	mode      PersistenceMode
	nMessages int // used for SlidingWindow and TailEndsPruning modes

	operator SessionOperator
}

func WithPersistenceMode(mode PersistenceMode, n int) SessionOption {
	return func(opts *sessionOptions) {
		opts.mode = mode
		opts.nMessages = n
	}
}

func WithCustomSessionOperator(operator SessionOperator) SessionOption {
	return func(opts *sessionOptions) {
		opts.operator = operator
	}
}

type PersistenceMode int

const (
	All             PersistenceMode = iota // Load every message
	SlidingWindow                          // Load last N messages
	TailEndsPruning                        // Load the first N and last N messages, prune the rest
)

type Session struct {
	opts sessionOptions
}

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
