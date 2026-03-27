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

package flows

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/firebase/genkit/go/genkit"
)

// Heartbeat schedules and executes periodic heartbeat checks over session state.
type Heartbeat struct {
	cfg      *HeartbeatConfig
	flow     *core.Flow[*HeartbeatInput, *HeartbeatOutput, struct{}]
	onResult func(*HeartbeatOutput)

	running atomic.Bool
	stopCh  chan struct{}
	once    sync.Once
}

// HeartbeatInput is the flow input for a single heartbeat execution.
type HeartbeatInput struct {
	SessionID   string           `json:"sessionID"`
	AgentConfig *AgentLoopConfig `json:"agentConfig,omitempty"`
	TenantID    string           `json:"tenantID"`
	RunAt       time.Time        `json:"runAt"`
}

type heartbeatOptions struct {
	bus           *EventBus
	defaultConfig *AgentLoopConfig
	onResult      func(*HeartbeatOutput)
	operator      AgentLoopOperator
}

// HeartbeatOption configures NewHeartbeat behavior.
type HeartbeatOption func(*heartbeatOptions)

// WithHeartbeatEventBus enables lifecycle event emission for heartbeat runs.
func WithHeartbeatEventBus(bus *EventBus) HeartbeatOption {
	return func(opts *heartbeatOptions) {
		opts.bus = bus
	}
}

// WithHeartbeatOnResult registers a callback invoked after each heartbeat run.
func WithHeartbeatOnResult(onResult func(*HeartbeatOutput)) HeartbeatOption {
	return func(opts *heartbeatOptions) {
		opts.onResult = onResult
	}
}

// WithHeartbeatLoopOperator injects a custom agent loop operator.
func WithHeartbeatLoopOperator(loopOperator AgentLoopOperator) HeartbeatOption {
	return func(opts *heartbeatOptions) {
		opts.operator = loopOperator
	}
}

// WithCustomHeartbeatAgentConfig sets the default agent loop config for
// heartbeat runs.
func WithCustomHeartbeatAgentConfig(config AgentLoopConfig) HeartbeatOption {
	return func(opts *heartbeatOptions) {
		opts.defaultConfig = &config
	}
}

// NewHeartbeat creates a heartbeat runner with an internally registered flow.
func NewHeartbeat(
	g *genkit.Genkit,
	store *memory.Session,
	cfg HeartbeatConfig,
	opts ...HeartbeatOption,
) *Heartbeat {
	options := &heartbeatOptions{
		operator: &defaultAgentLoopOperator{g: g},
	}
	for _, opt := range opts {
		opt(options)
	}

	if options.onResult == nil {
		options.onResult = func(*HeartbeatOutput) {}
	}

	h := &Heartbeat{
		cfg:      &cfg,
		onResult: options.onResult,
		stopCh:   make(chan struct{}),
	}

	h.flow = genkit.DefineFlow(
		g,
		"heartbeat",
		func(ctx context.Context, input *HeartbeatInput) (*HeartbeatOutput, error) {
			tenantStore := store.ForTenant(input.TenantID)
			sess, err := session.Load(ctx, tenantStore, input.SessionID)
			if err != nil {
				sess, err = session.New(ctx,
					session.WithID[memory.SessionState](input.SessionID),
					session.WithStore(tenantStore),
					session.WithInitialState(memory.SessionState{
						TenantID: input.TenantID,
					}),
				)
				if err != nil {
					return nil, fmt.Errorf("create new session: %w", err)
				}
			}

			ctx = session.NewContext(ctx, sess)

			history := make([]*ai.Message, 0, len(sess.State().Messages))
			for _, msg := range sess.State().Messages {
				history = append(history, &msg.Content)
			}
			priorHistoryLen := len(history)

			resolvedConfig := mergeAgentConfig(options.defaultConfig, input.AgentConfig)
			loopInput := &AgentLoopInput{
				SessionID: input.SessionID,
				Messages:  history,
				Config:    resolvedConfig,
			}

			agentLoop := NewAgentLoop(
				g,
				WithEventBus(options.bus),
				WithCustomAgentLoopOperator(options.operator),
			)

			loopOutput, err := agentLoop.Run(ctx, loopInput)
			if err != nil {
				result := errorResult(input.SessionID, input.RunAt, err)
				return &result, nil
			}

			newMessages := loopOutput.History[priorHistoryLen:]

			var sessionMessages []memory.SessionMessage
			for _, msg := range newMessages {
				sessionMessages = append(sessionMessages, memory.SessionMessage{
					Origin:  originForRole(msg.Role, memory.HeartbeatMessage),
					Content: *msg,
					Kind:    memory.KindForMessage(msg.Role),
				})
			}

			state := sess.State()
			state.Messages = append(state.Messages, sessionMessages...)
			if err := sess.UpdateState(ctx, state); err != nil {
				return nil, fmt.Errorf("update session state: %w", err)
			}

			rawContent := extractText(loopOutput.Response)

			result := evaluateHeartbeatResult(input.SessionID, input.RunAt, rawContent, loopOutput.Turns, h.cfg, loopOutput.Response)

			return &result, nil
		},
	)

	return h
}

// Start begins periodic heartbeat execution based on the configured interval.
func (h *Heartbeat) Start(ctx context.Context) {
	if h.cfg.Interval <= 0 {
		return
	}

	go func() {
		ticker := time.NewTicker(h.cfg.Interval)
		defer ticker.Stop()

		for {
			select {
			case <-h.stopCh:
				return
			case <-ctx.Done():
				return
			case t := <-ticker.C:
				h.Run(ctx, t)
			}
		}
	}()
}

// Wake triggers an immediate heartbeat run asynchronously.
func (h *Heartbeat) Wake(ctx context.Context) {
	go h.Run(ctx, time.Now())
}

// Stop terminates periodic heartbeat scheduling.
func (h *Heartbeat) Stop() {
	h.once.Do(func() {
		close(h.stopCh)
	})
}

func (h *Heartbeat) tryRun(ctx context.Context, runAt time.Time) (*HeartbeatOutput, error) {
	sessionID := h.sessionID()

	tenantID := h.cfg.TenantID

	input := &HeartbeatInput{
		SessionID:   sessionID,
		TenantID:    tenantID,
		AgentConfig: h.cfg.AgentConfig,
		RunAt:       runAt,
	}

	return h.flow.Run(ctx, input)
}

// Run executes a single heartbeat tick, enforcing active-hours and
// single-flight semantics.
func (h *Heartbeat) Run(ctx context.Context, tickTime time.Time) {
	sessionID := h.sessionID()

	if !h.cfg.ActiveHours.Contains(tickTime) {
		skipResult := skippedResult(sessionID, SkipReasonOutsideHours)
		h.onResult(&skipResult)
		return
	}

	if !h.running.CompareAndSwap(false, true) {
		skipResult := skippedResult(sessionID, SkipReasonBusy)
		h.onResult(&skipResult)
		return
	}
	defer h.running.Store(false)

	result, err := h.tryRun(ctx, tickTime)
	if err != nil {
		errResult := errorResult(sessionID, tickTime, err)
		h.onResult(&errResult)
		return
	}
	h.onResult(result)
}

func (h *Heartbeat) sessionID() string {
	if h.cfg.SessionID != "" {
		return h.cfg.SessionID
	}
	return "heartbeat"
}

func extractText(msg *ai.Message) string {
	if msg == nil {
		return ""
	}
	var b strings.Builder
	for _, part := range msg.Content {
		if part.IsText() {
			b.WriteString(part.Text)
		}
	}
	return b.String()
}
