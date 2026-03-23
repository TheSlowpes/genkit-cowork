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
	"sync"
	"time"
)

// EventType identifies a lifecycle event in flow execution.
type EventType string

const (
	// AgentStart is emitted when an agent loop begins.
	AgentStart EventType = "agent-start"
	// AgentEnd is emitted when an agent loop ends.
	AgentEnd EventType = "agent-end"
	// TurnStart is emitted at the start of each model turn.
	TurnStart EventType = "turn-start"
	// TurnEnd is emitted at the end of each model turn.
	TurnEnd EventType = "turn-end"
	// MessageStart is emitted when a message emission starts.
	MessageStart EventType = "message-start"
	// MessageUpdate is emitted for streamed message chunks.
	MessageUpdate EventType = "message-update"
	// MessageEnd is emitted when message emission completes.
	MessageEnd EventType = "message-end"
	// ToolExecutionStart is emitted before a tool is run.
	ToolExecutionStart EventType = "tool-execution-start"
	// ToolExecutionUpdate is emitted for in-flight tool updates.
	ToolExecutionUpdate EventType = "tool-execution-update"
	// ToolExecutionEnd is emitted after a tool completes.
	ToolExecutionEnd EventType = "tool-execution-end"
)

// Event is a typed event envelope emitted on the EventBus.
type Event[T any] struct {
	Type      EventType `json:"type"`
	Timestamp time.Time `json:"timestamp"`
	Data      T         `json:"data"`
}

// EventHandler handles an event payload of type T.
type EventHandler[T any] func(ctx context.Context, event *Event[T]) error

// EventBus is an in-process synchronous pub/sub event dispatcher.
type EventBus struct {
	mu          sync.RWMutex
	subscribers map[EventType][]*subscriber
}

type subscriber struct {
	handler any
	removed bool
}

// NewEventBus creates an empty event bus.
func NewEventBus() *EventBus {
	return &EventBus{}
}

// Subscribe registers a typed handler for an event type and returns an
// unsubscribe function.
func Subscribe[T any](bus *EventBus, eventType EventType, handler EventHandler[T]) func() {
	bus.mu.Lock()
	defer bus.mu.Unlock()

	if bus.subscribers == nil {
		bus.subscribers = make(map[EventType][]*subscriber)
	}
	subscriberStruct := &subscriber{handler: handler}
	bus.subscribers[eventType] = append(bus.subscribers[eventType], subscriberStruct)
	removed := false
	unsubscribe := func() {
		bus.mu.Lock()
		defer bus.mu.Unlock()

		if removed {
			return
		}
		removed = true
		subscriberStruct.removed = true
	}
	return unsubscribe
}

// Emit dispatches a typed event to all subscribers for its event type.
func Emit[T any](bus *EventBus, ctx context.Context, event *Event[T]) error {
	bus.mu.RLock()
	defer bus.mu.RUnlock()

	handlers, ok := bus.subscribers[event.Type]
	if !ok {
		return nil
	}
	for _, handler := range handlers {
		if handler.removed {
			continue
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		eventHandler, ok := handler.handler.(EventHandler[T])
		if !ok {
			return fmt.Errorf("handler type mismatch for event type: %s", event.Type)
		}
		if err := eventHandler(ctx, event); err != nil {
			return fmt.Errorf("error handling event: %w", err)
		}
	}

	return nil
}

// EmitEvent creates and emits a typed event in one call.
func EmitEvent[T any](bus *EventBus, ctx context.Context, eventType EventType, data T) (*Event[T], error) {
	event := &Event[T]{
		Type:      eventType,
		Timestamp: time.Now(),
		Data:      data,
	}
	if err := Emit(bus, ctx, event); err != nil {
		return event, err
	}
	return event, nil
}
