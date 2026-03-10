package flows

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type EventType string

const (
	AgentStart          EventType = "agent-start"
	AgentEnd            EventType = "agent-end"
	TurnStart           EventType = "turn-start"
	TurnEnd             EventType = "turn-end"
	MessageStart        EventType = "message-start"
	MessageUpdate       EventType = "message-update"
	MessageEnd          EventType = "message-end"
	ToolExecutionStart  EventType = "tool-execution-start"
	ToolExecutionUpdate EventType = "tool-execution-update"
	ToolExecutionEnd    EventType = "tool-execution-end"
)

type Event[T any] struct {
	Type      EventType `json:"type"`
	Timestamp time.Time `json:"timestamp"`
	Data      T         `json:"data"`
}

type EventHandler[T any] func(ctx context.Context, event *Event[T]) error

type EventBus struct {
	mu          sync.RWMutex
	subscribers map[EventType][]*subscriber
}

type subscriber struct {
	handler any
	removed bool
}

func NewEventBus() *EventBus {
	return &EventBus{}
}

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
