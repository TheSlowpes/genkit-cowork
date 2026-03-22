// Copyright 2025 Google LLC
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
	"errors"
	"sync"
	"testing"
	"time"
)

// testContext is a simple typed payload for testing the generic event system.
type testContext struct {
	Value   string
	Counter int
}

func TestEmit_SingleHandler(t *testing.T) {
	bus := NewEventBus()
	var received *Event[testContext]

	Subscribe(bus, AgentStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		received = event
		return nil
	}))

	ctx := context.Background()
	event, err := EmitEvent(bus, ctx, AgentStart, testContext{Value: "hello", Counter: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if received == nil {
		t.Fatal("handler was not called")
	}
	if received.Data.Value != "hello" {
		t.Errorf("expected Value 'hello', got %q", received.Data.Value)
	}
	if received.Data.Counter != 1 {
		t.Errorf("expected Counter 1, got %d", received.Data.Counter)
	}
	if event.Type != AgentStart {
		t.Errorf("expected event type %q, got %q", AgentStart, event.Type)
	}
	if event.Timestamp.IsZero() {
		t.Error("expected non-zero timestamp")
	}
}

func TestEmit_ChainedMutation(t *testing.T) {
	bus := NewEventBus()

	// Handler A increments counter
	Subscribe(bus, TurnStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		event.Data.Counter += 10
		return nil
	}))

	// Handler B should see handler A's mutation and mutate further
	Subscribe(bus, TurnStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		event.Data.Counter *= 2
		return nil
	}))

	ctx := context.Background()
	event, err := EmitEvent(bus, ctx, TurnStart, testContext{Counter: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Expected: (1 + 10) * 2 = 22
	if event.Data.Counter != 22 {
		t.Errorf("expected Counter 22 after chained mutation, got %d", event.Data.Counter)
	}
}

func TestEmit_ErrorAbort(t *testing.T) {
	bus := NewEventBus()
	handlerAcalled := false
	handlerBcalled := false
	handlerCcalled := false

	Subscribe(bus, MessageStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		handlerAcalled = true
		return nil
	}))

	expectedErr := errors.New("handler B failed")
	Subscribe(bus, MessageStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		handlerBcalled = true
		return expectedErr
	}))

	Subscribe(bus, MessageStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		handlerCcalled = true
		return nil
	}))

	ctx := context.Background()
	_, err := EmitEvent(bus, ctx, MessageStart, testContext{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, expectedErr) {
		t.Errorf("expected wrapped error to contain %q, got %q", expectedErr, err)
	}
	if !handlerAcalled {
		t.Error("handler A should have been called")
	}
	if !handlerBcalled {
		t.Error("handler B should have been called")
	}
	if handlerCcalled {
		t.Error("handler C should NOT have been called after handler B errored")
	}
}

func TestEmit_ContextCancellation(t *testing.T) {
	bus := NewEventBus()
	handlerAcalled := false
	handlerBcalled := false

	// Handler A cancels the context
	Subscribe(bus, ToolExecutionStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		handlerAcalled = true
		return nil
	}))

	Subscribe(bus, ToolExecutionStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		handlerBcalled = true
		return nil
	}))

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before emitting

	_, err := EmitEvent(bus, ctx, ToolExecutionStart, testContext{})
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
	if handlerAcalled {
		t.Error("handler A should NOT have been called with cancelled context")
	}
	if handlerBcalled {
		t.Error("handler B should NOT have been called with cancelled context")
	}
}

func TestEmit_ContextCancelledMidPipeline(t *testing.T) {
	bus := NewEventBus()
	handlerBcalled := false

	ctx, cancel := context.WithCancel(context.Background())

	// Handler A runs successfully then cancels the context
	Subscribe(bus, TurnEnd, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		cancel()
		return nil
	}))

	// Handler B should be skipped because context was cancelled after handler A
	Subscribe(bus, TurnEnd, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		handlerBcalled = true
		return nil
	}))

	_, err := EmitEvent(bus, ctx, TurnEnd, testContext{})
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
	if handlerBcalled {
		t.Error("handler B should NOT have been called after context was cancelled")
	}
}

func TestUnsubscribe(t *testing.T) {
	bus := NewEventBus()
	callCount := 0

	unsub := Subscribe(bus, AgentEnd, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		callCount++
		return nil
	}))

	ctx := context.Background()

	// First emit -- handler should be called
	_, err := EmitEvent(bus, ctx, AgentEnd, testContext{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if callCount != 1 {
		t.Errorf("expected callCount 1, got %d", callCount)
	}

	// Unsubscribe
	unsub()

	// Second emit -- handler should NOT be called
	_, err = EmitEvent(bus, ctx, AgentEnd, testContext{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if callCount != 1 {
		t.Errorf("expected callCount still 1 after unsubscribe, got %d", callCount)
	}
}

func TestUnsubscribe_Idempotent(t *testing.T) {
	bus := NewEventBus()

	unsub := Subscribe(bus, MessageEnd, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		return nil
	}))

	// Calling unsubscribe multiple times should not panic
	unsub()
	unsub()
	unsub()
}

func TestUnsubscribe_OnlyRemovesTargetHandler(t *testing.T) {
	bus := NewEventBus()
	handlerAcount := 0
	handlerBcount := 0

	Subscribe(bus, MessageUpdate, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		handlerAcount++
		return nil
	}))

	unsubB := Subscribe(bus, MessageUpdate, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		handlerBcount++
		return nil
	}))

	ctx := context.Background()

	// Both handlers called
	EmitEvent(bus, ctx, MessageUpdate, testContext{})
	if handlerAcount != 1 || handlerBcount != 1 {
		t.Fatalf("expected both handlers called once, got A=%d B=%d", handlerAcount, handlerBcount)
	}

	// Unsubscribe B only
	unsubB()

	// Only handler A should be called
	EmitEvent(bus, ctx, MessageUpdate, testContext{})
	if handlerAcount != 2 {
		t.Errorf("expected handler A called twice, got %d", handlerAcount)
	}
	if handlerBcount != 1 {
		t.Errorf("expected handler B still called once after unsubscribe, got %d", handlerBcount)
	}
}

func TestEmit_NoSubscribers(t *testing.T) {
	bus := NewEventBus()
	ctx := context.Background()

	// Emitting with no subscribers should return nil, not error
	err := Emit(bus, ctx, &Event[testContext]{Type: AgentStart, Timestamp: time.Now(), Data: testContext{}})
	if err != nil {
		t.Errorf("expected nil error for no subscribers, got %v", err)
	}
}

func TestEmit_NoSubscribersForEventType(t *testing.T) {
	bus := NewEventBus()
	ctx := context.Background()

	// Subscribe to one event type
	Subscribe(bus, AgentStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		return nil
	}))

	// Emit a different event type -- should return nil
	err := Emit(bus, ctx, &Event[testContext]{Type: AgentEnd, Timestamp: time.Now(), Data: testContext{}})
	if err != nil {
		t.Errorf("expected nil error for unsubscribed event type, got %v", err)
	}
}

func TestEmit_HandlerTypeMismatch(t *testing.T) {
	bus := NewEventBus()
	ctx := context.Background()

	// Subscribe with testContext type
	Subscribe(bus, AgentStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		return nil
	}))

	// Emit with a different type (AgentContext) on the same event type
	// The handler was registered as EventHandler[testContext] but we emit Event[AgentContext]
	err := Emit(bus, ctx, &Event[AgentContext]{Type: AgentStart, Timestamp: time.Now(), Data: AgentContext{}})
	if err == nil {
		t.Fatal("expected type mismatch error, got nil")
	}
	if expected := "handler type mismatch"; !containsString(err.Error(), expected) {
		t.Errorf("expected error containing %q, got %q", expected, err.Error())
	}
}

func TestEmitEvent_SetsTimestamp(t *testing.T) {
	bus := NewEventBus()
	ctx := context.Background()

	before := time.Now()
	event, err := EmitEvent(bus, ctx, AgentStart, testContext{Value: "test"})
	after := time.Now()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event.Timestamp.Before(before) || event.Timestamp.After(after) {
		t.Errorf("expected timestamp between %v and %v, got %v", before, after, event.Timestamp)
	}
}

func TestEmitEvent_ReturnsEventOnError(t *testing.T) {
	bus := NewEventBus()
	ctx := context.Background()

	Subscribe(bus, AgentStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		return errors.New("something went wrong")
	}))

	event, err := EmitEvent(bus, ctx, AgentStart, testContext{Value: "test"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// Event should still be returned even on error
	if event == nil {
		t.Fatal("expected event to be returned even on error")
	}
	if event.Data.Value != "test" {
		t.Errorf("expected event data preserved, got %q", event.Data.Value)
	}
}

func TestEmit_HandlerRegistrationOrder(t *testing.T) {
	bus := NewEventBus()
	var order []int

	Subscribe(bus, ToolExecutionEnd, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		order = append(order, 1)
		return nil
	}))
	Subscribe(bus, ToolExecutionEnd, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		order = append(order, 2)
		return nil
	}))
	Subscribe(bus, ToolExecutionEnd, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		order = append(order, 3)
		return nil
	}))

	ctx := context.Background()
	EmitEvent(bus, ctx, ToolExecutionEnd, testContext{})

	if len(order) != 3 {
		t.Fatalf("expected 3 handlers called, got %d", len(order))
	}
	for i, v := range order {
		if v != i+1 {
			t.Errorf("expected handler %d at position %d, got %d", i+1, i, v)
		}
	}
}

func TestEmit_DifferentEventTypesDontInterfere(t *testing.T) {
	bus := NewEventBus()
	agentCalled := false
	turnCalled := false

	Subscribe(bus, AgentStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		agentCalled = true
		return nil
	}))

	Subscribe(bus, TurnStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
		turnCalled = true
		return nil
	}))

	ctx := context.Background()

	// Emit only AgentStart
	EmitEvent(bus, ctx, AgentStart, testContext{})

	if !agentCalled {
		t.Error("agent handler should have been called")
	}
	if turnCalled {
		t.Error("turn handler should NOT have been called")
	}
}

func TestSubscribe_ConcurrentSafety(t *testing.T) {
	bus := NewEventBus()
	ctx := context.Background()
	var wg sync.WaitGroup
	n := 100

	// Concurrently subscribe
	for range n {
		wg.Go(
			func() {
				Subscribe(bus, AgentStart, EventHandler[testContext](func(ctx context.Context, event *Event[testContext]) error {
					return nil
				}))
			})
	}
	wg.Wait()

	// Concurrently emit
	for range n {
		wg.Go(
			func() {
				EmitEvent(bus, ctx, AgentStart, testContext{})
			})
	}
	wg.Wait()

	// If we get here without a race condition panic, the test passes.
	// Run with: go test -race
}

func TestEmit_WithRealContextTypes(t *testing.T) {
	bus := NewEventBus()
	ctx := context.Background()

	// Test with AgentContext
	var receivedAgent AgentContext
	Subscribe(bus, AgentStart, EventHandler[AgentContext](func(ctx context.Context, event *Event[AgentContext]) error {
		receivedAgent = event.Data
		return nil
	}))

	EmitEvent(bus, ctx, AgentStart, AgentContext{
		SessionID: "session-123",
		ModelName: "gemini-pro",
		Tools:     []string{"bash", "read", "edit"},
	})

	if receivedAgent.SessionID != "session-123" {
		t.Errorf("expected SessionID 'session-123', got %q", receivedAgent.SessionID)
	}
	if receivedAgent.ModelName != "gemini-pro" {
		t.Errorf("expected ModelName 'gemini-pro', got %q", receivedAgent.ModelName)
	}
	if len(receivedAgent.Tools) != 3 {
		t.Errorf("expected 3 tools, got %d", len(receivedAgent.Tools))
	}
}

func TestEmit_WithToolExecutionContext(t *testing.T) {
	bus := NewEventBus()
	ctx := context.Background()

	var receivedTool ToolExecutionContext
	Subscribe(bus, ToolExecutionStart, EventHandler[ToolExecutionContext](func(ctx context.Context, event *Event[ToolExecutionContext]) error {
		receivedTool = event.Data
		// Interceptor: modify the tool input before execution
		event.Data.Input = map[string]string{"command": "ls -la (sanitized)"}
		return nil
	}))

	event, err := EmitEvent(bus, ctx, ToolExecutionStart, ToolExecutionContext{
		SessionID: "session-456",
		ToolName:  "bash",
		Input:     map[string]string{"command": "ls -la"},
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify handler received original data
	if receivedTool.ToolName != "bash" {
		t.Errorf("expected ToolName 'bash', got %q", receivedTool.ToolName)
	}

	// Verify mutation was applied to the event
	input, ok := event.Data.Input.(map[string]string)
	if !ok {
		t.Fatal("expected Input to be map[string]string")
	}
	if input["command"] != "ls -la (sanitized)" {
		t.Errorf("expected mutated input, got %q", input["command"])
	}
}

// containsString checks if s contains substr.
func containsString(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
