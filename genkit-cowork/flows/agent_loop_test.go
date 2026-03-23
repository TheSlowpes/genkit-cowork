// Copyright 2025 Kevin Lopes
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
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// --- Test Tool Input Types ---

// CalculatorInput is the input schema for the calculator test tool.
type CalculatorInput struct {
	Expr string `json:"expr,omitempty"`
}

// ConfirmInput is the input schema for the confirm test tool.
type ConfirmInput struct {
	Action       string `json:"action,omitempty"`
	UserApproved bool   `json:"user_approved,omitempty"`
}

// ConfirmOutput is the output schema for the confirm test tool.
type ConfirmOutput struct {
	Confirmed bool   `json:"confirmed,omitempty"`
	Status    string `json:"status,omitempty"`
}

// GenericInput is a permissive input schema for simple test tools.
type GenericInput struct {
	Command string `json:"command,omitempty"`
	Value   string `json:"value,omitempty"`
}

// ReadInput is the input schema for the read test tool.
type ReadInput struct {
	Path string `json:"path,omitempty"`
}

// --- Genkit Test Helpers ---

func newGenkitInstance(ctx context.Context, opts ...genkit.GenkitOption) *genkit.Genkit {
	return genkit.Init(ctx, opts...)
}

func mockDefineModel(g *genkit.Genkit, name string, modelResponseFunc func(ctx context.Context, gr *ai.ModelRequest, msc ai.ModelStreamCallback) (*ai.ModelResponse, error)) ai.Model {
	return genkit.DefineModel(
		g,
		"test/"+name,
		&ai.ModelOptions{
			Label: name,
			Supports: &ai.ModelSupports{
				Multiturn:   true,
				Tools:       true,
				SystemRole:  true,
				Media:       false,
				Constrained: ai.ConstrainedSupportNone,
			},
		},
		modelResponseFunc,
	)
}

func mockDefineTool[T any](g *genkit.Genkit, name, description string, toolFunc ai.MultipartToolFunc[T]) ai.Tool {
	return genkit.DefineMultipartTool(
		g,
		name,
		description,
		toolFunc,
	)
}

// --- Response Helpers ---

// textResponse creates a ModelResponse with a single text message.
func textResponse(text string) *ai.ModelResponse {
	return &ai.ModelResponse{
		FinishReason: ai.FinishReasonStop,
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(text)},
		},
	}
}

// toolCallResponse creates a ModelResponse with one or more tool request parts.
func toolCallResponse(calls ...ai.ToolRequest) *ai.ModelResponse {
	parts := make([]*ai.Part, len(calls))
	for i := range calls {
		parts[i] = ai.NewToolRequestPart(&calls[i])
	}
	return &ai.ModelResponse{
		FinishReason: ai.FinishReasonStop,
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: parts,
		},
	}
}

// --- Phase 1: Core Agent Loop Tests ---

func TestAgentLoop_SingleTurnNoTools(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	mockDefineModel(g, "single-turn", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("Hello, world!"), nil
	})

	agentLoop := NewAgentLoop(g)

	output, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-1",
			Messages:  []*ai.Message{ai.NewUserTextMessage("Hi")},
			Config:    AgentLoopConfig{Model: "test/single-turn"},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Turns != 1 {
		t.Errorf("expected 1 turn, got %d", output.Turns)
	}
	if output.Response.Text() != "Hello, world!" {
		t.Errorf("expected 'Hello, world!', got %q", output.Response.Text())
	}
	if output.SessionID != "sess-1" {
		t.Errorf("expected session ID 'sess-1', got %q", output.SessionID)
	}
	if len(output.History) != 2 {
		t.Errorf("expected 2 messages in history, got %d", len(output.History))
	}
}

func TestAgentLoop_MultiTurnToolExecution(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	var modelCalls atomic.Int32
	mockDefineModel(g, "multi-turn", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		switch call {
		case 1:
			return toolCallResponse(ai.ToolRequest{
				Name:  "calculator",
				Input: map[string]any{"expr": "6*7"},
				Ref:   "call-1",
			}), nil
		case 2:
			return textResponse("The answer is 42."), nil
		default:
			t.Fatalf("unexpected model call %d", call)
			return nil, nil
		}
	})

	genkit.DefineMultipartTool(g, "calculator", "calculator tool",
		func(tc *ai.ToolContext, input CalculatorInput) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{
				Output: map[string]any{"result": 42},
			}, nil
		},
	)

	agentLoop := NewAgentLoop(g)

	output, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-2",
			Messages:  []*ai.Message{ai.NewUserTextMessage("What is 6*7?")},
			Config: AgentLoopConfig{
				Model: "test/multi-turn",
				Tools: []string{"calculator"},
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Turns != 2 {
		t.Errorf("expected 2 turns, got %d", output.Turns)
	}
	if output.Response.Text() != "The answer is 42." {
		t.Errorf("expected 'The answer is 42.', got %q", output.Response.Text())
	}
	if len(output.History) != 4 {
		t.Errorf("expected 4 messages in history, got %d", len(output.History))
	}
	if output.History[2].Role != ai.RoleTool {
		t.Errorf("expected history[2] role 'tool', got %q", output.History[2].Role)
	}
}

func TestAgentLoop_MaxTurnsSafetyLimit(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	var modelCalls atomic.Int32
	mockDefineModel(g, "looper", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		return toolCallResponse(ai.ToolRequest{
			Name:  "loop-tool",
			Input: map[string]any{},
			Ref:   fmt.Sprintf("call-%d", call),
		}), nil
	})

	mockDefineTool(g, "loop-tool", "loops forever",
		func(tc *ai.ToolContext, input any) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{Output: "ok"}, nil
		},
	)

	agentLoop := NewAgentLoop(g)

	_, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-3",
			Messages:  []*ai.Message{ai.NewUserTextMessage("loop forever")},
			Config: AgentLoopConfig{
				Model:    "test/looper",
				Tools:    []string{"loop-tool"},
				MaxTurns: 3,
			},
		},
	)
	if err == nil {
		t.Fatal("expected error for max turns exceeded, got nil")
	}
	if !strings.Contains(err.Error(), "agent loop exceeded max turns (3)") {
		t.Errorf("expected max turns error, got %q", err.Error())
	}
}

func TestAgentLoop_EventEmissionOrder(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	var modelCalls atomic.Int32
	mockDefineModel(g, "event-test", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		if call == 1 {
			return toolCallResponse(ai.ToolRequest{
				Name:  "echo",
				Input: map[string]any{},
				Ref:   "ref-1",
			}), nil
		}
		return textResponse("pong"), nil
	})

	mockDefineTool(g, "echo", "echo tool",
		func(tc *ai.ToolContext, input any) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{Output: input}, nil
		},
	)

	bus := NewEventBus()
	var events []EventType
	var mu sync.Mutex

	record := func(et EventType) {
		mu.Lock()
		events = append(events, et)
		mu.Unlock()
	}

	Subscribe(bus, AgentStart, EventHandler[AgentContext](func(ctx context.Context, e *Event[AgentContext]) error {
		record(AgentStart)
		return nil
	}))
	Subscribe(bus, AgentEnd, EventHandler[AgentContext](func(ctx context.Context, e *Event[AgentContext]) error {
		record(AgentEnd)
		return nil
	}))
	Subscribe(bus, TurnStart, EventHandler[TurnContext](func(ctx context.Context, e *Event[TurnContext]) error {
		record(TurnStart)
		return nil
	}))
	Subscribe(bus, TurnEnd, EventHandler[TurnContext](func(ctx context.Context, e *Event[TurnContext]) error {
		record(TurnEnd)
		return nil
	}))
	Subscribe(bus, MessageStart, EventHandler[MessageContext](func(ctx context.Context, e *Event[MessageContext]) error {
		record(MessageStart)
		return nil
	}))
	Subscribe(bus, MessageEnd, EventHandler[MessageContext](func(ctx context.Context, e *Event[MessageContext]) error {
		record(MessageEnd)
		return nil
	}))
	Subscribe(bus, ToolExecutionStart, EventHandler[ToolExecutionContext](func(ctx context.Context, e *Event[ToolExecutionContext]) error {
		record(ToolExecutionStart)
		return nil
	}))
	Subscribe(bus, ToolExecutionEnd, EventHandler[ToolExecutionContext](func(ctx context.Context, e *Event[ToolExecutionContext]) error {
		record(ToolExecutionEnd)
		return nil
	}))

	agentLoop := NewAgentLoop(g, WithEventBus(bus))

	_, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-4",
			Messages:  []*ai.Message{ai.NewUserTextMessage("echo test")},
			Config: AgentLoopConfig{
				Model: "test/event-test",
				Tools: []string{"echo"},
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []EventType{
		AgentStart,
		TurnStart, MessageStart, MessageEnd,
		ToolExecutionStart, ToolExecutionEnd,
		TurnEnd,
		TurnStart, MessageStart, MessageEnd,
		TurnEnd,
		AgentEnd,
	}

	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d: %v", len(expected), len(events), events)
	}
	for i, e := range expected {
		if events[i] != e {
			t.Errorf("event[%d]: expected %q, got %q", i, e, events[i])
		}
	}
}

func TestAgentLoop_NilEventBus(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	mockDefineModel(g, "no-bus", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("no bus"), nil
	})

	agentLoop := NewAgentLoop(g)

	output, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-5",
			Messages:  []*ai.Message{ai.NewUserTextMessage("test")},
			Config:    AgentLoopConfig{Model: "test/no-bus"},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Response.Text() != "no bus" {
		t.Errorf("expected 'no bus', got %q", output.Response.Text())
	}
}

func TestAgentLoop_HookMutatesToolInput(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	var modelCalls atomic.Int32
	mockDefineModel(g, "hook-test", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		if call == 1 {
			return toolCallResponse(ai.ToolRequest{
				Name:  "bash",
				Input: map[string]any{"command": "rm -rf /"},
				Ref:   "ref-1",
			}), nil
		}
		return textResponse("done"), nil
	})

	var receivedInput any
	mockDefineTool(g, "bash", "bash tool",
		func(tc *ai.ToolContext, input any) (*ai.MultipartToolResponse, error) {
			receivedInput = input
			return &ai.MultipartToolResponse{Output: "executed"}, nil
		},
	)

	bus := NewEventBus()
	Subscribe(bus, ToolExecutionStart, EventHandler[ToolExecutionContext](func(ctx context.Context, e *Event[ToolExecutionContext]) error {
		e.Data.Input = map[string]any{"command": "echo sanitized"}
		return nil
	}))

	agentLoop := NewAgentLoop(g, WithEventBus(bus))

	_, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-6",
			Messages:  []*ai.Message{ai.NewUserTextMessage("do something dangerous")},
			Config: AgentLoopConfig{
				Model: "test/hook-test",
				Tools: []string{"bash"},
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	inputMap, ok := receivedInput.(map[string]any)
	if !ok {
		t.Fatalf("expected map input, got %T", receivedInput)
	}
	if inputMap["command"] != "echo sanitized" {
		t.Errorf("expected sanitized command, got %q", inputMap["command"])
	}
}

func TestAgentLoop_ToolNotFound(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	mockDefineModel(g, "missing-tool", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return toolCallResponse(ai.ToolRequest{
			Name:  "nonexistent",
			Input: map[string]any{},
			Ref:   "ref-1",
		}), nil
	})

	agentLoop := NewAgentLoop(g)

	_, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-7",
			Messages:  []*ai.Message{ai.NewUserTextMessage("call missing tool")},
			Config:    AgentLoopConfig{Model: "test/missing-tool"},
		},
	)
	if err == nil {
		t.Fatal("expected error for missing tool, got nil")
	}
	if !strings.Contains(err.Error(), "tool not found: nonexistent") {
		t.Errorf("expected error containing 'tool not found: nonexistent', got %q", err.Error())
	}
}

func TestAgentLoop_GenerateError(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	generateErr := errors.New("model unavailable")
	mockDefineModel(g, "fail-model", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, generateErr
	})

	bus := NewEventBus()
	var agentEndError error
	Subscribe(bus, AgentEnd, EventHandler[AgentContext](func(ctx context.Context, e *Event[AgentContext]) error {
		agentEndError = e.Data.Error
		return nil
	}))

	agentLoop := NewAgentLoop(g, WithEventBus(bus))

	_, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-8",
			Messages:  []*ai.Message{ai.NewUserTextMessage("fail")},
			Config:    AgentLoopConfig{Model: "test/fail-model"},
		},
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "model unavailable") {
		t.Errorf("expected error containing 'model unavailable', got %q", err.Error())
	}
	if agentEndError == nil {
		t.Error("expected agent-end event to carry the error")
	}
}

func TestAgentLoop_MultipleToolCallsInOneTurn(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	var modelCalls atomic.Int32
	mockDefineModel(g, "multi-tool", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		if call == 1 {
			return toolCallResponse(
				ai.ToolRequest{Name: "read", Input: map[string]any{}, Ref: "ref-1"},
				ai.ToolRequest{Name: "bash", Input: map[string]any{}, Ref: "ref-2"},
			), nil
		}
		return textResponse("Got both results."), nil
	})

	var toolCallOrder []string
	var mu sync.Mutex
	mockDefineTool(g, "read", "read tool",
		func(tc *ai.ToolContext, input any) (*ai.MultipartToolResponse, error) {
			mu.Lock()
			toolCallOrder = append(toolCallOrder, "read")
			mu.Unlock()
			return &ai.MultipartToolResponse{Output: "file contents"}, nil
		},
	)
	mockDefineTool(g, "bash", "bash tool",
		func(tc *ai.ToolContext, input any) (*ai.MultipartToolResponse, error) {
			mu.Lock()
			toolCallOrder = append(toolCallOrder, "bash")
			mu.Unlock()
			return &ai.MultipartToolResponse{Output: "command output"}, nil
		},
	)

	agentLoop := NewAgentLoop(g)

	output, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-10",
			Messages:  []*ai.Message{ai.NewUserTextMessage("read and run")},
			Config: AgentLoopConfig{
				Model: "test/multi-tool",
				Tools: []string{"read", "bash"},
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Turns != 2 {
		t.Errorf("expected 2 turns, got %d", output.Turns)
	}
	if len(toolCallOrder) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(toolCallOrder))
	}
	// History: user + model(tool calls) + tool(responses) + model(final) = 4
	if len(output.History) != 4 {
		t.Errorf("expected 4 messages in history, got %d", len(output.History))
	}
	toolRespMsg := output.History[2]
	if len(toolRespMsg.Content) != 2 {
		t.Errorf("expected 2 tool response parts, got %d", len(toolRespMsg.Content))
	}
}

func TestAgentLoop_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	g := newGenkitInstance(ctx)

	mockDefineModel(g, "cancel-model", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		cancel()
		return nil, ctx.Err()
	})

	agentLoop := NewAgentLoop(g)

	_, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-11",
			Messages:  []*ai.Message{ai.NewUserTextMessage("cancel me")},
			Config:    AgentLoopConfig{Model: "test/cancel-model"},
		},
	)
	if err == nil {
		t.Fatal("expected error from context cancellation, got nil")
	}
}

func TestAgentLoop_AgentContextPopulatedCorrectly(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	mockDefineModel(g, "ctx-model", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("ok"), nil
	})

	bus := NewEventBus()
	var startCtx, endCtx AgentContext
	Subscribe(bus, AgentStart, EventHandler[AgentContext](func(ctx context.Context, e *Event[AgentContext]) error {
		startCtx = e.Data
		return nil
	}))
	Subscribe(bus, AgentEnd, EventHandler[AgentContext](func(ctx context.Context, e *Event[AgentContext]) error {
		endCtx = e.Data
		return nil
	}))

	agentLoop := NewAgentLoop(g, WithEventBus(bus))

	_, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-12",
			Messages:  []*ai.Message{ai.NewUserTextMessage("check context")},
			Config: AgentLoopConfig{
				Model: "test/ctx-model",
				Tools: []string{"tool-a", "tool-b"},
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if startCtx.SessionID != "sess-12" {
		t.Errorf("agent-start: expected session 'sess-12', got %q", startCtx.SessionID)
	}
	if startCtx.ModelName != "test/ctx-model" {
		t.Errorf("agent-start: expected model 'test/ctx-model', got %q", startCtx.ModelName)
	}
	if len(startCtx.Tools) != 2 {
		t.Errorf("agent-start: expected 2 tools, got %d", len(startCtx.Tools))
	}
	if endCtx.SessionID != "sess-12" {
		t.Errorf("agent-end: expected session 'sess-12', got %q", endCtx.SessionID)
	}
	if endCtx.Error != nil {
		t.Errorf("agent-end: expected no error, got %v", endCtx.Error)
	}
}

func TestAgentLoop_TurnContextPopulatedCorrectly(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	var modelCalls atomic.Int32
	mockDefineModel(g, "turn-ctx", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		if call == 1 {
			return toolCallResponse(ai.ToolRequest{Name: "noop", Input: map[string]any{}, Ref: "r1"}), nil
		}
		return textResponse("done"), nil
	})

	mockDefineTool(g, "noop", "no-op tool",
		func(tc *ai.ToolContext, input any) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{Output: "ok"}, nil
		},
	)

	bus := NewEventBus()
	var turnStarts []TurnContext
	var turnEnds []TurnContext
	Subscribe(bus, TurnStart, EventHandler[TurnContext](func(ctx context.Context, e *Event[TurnContext]) error {
		turnStarts = append(turnStarts, e.Data)
		return nil
	}))
	Subscribe(bus, TurnEnd, EventHandler[TurnContext](func(ctx context.Context, e *Event[TurnContext]) error {
		turnEnds = append(turnEnds, e.Data)
		return nil
	}))

	agentLoop := NewAgentLoop(g, WithEventBus(bus))

	_, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-13",
			Messages:  []*ai.Message{ai.NewUserTextMessage("test turns")},
			Config: AgentLoopConfig{
				Model: "test/turn-ctx",
				Tools: []string{"noop"},
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(turnStarts) != 2 {
		t.Fatalf("expected 2 turn-starts, got %d", len(turnStarts))
	}
	if len(turnEnds) != 2 {
		t.Fatalf("expected 2 turn-ends, got %d", len(turnEnds))
	}

	if turnStarts[0].TurnNumber != 1 {
		t.Errorf("turn-start[0]: expected turn 1, got %d", turnStarts[0].TurnNumber)
	}
	if len(turnStarts[0].Messages) != 1 {
		t.Errorf("turn-start[0]: expected 1 message, got %d", len(turnStarts[0].Messages))
	}
	if turnEnds[0].Response == nil {
		t.Error("turn-end[0]: expected response to be set")
	}
	if len(turnEnds[0].ToolCalls) != 1 {
		t.Errorf("turn-end[0]: expected 1 tool call message, got %d", len(turnEnds[0].ToolCalls))
	}

	if turnStarts[1].TurnNumber != 2 {
		t.Errorf("turn-start[1]: expected turn 2, got %d", turnStarts[1].TurnNumber)
	}
	if len(turnStarts[1].Messages) != 3 {
		t.Errorf("turn-start[1]: expected 3 messages, got %d", len(turnStarts[1].Messages))
	}
}

func TestAgentLoop_MaxTurnsZeroMeansUnlimited(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	var modelCalls atomic.Int32
	mockDefineModel(g, "unlimited", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		if call <= 5 {
			return toolCallResponse(ai.ToolRequest{Name: "step", Input: map[string]any{}, Ref: fmt.Sprintf("r%d", call)}), nil
		}
		return textResponse("finally done"), nil
	})

	mockDefineTool(g, "step", "step tool",
		func(tc *ai.ToolContext, input any) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{Output: "ok"}, nil
		},
	)

	agentLoop := NewAgentLoop(g)

	output, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-14",
			Messages:  []*ai.Message{ai.NewUserTextMessage("go")},
			Config: AgentLoopConfig{
				Model:    "test/unlimited",
				Tools:    []string{"step"},
				MaxTurns: 0, // unlimited
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Turns != 6 {
		t.Errorf("expected 6 turns, got %d", output.Turns)
	}
}

func TestAgentLoop_ToolExecutionContextDuration(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	var modelCalls atomic.Int32
	mockDefineModel(g, "duration-test", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		if call == 1 {
			return toolCallResponse(ai.ToolRequest{Name: "slow", Input: map[string]any{}, Ref: "r1"}), nil
		}
		return textResponse("ok"), nil
	})

	mockDefineTool(g, "slow", "slow tool",
		func(tc *ai.ToolContext, input any) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{Output: "done"}, nil
		},
	)

	bus := NewEventBus()
	var endCtx ToolExecutionContext
	Subscribe(bus, ToolExecutionEnd, EventHandler[ToolExecutionContext](func(ctx context.Context, e *Event[ToolExecutionContext]) error {
		endCtx = e.Data
		return nil
	}))

	agentLoop := NewAgentLoop(g, WithEventBus(bus))

	_, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-15",
			Messages:  []*ai.Message{ai.NewUserTextMessage("slow tool")},
			Config: AgentLoopConfig{
				Model: "test/duration-test",
				Tools: []string{"slow"},
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if endCtx.ToolName != "slow" {
		t.Errorf("expected tool name 'slow', got %q", endCtx.ToolName)
	}
	if endCtx.Duration <= 0 {
		t.Errorf("expected positive duration, got %v", endCtx.Duration)
	}
}

// --- Phase 2: Interrupt Tests ---

func TestAgentLoop_ToolInterrupt_SingleTool(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	mockDefineModel(g, "int-single", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return toolCallResponse(ai.ToolRequest{
			Name:  "confirm",
			Input: map[string]any{"action": "delete"},
			Ref:   "call-1",
		}), nil
	})

	mockDefineTool(g, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			return nil, tc.Interrupt(&ai.InterruptOptions{
				Metadata: map[string]any{"step": "user_confirm"},
			})
		},
	)

	agentLoop := NewAgentLoop(g)

	output, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-int-1",
			Messages:  []*ai.Message{ai.NewUserTextMessage("delete the file")},
			Config: AgentLoopConfig{
				Model: "test/int-single",
				Tools: []string{"confirm"},
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.FinishReason != ai.FinishReasonInterrupted {
		t.Errorf("expected FinishReason 'interrupted', got %q", output.FinishReason)
	}
	if len(output.Interrupts) != 1 {
		t.Fatalf("expected 1 interrupt part, got %d", len(output.Interrupts))
	}
	if output.Interrupts[0].Metadata == nil || output.Interrupts[0].Metadata["interrupt"] == nil {
		t.Error("expected interrupt metadata on interrupt part")
	}
	if output.Response.Role != ai.RoleModel {
		t.Errorf("expected response role 'model', got %q", output.Response.Role)
	}
	if output.Turns != 1 {
		t.Errorf("expected 1 turn, got %d", output.Turns)
	}
	if len(output.History) != 2 {
		t.Errorf("expected 2 messages in history, got %d", len(output.History))
	}
}

func TestAgentLoop_ToolInterrupt_BatchPartialComplete(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	mockDefineModel(g, "int-batch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return toolCallResponse(
			ai.ToolRequest{Name: "read", Input: map[string]any{}, Ref: "ref-read"},
			ai.ToolRequest{Name: "confirm", Input: map[string]any{}, Ref: "ref-confirm"},
			ai.ToolRequest{Name: "bash", Input: map[string]any{}, Ref: "ref-bash"},
		), nil
	})

	mockDefineTool(g, "read", "read tool",
		func(tc *ai.ToolContext, input ReadInput) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{Output: "file contents here"}, nil
		},
	)
	mockDefineTool(g, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			return nil, tc.Interrupt(&ai.InterruptOptions{
				Metadata: map[string]any{"step": "confirm_delete"},
			})
		},
	)
	mockDefineTool(g, "bash", "bash tool",
		func(tc *ai.ToolContext, input GenericInput) (*ai.MultipartToolResponse, error) {
			t.Fatal("bash should not be called — it comes after the interrupt")
			return nil, nil
		},
	)

	agentLoop := NewAgentLoop(g)

	output, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-int-2",
			Messages:  []*ai.Message{ai.NewUserTextMessage("read, confirm, delete")},
			Config: AgentLoopConfig{
				Model: "test/int-batch",
				Tools: []string{"read", "confirm", "bash"},
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.FinishReason != ai.FinishReasonInterrupted {
		t.Errorf("expected FinishReason 'interrupted', got %q", output.FinishReason)
	}
	// 2 interrupt parts: confirm + bash (skipped)
	if len(output.Interrupts) != 2 {
		t.Fatalf("expected 2 interrupt parts, got %d", len(output.Interrupts))
	}

	annotated := output.Response
	// Part 0 = read (completed) — should have pendingOutput
	readPart := annotated.Content[0]
	if readPart.Metadata == nil || readPart.Metadata["pendingOutput"] == nil {
		t.Error("expected pendingOutput on completed read tool request part")
	}
	// Part 1 = confirm (interrupted) — should have interrupt metadata
	confirmPart := annotated.Content[1]
	if confirmPart.Metadata == nil || confirmPart.Metadata["interrupt"] == nil {
		t.Error("expected interrupt metadata on confirm part")
	}
	interruptMeta, ok := confirmPart.Metadata["interrupt"].(map[string]any)
	if !ok {
		t.Fatalf("expected interrupt metadata to be map, got %T", confirmPart.Metadata["interrupt"])
	}
	if interruptMeta["step"] != "confirm_delete" {
		t.Errorf("expected interrupt step 'confirm_delete', got %v", interruptMeta["step"])
	}
	// Part 2 = bash (skipped) — should have interrupt=true
	bashPart := annotated.Content[2]
	if bashPart.Metadata == nil || bashPart.Metadata["interrupt"] == nil {
		t.Error("expected interrupt metadata on skipped bash part")
	}
}

func TestAgentLoop_ToolInterrupt_EventEmission(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	mockDefineModel(g, "int-events", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return toolCallResponse(ai.ToolRequest{
			Name:  "confirm",
			Input: map[string]any{},
			Ref:   "ref-1",
		}), nil
	})

	mockDefineTool(g, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			return nil, tc.Interrupt(&ai.InterruptOptions{
				Metadata: map[string]any{"reason": "needs_approval"},
			})
		},
	)

	bus := NewEventBus()
	var events []EventType
	var mu sync.Mutex
	var updateCtx ToolExecutionContext

	record := func(et EventType) {
		mu.Lock()
		events = append(events, et)
		mu.Unlock()
	}

	Subscribe(bus, AgentStart, EventHandler[AgentContext](func(ctx context.Context, e *Event[AgentContext]) error { record(AgentStart); return nil }))
	Subscribe(bus, AgentEnd, EventHandler[AgentContext](func(ctx context.Context, e *Event[AgentContext]) error { record(AgentEnd); return nil }))
	Subscribe(bus, TurnStart, EventHandler[TurnContext](func(ctx context.Context, e *Event[TurnContext]) error { record(TurnStart); return nil }))
	Subscribe(bus, TurnEnd, EventHandler[TurnContext](func(ctx context.Context, e *Event[TurnContext]) error { record(TurnEnd); return nil }))
	Subscribe(bus, MessageStart, EventHandler[MessageContext](func(ctx context.Context, e *Event[MessageContext]) error { record(MessageStart); return nil }))
	Subscribe(bus, MessageEnd, EventHandler[MessageContext](func(ctx context.Context, e *Event[MessageContext]) error { record(MessageEnd); return nil }))
	Subscribe(bus, ToolExecutionStart, EventHandler[ToolExecutionContext](func(ctx context.Context, e *Event[ToolExecutionContext]) error {
		record(ToolExecutionStart)
		return nil
	}))
	Subscribe(bus, ToolExecutionUpdate, EventHandler[ToolExecutionContext](func(ctx context.Context, e *Event[ToolExecutionContext]) error {
		record(ToolExecutionUpdate)
		updateCtx = e.Data
		return nil
	}))
	Subscribe(bus, ToolExecutionEnd, EventHandler[ToolExecutionContext](func(ctx context.Context, e *Event[ToolExecutionContext]) error { record(ToolExecutionEnd); return nil }))

	agentLoop := NewAgentLoop(g, WithEventBus(bus))

	_, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-int-3",
			Messages:  []*ai.Message{ai.NewUserTextMessage("confirm action")},
			Config: AgentLoopConfig{
				Model: "test/int-events",
				Tools: []string{"confirm"},
			},
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []EventType{
		AgentStart,
		TurnStart, MessageStart, MessageEnd,
		ToolExecutionStart, ToolExecutionUpdate, ToolExecutionEnd,
		TurnEnd,
		AgentEnd,
	}

	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d: %v", len(expected), len(events), events)
	}
	for i, e := range expected {
		if events[i] != e {
			t.Errorf("event[%d]: expected %q, got %q", i, e, events[i])
		}
	}

	if !updateCtx.Interrupted {
		t.Error("expected ToolExecutionUpdate.Interrupted to be true")
	}
	if updateCtx.InterruptMetadata == nil {
		t.Fatal("expected ToolExecutionUpdate.InterruptMetadata to be non-nil")
	}
	if updateCtx.InterruptMetadata["reason"] != "needs_approval" {
		t.Errorf("expected interrupt reason 'needs_approval', got %v", updateCtx.InterruptMetadata["reason"])
	}
}

// --- Phase 2: Resume Tests ---

func TestAgentLoop_ToolInterrupt_ResumeWithRespond(t *testing.T) {
	// Phase 1: Interrupt
	ctx := context.Background()
	g1 := newGenkitInstance(ctx)

	mockDefineModel(g1, "respond-p1", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return toolCallResponse(ai.ToolRequest{
			Name:  "confirm",
			Input: map[string]any{"action": "deploy"},
			Ref:   "ref-confirm",
		}), nil
	})
	mockDefineTool(g1, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			return nil, tc.Interrupt(&ai.InterruptOptions{
				Metadata: map[string]any{"step": "confirm"},
			})
		},
	)

	agentLoop := NewAgentLoop(g1)

	interruptOutput, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-int-4",
			Messages:  []*ai.Message{ai.NewUserTextMessage("deploy")},
			Config: AgentLoopConfig{
				Model: "test/respond-p1",
				Tools: []string{"confirm"},
			},
		},
	)
	if err != nil {
		t.Fatalf("phase 1: unexpected error: %v", err)
	}
	if interruptOutput.FinishReason != ai.FinishReasonInterrupted {
		t.Fatalf("phase 1: expected interrupted, got %q", interruptOutput.FinishReason)
	}

	// Phase 2: Resume with respond
	g2 := newGenkitInstance(ctx)

	var modelCalls atomic.Int32
	mockDefineModel(g2, "respond-p2", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		modelCalls.Add(1)
		return textResponse("Deployed successfully."), nil
	})
	mockDefineTool(g2, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			t.Fatal("confirm tool should not be re-executed during respond")
			return nil, nil
		},
	)

	respondPart := ai.NewToolResponsePart(&ai.ToolResponse{
		Name:   "confirm",
		Ref:    "ref-confirm",
		Output: &ai.MultipartToolResponse{Output: ConfirmOutput{Confirmed: true}},
	})
	respondPart.Metadata = map[string]any{"interruptResponse": true}

	agentLoop2 := NewAgentLoop(g2)

	resumeOutput, err := agentLoop2.Run(ctx,
		&AgentLoopInput{
			SessionID:     "sess-int-4",
			Messages:      interruptOutput.History,
			Config:        AgentLoopConfig{Model: "test/respond-p2", Tools: []string{"confirm"}},
			ToolResponses: []*ai.Part{respondPart},
		},
	)
	if err != nil {
		t.Fatalf("phase 2: unexpected error: %v", err)
	}
	if resumeOutput.FinishReason != ai.FinishReasonStop {
		t.Errorf("phase 2: expected FinishReason 'stop', got %q", resumeOutput.FinishReason)
	}
	if resumeOutput.Response.Text() != "Deployed successfully." {
		t.Errorf("phase 2: expected 'Deployed successfully.', got %q", resumeOutput.Response.Text())
	}
	if modelCalls.Load() != 1 {
		t.Errorf("phase 2: expected 1 model call, got %d", modelCalls.Load())
	}
	// History: user + annotated model + tool response (from resume) + final model = 4
	if len(resumeOutput.History) != 4 {
		t.Errorf("phase 2: expected 4 messages in history, got %d", len(resumeOutput.History))
	}
	if len(resumeOutput.History) >= 3 {
		toolRespMsg := resumeOutput.History[2]
		if toolRespMsg.Role != ai.RoleTool {
			t.Errorf("phase 2: expected history[2] to be tool message, got %q", toolRespMsg.Role)
		}
		if toolRespMsg.Metadata == nil || toolRespMsg.Metadata["resumed"] != true {
			t.Error("phase 2: expected tool response message to have resumed=true metadata")
		}
	}
}

func TestAgentLoop_ToolInterrupt_ResumeWithRestart(t *testing.T) {
	// Phase 1: Interrupt
	ctx := context.Background()
	g1 := newGenkitInstance(ctx)

	mockDefineModel(g1, "restart-p1", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return toolCallResponse(ai.ToolRequest{
			Name:  "confirm",
			Input: map[string]any{"action": "deploy"},
			Ref:   "ref-confirm",
		}), nil
	})
	mockDefineTool(g1, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			return nil, tc.Interrupt(&ai.InterruptOptions{
				Metadata: map[string]any{"step": "confirm"},
			})
		},
	)

	agentLoop := NewAgentLoop(g1)

	interruptOutput, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-int-5",
			Messages:  []*ai.Message{ai.NewUserTextMessage("deploy")},
			Config: AgentLoopConfig{
				Model: "test/restart-p1",
				Tools: []string{"confirm"},
			},
		},
	)
	if err != nil {
		t.Fatalf("phase 1: unexpected error: %v", err)
	}
	if interruptOutput.FinishReason != ai.FinishReasonInterrupted {
		t.Fatalf("phase 1: expected interrupted, got %q", interruptOutput.FinishReason)
	}

	// Phase 2: Resume with restart
	g2 := newGenkitInstance(ctx)

	mockDefineModel(g2, "restart-p2", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("Deployment confirmed."), nil
	})

	var restartCallCount atomic.Int32
	mockDefineTool(g2, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			restartCallCount.Add(1)
			return &ai.MultipartToolResponse{Output: ConfirmOutput{Status: "confirmed"}}, nil
		},
	)

	restartPart := ai.NewToolRequestPart(&ai.ToolRequest{
		Name:  "confirm",
		Ref:   "ref-confirm",
		Input: map[string]any{"action": "deploy", "user_approved": true},
	})
	restartPart.Metadata = map[string]any{"resumed": true}

	agentLoop2 := NewAgentLoop(g2)

	resumeOutput, err := agentLoop2.Run(ctx,
		&AgentLoopInput{
			SessionID:    "sess-int-5",
			Messages:     interruptOutput.History,
			Config:       AgentLoopConfig{Model: "test/restart-p2", Tools: []string{"confirm"}},
			ToolRestarts: []*ai.Part{restartPart},
		},
	)
	if err != nil {
		t.Fatalf("phase 2: unexpected error: %v", err)
	}
	if resumeOutput.FinishReason != ai.FinishReasonStop {
		t.Errorf("phase 2: expected FinishReason 'stop', got %q", resumeOutput.FinishReason)
	}
	if restartCallCount.Load() != 1 {
		t.Errorf("phase 2: expected tool to be called once during restart, got %d", restartCallCount.Load())
	}
	if resumeOutput.Response.Text() != "Deployment confirmed." {
		t.Errorf("phase 2: expected 'Deployment confirmed.', got %q", resumeOutput.Response.Text())
	}
	// History: user + annotated model + tool response (resumed) + final model = 4
	if len(resumeOutput.History) != 4 {
		t.Errorf("phase 2: expected 4 messages in history, got %d", len(resumeOutput.History))
	}
	if len(resumeOutput.History) >= 3 {
		toolRespMsg := resumeOutput.History[2]
		if toolRespMsg.Metadata == nil || toolRespMsg.Metadata["resumed"] != true {
			t.Error("phase 2: expected tool response message to have resumed=true metadata")
		}
	}
}

func TestAgentLoop_ToolInterrupt_ResumeRestartReinterrupts(t *testing.T) {
	// Phase 1: Interrupt
	ctx := context.Background()
	g1 := newGenkitInstance(ctx)

	mockDefineModel(g1, "reint-p1", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return toolCallResponse(ai.ToolRequest{
			Name:  "confirm",
			Input: map[string]any{},
			Ref:   "ref-confirm",
		}), nil
	})
	mockDefineTool(g1, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			return nil, tc.Interrupt(&ai.InterruptOptions{
				Metadata: map[string]any{"step": "confirm"},
			})
		},
	)

	agentLoop := NewAgentLoop(g1)

	interruptOutput, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-int-6",
			Messages:  []*ai.Message{ai.NewUserTextMessage("go")},
			Config: AgentLoopConfig{
				Model: "test/reint-p1",
				Tools: []string{"confirm"},
			},
		},
	)
	if err != nil {
		t.Fatalf("phase 1: unexpected error: %v", err)
	}

	// Phase 2: Resume with restart — tool interrupts again → FAILED_PRECONDITION
	g2 := newGenkitInstance(ctx)

	mockDefineModel(g2, "reint-p2", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		t.Fatal("model should not be called when restart re-interrupts")
		return nil, nil
	})
	mockDefineTool(g2, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			return nil, tc.Interrupt(&ai.InterruptOptions{
				Metadata: map[string]any{"step": "confirm_again"},
			})
		},
	)

	restartPart := ai.NewToolRequestPart(&ai.ToolRequest{
		Name:  "confirm",
		Ref:   "ref-confirm",
		Input: map[string]any{},
	})
	restartPart.Metadata = map[string]any{"resumed": true}

	agentLoop2 := NewAgentLoop(g2)

	_, err = agentLoop2.Run(ctx,
		&AgentLoopInput{
			SessionID:    "sess-int-6",
			Messages:     interruptOutput.History,
			Config:       AgentLoopConfig{Model: "test/reint-p2", Tools: []string{"confirm"}},
			ToolRestarts: []*ai.Part{restartPart},
		},
	)
	if err == nil {
		t.Fatal("expected error for re-interrupt during resume, got nil")
	}
	// Genkit returns FAILED_PRECONDITION about re-interruption
	if !strings.Contains(err.Error(), "interrupt") {
		t.Errorf("expected error about interrupt during resume, got %q", err.Error())
	}
}

func TestAgentLoop_ToolInterrupt_PendingOutputHandled(t *testing.T) {
	// Phase 1: read completes, confirm interrupts
	ctx := context.Background()
	g1 := newGenkitInstance(ctx)

	mockDefineModel(g1, "pending-p1", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return toolCallResponse(
			ai.ToolRequest{Name: "read", Input: map[string]any{}, Ref: "ref-read"},
			ai.ToolRequest{Name: "confirm", Input: map[string]any{}, Ref: "ref-confirm"},
		), nil
	})

	var readCallCount atomic.Int32
	mockDefineTool(g1, "read", "read tool",
		func(tc *ai.ToolContext, input ReadInput) (*ai.MultipartToolResponse, error) {
			readCallCount.Add(1)
			return &ai.MultipartToolResponse{Output: "file contents"}, nil
		},
	)
	mockDefineTool(g1, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			return nil, tc.Interrupt(&ai.InterruptOptions{
				Metadata: map[string]any{"step": "confirm"},
			})
		},
	)

	agentLoop := NewAgentLoop(g1)

	interruptOutput, err := agentLoop.Run(ctx,
		&AgentLoopInput{
			SessionID: "sess-int-7",
			Messages:  []*ai.Message{ai.NewUserTextMessage("read and confirm")},
			Config: AgentLoopConfig{
				Model: "test/pending-p1",
				Tools: []string{"read", "confirm"},
			},
		},
	)
	if err != nil {
		t.Fatalf("phase 1: unexpected error: %v", err)
	}
	if readCallCount.Load() != 1 {
		t.Fatalf("phase 1: expected read called once, got %d", readCallCount.Load())
	}

	// Phase 2: Resume with respond for confirm. Read should NOT be re-executed.
	g2 := newGenkitInstance(ctx)

	mockDefineModel(g2, "pending-p2", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("Done."), nil
	})

	var readResumeCallCount atomic.Int32
	mockDefineTool(g2, "read", "read tool",
		func(tc *ai.ToolContext, input ReadInput) (*ai.MultipartToolResponse, error) {
			readResumeCallCount.Add(1)
			t.Fatal("read tool should not be re-executed during resume — pendingOutput should be used")
			return nil, nil
		},
	)
	mockDefineTool(g2, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			t.Fatal("confirm tool should not be called during respond-resume")
			return nil, nil
		},
	)

	respondPart := ai.NewToolResponsePart(&ai.ToolResponse{
		Name:   "confirm",
		Ref:    "ref-confirm",
		Output: &ai.MultipartToolResponse{Output: ConfirmOutput{Confirmed: true}},
	})
	respondPart.Metadata = map[string]any{"interruptResponse": true}

	agentLoop2 := NewAgentLoop(g2)

	resumeOutput, err := agentLoop2.Run(ctx,
		&AgentLoopInput{
			SessionID:     "sess-int-7",
			Messages:      interruptOutput.History,
			Config:        AgentLoopConfig{Model: "test/pending-p2", Tools: []string{"read", "confirm"}},
			ToolResponses: []*ai.Part{respondPart},
		},
	)
	if err != nil {
		t.Fatalf("phase 2: unexpected error: %v", err)
	}
	if readResumeCallCount.Load() != 0 {
		t.Errorf("phase 2: read tool should not have been called, but was called %d times", readResumeCallCount.Load())
	}
	if resumeOutput.FinishReason != ai.FinishReasonStop {
		t.Errorf("phase 2: expected FinishReason 'stop', got %q", resumeOutput.FinishReason)
	}
	if resumeOutput.Response.Text() != "Done." {
		t.Errorf("phase 2: expected 'Done.', got %q", resumeOutput.Response.Text())
	}
	// History: user + annotated model + tool response (from resume) + final model = 4
	if len(resumeOutput.History) != 4 {
		t.Errorf("phase 2: expected 4 messages in history, got %d", len(resumeOutput.History))
	}
	if len(resumeOutput.History) >= 3 {
		toolRespMsg := resumeOutput.History[2]
		if toolRespMsg.Role != ai.RoleTool {
			t.Fatalf("phase 2: expected history[2] to be tool message, got %q", toolRespMsg.Role)
		}
	}
}
