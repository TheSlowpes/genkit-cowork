package flows

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
)

// --- Helpers ---

func newSessionStore() *memory.Session {
	return memory.NewSession()
}

// --- Phase 1: Core HandleMessageFlow Tests ---

func TestHandleMessage_SingleTurnNoTools(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "msg-single", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("Hello from the agent!"), nil
	})

	flow := NewHandleMessageFlow(g, store,
		WithDefaultAgentConfig(AgentLoopConfig{Model: "test/msg-single"}),
	)

	output, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "sess-msg-1",
		TenantID:  "tenant-1",
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("Hi there"),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.SessionID != "sess-msg-1" {
		t.Errorf("expected session ID 'sess-msg-1', got %q", output.SessionID)
	}
	if output.Response.Text() != "Hello from the agent!" {
		t.Errorf("expected 'Hello from the agent!', got %q", output.Response.Text())
	}
	if output.Turns != 1 {
		t.Errorf("expected 1 turn, got %d", output.Turns)
	}
	if output.FinishReason != ai.FinishReasonStop {
		t.Errorf("expected FinishReason 'stop', got %q", output.FinishReason)
	}
	if len(output.History) != 2 {
		t.Errorf("expected 2 messages in history, got %d", len(output.History))
	}
}

func TestHandleMessage_SessionPersistence(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var modelCalls atomic.Int32
	mockDefineModel(g, "msg-persist", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		switch call {
		case 1:
			return textResponse("First response"), nil
		case 2:
			// Verify history from session is included in messages
			userCount := 0
			for _, msg := range req.Messages {
				if msg.Role == ai.RoleUser {
					userCount++
				}
			}
			if userCount != 2 {
				t.Errorf("expected 2 user messages in second call, got %d", userCount)
			}
			return textResponse("Second response"), nil
		default:
			t.Fatalf("unexpected model call %d", call)
			return nil, nil
		}
	})

	flow := NewHandleMessageFlow(g, store,
		WithDefaultAgentConfig(AgentLoopConfig{Model: "test/msg-persist"}),
	)

	// First message
	output1, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "sess-persist",
		TenantID:  "tenant-1",
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("Hello"),
	})
	if err != nil {
		t.Fatalf("first message: unexpected error: %v", err)
	}
	if output1.Response.Text() != "First response" {
		t.Errorf("first message: expected 'First response', got %q", output1.Response.Text())
	}

	// Second message on the same session
	output2, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "sess-persist",
		TenantID:  "tenant-1",
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("Follow-up"),
	})
	if err != nil {
		t.Fatalf("second message: unexpected error: %v", err)
	}
	if output2.Response.Text() != "Second response" {
		t.Errorf("second message: expected 'Second response', got %q", output2.Response.Text())
	}
}

func TestHandleMessage_NewSessionCreatedOnFirstMessage(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "msg-new-sess", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("Welcome!"), nil
	})

	flow := NewHandleMessageFlow(g, store,
		WithDefaultAgentConfig(AgentLoopConfig{Model: "test/msg-new-sess"}),
	)

	output, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "brand-new-session",
		TenantID:  "tenant-new",
		Origin:    memory.WhatsAppMessage,
		Content:   *ai.NewUserTextMessage("First contact"),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.SessionID != "brand-new-session" {
		t.Errorf("expected session ID 'brand-new-session', got %q", output.SessionID)
	}

	// Verify session was persisted by loading it
	sessData, err := store.Get(ctx, "brand-new-session")
	if err != nil {
		t.Fatalf("failed to load session: %v", err)
	}
	if sessData == nil {
		t.Fatal("expected session to be persisted, got nil")
	}
	if sessData.State.TenantID != "tenant-new" {
		t.Errorf("expected tenantID 'tenant-new', got %q", sessData.State.TenantID)
	}
	// Should have user message + model response
	if len(sessData.State.Messages) != 2 {
		t.Errorf("expected 2 persisted messages, got %d", len(sessData.State.Messages))
	}
}

// --- Phase 2: Multi-Turn Tool Execution Tests ---

func TestHandleMessage_MultiTurnToolExecution(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var modelCalls atomic.Int32
	mockDefineModel(g, "msg-tools", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		switch call {
		case 1:
			return toolCallResponse(ai.ToolRequest{
				Name:  "calculator",
				Input: map[string]any{"expr": "2+2"},
				Ref:   "calc-1",
			}), nil
		case 2:
			return textResponse("The result is 4."), nil
		default:
			t.Fatalf("unexpected model call %d", call)
			return nil, nil
		}
	})

	mockDefineTool(g, "calculator", "calculator tool",
		func(tc *ai.ToolContext, input CalculatorInput) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{
				Output: map[string]any{"result": 4},
			}, nil
		},
	)

	flow := NewHandleMessageFlow(g, store,
		WithDefaultAgentConfig(AgentLoopConfig{
			Model: "test/msg-tools",
			Tools: []string{"calculator"},
		}),
	)

	output, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "sess-msg-tools",
		TenantID:  "tenant-1",
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("What is 2+2?"),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Turns != 2 {
		t.Errorf("expected 2 turns, got %d", output.Turns)
	}
	if output.Response.Text() != "The result is 4." {
		t.Errorf("expected 'The result is 4.', got %q", output.Response.Text())
	}
	if output.FinishReason != ai.FinishReasonStop {
		t.Errorf("expected FinishReason 'stop', got %q", output.FinishReason)
	}
	// History: user + model(tool call) + tool(response) + model(final) = 4
	if len(output.History) != 4 {
		t.Errorf("expected 4 messages in history, got %d", len(output.History))
	}
}

func TestHandleMessage_AllMessagesPersistedToSession(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var modelCalls atomic.Int32
	mockDefineModel(g, "msg-persist-all", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		switch call {
		case 1:
			return toolCallResponse(ai.ToolRequest{
				Name:  "echo",
				Input: map[string]any{"value": "test"},
				Ref:   "echo-1",
			}), nil
		case 2:
			return textResponse("Echo complete."), nil
		default:
			return nil, nil
		}
	})

	mockDefineTool(g, "echo", "echo tool",
		func(tc *ai.ToolContext, input GenericInput) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{Output: "echoed"}, nil
		},
	)

	flow := NewHandleMessageFlow(g, store,
		WithDefaultAgentConfig(AgentLoopConfig{
			Model: "test/msg-persist-all",
			Tools: []string{"echo"},
		}),
	)

	_, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "sess-all-msgs",
		TenantID:  "tenant-1",
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("echo test"),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	sessData, err := store.Get(ctx, "sess-all-msgs")
	if err != nil {
		t.Fatalf("failed to load session: %v", err)
	}
	if sessData == nil {
		t.Fatal("expected session to exist")
	}

	// All 4 messages should be saved: user, model(tool call), tool(response), model(final)
	msgs := sessData.State.Messages
	if len(msgs) != 4 {
		t.Fatalf("expected 4 persisted messages, got %d", len(msgs))
	}

	// Verify origins
	if msgs[0].Origin != memory.UIMessage {
		t.Errorf("msg[0]: expected origin '%s', got '%s'", memory.UIMessage, msgs[0].Origin)
	}
	if msgs[1].Origin != memory.ModelMessage {
		t.Errorf("msg[1]: expected origin '%s', got '%s'", memory.ModelMessage, msgs[1].Origin)
	}
	if msgs[2].Origin != memory.ToolMessage {
		t.Errorf("msg[2]: expected origin '%s', got '%s'", memory.ToolMessage, msgs[2].Origin)
	}
	if msgs[3].Origin != memory.ModelMessage {
		t.Errorf("msg[3]: expected origin '%s', got '%s'", memory.ModelMessage, msgs[3].Origin)
	}

	// Verify roles
	if msgs[0].Content.Role != ai.RoleUser {
		t.Errorf("msg[0]: expected role 'user', got '%s'", msgs[0].Content.Role)
	}
	if msgs[1].Content.Role != ai.RoleModel {
		t.Errorf("msg[1]: expected role 'model', got '%s'", msgs[1].Content.Role)
	}
	if msgs[2].Content.Role != ai.RoleTool {
		t.Errorf("msg[2]: expected role 'tool', got '%s'", msgs[2].Content.Role)
	}
	if msgs[3].Content.Role != ai.RoleModel {
		t.Errorf("msg[3]: expected role 'model', got '%s'", msgs[3].Content.Role)
	}
}

// --- Phase 3: Config Merging Tests ---

func TestHandleMessage_DefaultConfigOnly(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "msg-default-cfg", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("default config"), nil
	})

	flow := NewHandleMessageFlow(g, store,
		WithDefaultAgentConfig(AgentLoopConfig{
			Model:    "test/msg-default-cfg",
			MaxTurns: 5,
		}),
	)

	output, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "sess-default-cfg",
		TenantID:  "tenant-1",
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("test"),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Response.Text() != "default config" {
		t.Errorf("expected 'default config', got %q", output.Response.Text())
	}
}

func TestHandleMessage_PerRequestConfigOverride(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "msg-override-model", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("override model response"), nil
	})
	// Define the default model too so it exists
	mockDefineModel(g, "msg-default-model", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		t.Fatal("default model should not be called when overridden")
		return nil, nil
	})

	flow := NewHandleMessageFlow(g, store,
		WithDefaultAgentConfig(AgentLoopConfig{
			Model:    "test/msg-default-model",
			MaxTurns: 10,
		}),
	)

	output, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "sess-override",
		TenantID:  "tenant-1",
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("test"),
		Config: &AgentLoopConfig{
			Model: "test/msg-override-model",
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Response.Text() != "override model response" {
		t.Errorf("expected 'override model response', got %q", output.Response.Text())
	}
}

func TestHandleMessage_PerRequestConfigWithNoDefault(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "msg-input-cfg", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("input config response"), nil
	})

	flow := NewHandleMessageFlow(g, store)

	output, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "sess-input-cfg",
		TenantID:  "tenant-1",
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("test"),
		Config: &AgentLoopConfig{
			Model: "test/msg-input-cfg",
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Response.Text() != "input config response" {
		t.Errorf("expected 'input config response', got %q", output.Response.Text())
	}
}

// --- Phase 4: Config Merge Unit Tests ---

func TestMergeAgentConfig_BothNil(t *testing.T) {
	result := mergeAgentConfig(nil, nil)
	if result.Model != "" || result.Tools != nil || result.MaxTurns != 0 {
		t.Errorf("expected zero AgentConfig, got %+v", result)
	}
}

func TestMergeAgentConfig_BaseOnly(t *testing.T) {
	base := &AgentLoopConfig{Model: "base-model", MaxTurns: 5, Tools: []string{"tool-a"}}
	result := mergeAgentConfig(base, nil)
	if result.Model != "base-model" {
		t.Errorf("expected model 'base-model', got %q", result.Model)
	}
	if result.MaxTurns != 5 {
		t.Errorf("expected maxTurns 5, got %d", result.MaxTurns)
	}
	if len(result.Tools) != 1 || result.Tools[0] != "tool-a" {
		t.Errorf("expected tools [tool-a], got %v", result.Tools)
	}
}

func TestMergeAgentConfig_OverrideOnly(t *testing.T) {
	override := &AgentLoopConfig{Model: "override-model", MaxTurns: 3}
	result := mergeAgentConfig(nil, override)
	if result.Model != "override-model" {
		t.Errorf("expected model 'override-model', got %q", result.Model)
	}
	if result.MaxTurns != 3 {
		t.Errorf("expected maxTurns 3, got %d", result.MaxTurns)
	}
}

func TestMergeAgentConfig_OverrideReplacesModel(t *testing.T) {
	base := &AgentLoopConfig{Model: "base-model", MaxTurns: 5, Tools: []string{"tool-a"}}
	override := &AgentLoopConfig{Model: "new-model"}
	result := mergeAgentConfig(base, override)
	if result.Model != "new-model" {
		t.Errorf("expected model 'new-model', got %q", result.Model)
	}
	// MaxTurns and Tools should come from base
	if result.MaxTurns != 5 {
		t.Errorf("expected maxTurns 5, got %d", result.MaxTurns)
	}
	if len(result.Tools) != 1 || result.Tools[0] != "tool-a" {
		t.Errorf("expected tools [tool-a], got %v", result.Tools)
	}
}

func TestMergeAgentConfig_OverrideReplacesTools(t *testing.T) {
	base := &AgentLoopConfig{Model: "base-model", Tools: []string{"tool-a", "tool-b"}}
	override := &AgentLoopConfig{Tools: []string{"tool-c"}}
	result := mergeAgentConfig(base, override)
	// Model should come from base (override is empty string)
	if result.Model != "base-model" {
		t.Errorf("expected model 'base-model', got %q", result.Model)
	}
	// Tools should be fully replaced
	if len(result.Tools) != 1 || result.Tools[0] != "tool-c" {
		t.Errorf("expected tools [tool-c], got %v", result.Tools)
	}
}

func TestMergeAgentConfig_OverrideZeroFieldsDoNotReplace(t *testing.T) {
	base := &AgentLoopConfig{Model: "base-model", MaxTurns: 10}
	override := &AgentLoopConfig{MaxTurns: 0} // zero value should not override
	result := mergeAgentConfig(base, override)
	if result.MaxTurns != 10 {
		t.Errorf("expected maxTurns 10, got %d", result.MaxTurns)
	}
}

// --- Phase 5: Interrupt / Resume Tests ---

func TestHandleMessage_InterruptAndResume(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	// Phase 1: Interrupt
	mockDefineModel(g, "msg-int-p1", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return toolCallResponse(ai.ToolRequest{
			Name:  "confirm",
			Input: map[string]any{"action": "deploy"},
			Ref:   "ref-confirm",
		}), nil
	})
	mockDefineTool(g, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			return nil, tc.Interrupt(&ai.InterruptOptions{
				Metadata: map[string]any{"step": "confirm"},
			})
		},
	)

	flow := NewHandleMessageFlow(g, store,
		WithDefaultAgentConfig(AgentLoopConfig{
			Model: "test/msg-int-p1",
			Tools: []string{"confirm"},
		}),
	)

	interruptOutput, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "sess-msg-int",
		TenantID:  "tenant-1",
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("deploy the app"),
	})
	if err != nil {
		t.Fatalf("phase 1: unexpected error: %v", err)
	}
	if interruptOutput.FinishReason != ai.FinishReasonInterrupted {
		t.Fatalf("phase 1: expected interrupted, got %q", interruptOutput.FinishReason)
	}
	if len(interruptOutput.Interrupts) != 1 {
		t.Fatalf("phase 1: expected 1 interrupt, got %d", len(interruptOutput.Interrupts))
	}

	// Verify session persisted the interrupted state
	sessData, err := store.Get(ctx, "sess-msg-int")
	if err != nil {
		t.Fatalf("phase 1: failed to load session: %v", err)
	}
	if sessData == nil {
		t.Fatal("phase 1: expected session to exist")
	}
	// Should have: user message + annotated model message = 2
	if len(sessData.State.Messages) != 2 {
		t.Fatalf("phase 1: expected 2 persisted messages, got %d", len(sessData.State.Messages))
	}

	// Phase 2: Resume with respond
	g2 := newGenkitInstance(ctx)

	mockDefineModel(g2, "msg-int-p2", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("Deployed successfully."), nil
	})
	mockDefineTool(g2, "confirm", "confirm action",
		func(tc *ai.ToolContext, input ConfirmInput) (*ai.MultipartToolResponse, error) {
			t.Fatal("confirm should not be re-executed during respond")
			return nil, nil
		},
	)

	respondPart := ai.NewToolResponsePart(&ai.ToolResponse{
		Name:   "confirm",
		Ref:    "ref-confirm",
		Output: &ai.MultipartToolResponse{Output: ConfirmOutput{Confirmed: true}},
	})
	respondPart.Metadata = map[string]any{"interruptResponse": true}

	flow2 := NewHandleMessageFlow(g2, store,
		WithDefaultAgentConfig(AgentLoopConfig{
			Model: "test/msg-int-p2",
			Tools: []string{"confirm"},
		}),
	)

	// Resume uses the same session with ToolResponses.
	// On resume, input.Content is ignored (not appended to history)
	// because Genkit requires the last message to be the model's
	// interrupted tool-request message.
	resumeOutput, err := flow2.Run(ctx, &HandleMessageInput{
		SessionID:     "sess-msg-int",
		TenantID:      "tenant-1",
		Origin:        memory.UIMessage,
		ToolResponses: []*ai.Part{respondPart},
	})
	if err != nil {
		t.Fatalf("phase 2: unexpected error: %v", err)
	}
	if resumeOutput.FinishReason != ai.FinishReasonStop {
		t.Errorf("phase 2: expected FinishReason 'stop', got %q", resumeOutput.FinishReason)
	}
	if resumeOutput.Response.Text() != "Deployed successfully." {
		t.Errorf("phase 2: expected 'Deployed successfully.', got %q", resumeOutput.Response.Text())
	}
}

// --- Phase 6: Event Bus Integration ---

func TestHandleMessage_EventBusIntegration(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var modelCalls atomic.Int32
	mockDefineModel(g, "msg-events", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		if call == 1 {
			return toolCallResponse(ai.ToolRequest{
				Name:  "echo",
				Input: map[string]any{},
				Ref:   "ref-1",
			}), nil
		}
		return textResponse("done"), nil
	})

	mockDefineTool(g, "echo", "echo tool",
		func(tc *ai.ToolContext, input GenericInput) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{Output: "echoed"}, nil
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

	flow := NewHandleMessageFlow(g, store,
		WithHandleMessageEventBus(bus),
		WithDefaultAgentConfig(AgentLoopConfig{
			Model: "test/msg-events",
			Tools: []string{"echo"},
		}),
	)

	_, err := flow.Run(ctx, &HandleMessageInput{
		SessionID: "sess-msg-events",
		TenantID:  "tenant-1",
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("echo test"),
	})
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

// --- Phase 7: Origin Mapping ---

func TestOriginForRole(t *testing.T) {
	tests := []struct {
		name        string
		role        ai.Role
		inputOrigin memory.MessageOrigin
		expected    memory.MessageOrigin
	}{
		{"user role uses input origin", ai.RoleUser, memory.UIMessage, memory.UIMessage},
		{"user role with whatsapp", ai.RoleUser, memory.WhatsAppMessage, memory.WhatsAppMessage},
		{"model role always ModelMessage", ai.RoleModel, memory.UIMessage, memory.ModelMessage},
		{"tool role always ToolMessage", ai.RoleTool, memory.UIMessage, memory.ToolMessage},
		{"system role defaults to ModelMessage", ai.RoleSystem, memory.UIMessage, memory.ModelMessage},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := originForRole(tt.role, tt.inputOrigin)
			if got != tt.expected {
				t.Errorf("originForRole(%q, %q) = %q, want %q", tt.role, tt.inputOrigin, got, tt.expected)
			}
		})
	}
}

// --- Phase 8: Multiple Messages Accumulate In Session ---

func TestHandleMessage_MultipleMessagesAccumulateInSession(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var modelCalls atomic.Int32
	mockDefineModel(g, "msg-accum", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		return textResponse("Response " + string(rune('A'-1+call))), nil
	})

	flow := NewHandleMessageFlow(g, store,
		WithDefaultAgentConfig(AgentLoopConfig{Model: "test/msg-accum"}),
	)

	for i := range 3 {
		_, err := flow.Run(ctx, &HandleMessageInput{
			SessionID: "sess-accum",
			TenantID:  "tenant-1",
			Origin:    memory.UIMessage,
			Content:   *ai.NewUserTextMessage("Message " + string(rune('1'+i))),
		})
		if err != nil {
			t.Fatalf("message %d: unexpected error: %v", i+1, err)
		}
	}

	sessData, err := store.Get(ctx, "sess-accum")
	if err != nil {
		t.Fatalf("failed to load session: %v", err)
	}
	if sessData == nil {
		t.Fatal("expected session to exist")
	}

	// 3 rounds x 2 messages (user + model) = 6
	if len(sessData.State.Messages) != 6 {
		t.Fatalf("expected 6 persisted messages, got %d", len(sessData.State.Messages))
	}

	// Verify alternating user/model
	for i, msg := range sessData.State.Messages {
		if i%2 == 0 {
			if msg.Origin != memory.UIMessage {
				t.Errorf("msg[%d]: expected origin '%s', got '%s'", i, memory.UIMessage, msg.Origin)
			}
			if msg.Content.Role != ai.RoleUser {
				t.Errorf("msg[%d]: expected role 'user', got '%s'", i, msg.Content.Role)
			}
		} else {
			if msg.Origin != memory.ModelMessage {
				t.Errorf("msg[%d]: expected origin '%s', got '%s'", i, memory.ModelMessage, msg.Origin)
			}
			if msg.Content.Role != ai.RoleModel {
				t.Errorf("msg[%d]: expected role 'model', got '%s'", i, msg.Content.Role)
			}
		}
	}
}

// --- Phase 9: Different Origins ---

func TestHandleMessage_DifferentOrigins(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "msg-origins", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("ok"), nil
	})

	flow := NewHandleMessageFlow(g, store,
		WithDefaultAgentConfig(AgentLoopConfig{Model: "test/msg-origins"}),
	)

	origins := []memory.MessageOrigin{
		memory.UIMessage,
		memory.WhatsAppMessage,
		memory.ZoomMessage,
	}

	for _, origin := range origins {
		_, err := flow.Run(ctx, &HandleMessageInput{
			SessionID: "sess-origins",
			TenantID:  "tenant-1",
			Origin:    origin,
			Content:   *ai.NewUserTextMessage("hello from " + string(origin)),
		})
		if err != nil {
			t.Fatalf("unexpected error for origin %q: %v", origin, err)
		}
	}

	sessData, err := store.Get(ctx, "sess-origins")
	if err != nil {
		t.Fatalf("failed to load session: %v", err)
	}

	// 3 rounds x 2 messages = 6
	if len(sessData.State.Messages) != 6 {
		t.Fatalf("expected 6 messages, got %d", len(sessData.State.Messages))
	}

	// Check each user message has the correct origin
	for i, expectedOrigin := range origins {
		userMsg := sessData.State.Messages[i*2]
		if userMsg.Origin != expectedOrigin {
			t.Errorf("msg[%d]: expected origin '%s', got '%s'", i*2, expectedOrigin, userMsg.Origin)
		}
	}
}
