package flows

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
)

// --- Mock ChannelHandler ---

type mockChannelHandler struct {
	mu               sync.Mutex
	setupCalls       []string
	sendReplyCalls   []*SendReplyInput
	acknowledgeCalls []*AcknowledgeInput
	setupErr         error
	sendReplyErr     error
	acknowledgeErr   error
}

func (m *mockChannelHandler) Setup(ctx context.Context, tenantID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.setupCalls = append(m.setupCalls, tenantID)
	return m.setupErr
}

func (m *mockChannelHandler) SendReply(ctx context.Context, input *SendReplyInput) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sendReplyCalls = append(m.sendReplyCalls, input)
	return m.sendReplyErr
}

func (m *mockChannelHandler) Acknowledge(ctx context.Context, input *AcknowledgeInput) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.acknowledgeCalls = append(m.acknowledgeCalls, input)
	return m.acknowledgeErr
}

// --- Phase 1: Routing Tests ---

func TestSendReply_SenderFound(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	handler := &mockChannelHandler{}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: handler,
	}

	flow := NewSendReplyFlow(g, senders)

	result, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-1",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		Channel:   memory.WhatsAppMessage,
		Target:    HeartbeatTargetLast,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Skipped {
		t.Errorf("expected not skipped, got skipped with reason: %q", result.Reason)
	}
	if !result.Delivered {
		t.Error("expected Delivered=true")
	}
}

func TestSendReply_SenderNotFound(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: &mockChannelHandler{},
	}

	flow := NewSendReplyFlow(g, senders)

	result, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-1",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		Channel:   memory.ZoomMessage, // no handler registered for zoom
		Target:    HeartbeatTargetLast,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Skipped {
		t.Error("expected Skipped=true when sender not found")
	}
	if result.Reason == "" {
		t.Error("expected a skip reason")
	}
	if result.Delivered {
		t.Error("expected Delivered=false when skipped")
	}
}

func TestSendReply_EmptySenderMap(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	flow := NewSendReplyFlow(g, map[memory.MessageOrigin]ChannelHandler{})

	result, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-1",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		Channel:   memory.WhatsAppMessage,
		Target:    HeartbeatTargetLast,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Skipped {
		t.Error("expected Skipped=true with empty sender map")
	}
}

// --- Phase 2: Target Logic Tests ---

func TestSendReply_TargetNoneSkips(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	handler := &mockChannelHandler{}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: handler,
	}

	flow := NewSendReplyFlow(g, senders)

	result, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-1",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		Channel:   memory.WhatsAppMessage,
		Target:    HeartbeatTargetNone,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Skipped {
		t.Error("expected Skipped=true when target is none")
	}
	if result.Reason != "target is none" {
		t.Errorf("expected reason 'target is none', got %q", result.Reason)
	}
	if len(handler.sendReplyCalls) != 0 {
		t.Error("expected SendReply not to be called when target is none")
	}
}

func TestSendReply_EmptyTargetSkips(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	handler := &mockChannelHandler{}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: handler,
	}

	flow := NewSendReplyFlow(g, senders)

	result, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-1",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		Channel:   memory.WhatsAppMessage,
		Target:    "", // empty target
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Skipped {
		t.Error("expected Skipped=true when target is empty")
	}
	if len(handler.sendReplyCalls) != 0 {
		t.Error("expected SendReply not to be called when target is empty")
	}
}

func TestSendReply_TargetLastProceeds(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	handler := &mockChannelHandler{}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: handler,
	}

	flow := NewSendReplyFlow(g, senders)

	result, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-1",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		Channel:   memory.WhatsAppMessage,
		Target:    HeartbeatTargetLast,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Skipped {
		t.Error("expected not skipped with target=last")
	}
	if !result.Delivered {
		t.Error("expected Delivered=true with target=last")
	}
	if len(handler.sendReplyCalls) != 1 {
		t.Fatalf("expected 1 SendReply call, got %d", len(handler.sendReplyCalls))
	}
}

// --- Phase 3: SendReply Success Tests ---

func TestSendReply_InputPassedCorrectly(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	handler := &mockChannelHandler{}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: handler,
	}

	flow := NewSendReplyFlow(g, senders)

	username := "agent-bot"
	threadID := "thread-123"
	msgContent := &ai.Message{
		Role:    ai.RoleModel,
		Content: []*ai.Part{ai.NewTextPart("report"), ai.NewMediaPart("image/png", "https://example.com/chart.png")},
	}

	_, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-42",
		Sender: Sender{
			TenantID:    "tenant-abc",
			DisplayName: "Agent",
			Username:    &username,
		},
		Content: msgContent,
		Channel: memory.WhatsAppMessage,
		Target:  HeartbeatTargetLast,
		Destination: Destination{
			ChatID:   "chat-99",
			ThreadID: &threadID,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(handler.sendReplyCalls) != 1 {
		t.Fatalf("expected 1 SendReply call, got %d", len(handler.sendReplyCalls))
	}

	call := handler.sendReplyCalls[0]
	if call.SessionID != "sess-42" {
		t.Errorf("expected SessionID 'sess-42', got %q", call.SessionID)
	}
	if call.Sender.TenantID != "tenant-abc" {
		t.Errorf("expected TenantID 'tenant-abc', got %q", call.Sender.TenantID)
	}
	if call.Sender.DisplayName != "Agent" {
		t.Errorf("expected DisplayName 'Agent', got %q", call.Sender.DisplayName)
	}
	if call.Sender.Username == nil || *call.Sender.Username != "agent-bot" {
		t.Error("expected Username 'agent-bot'")
	}
	if call.Content != msgContent {
		t.Error("expected Content to be the same ai.Message pointer")
	}
	if len(call.Content.Content) != 2 {
		t.Errorf("expected 2 parts in Content, got %d", len(call.Content.Content))
	}
	if call.Channel != memory.WhatsAppMessage {
		t.Errorf("expected Channel WhatsAppMessage, got %q", call.Channel)
	}
	if call.Destination.ChatID != "chat-99" {
		t.Errorf("expected ChatID 'chat-99', got %q", call.Destination.ChatID)
	}
	if call.Destination.ThreadID == nil || *call.Destination.ThreadID != "thread-123" {
		t.Error("expected ThreadID 'thread-123'")
	}
}

func TestSendReply_DeliveredOutputFields(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	handler := &mockChannelHandler{}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: handler,
	}

	flow := NewSendReplyFlow(g, senders)

	result, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-out",
		Sender:    Sender{TenantID: "tenant-out"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hi")}},
		Channel:   memory.WhatsAppMessage,
		Target:    HeartbeatTargetLast,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.SessionID != "sess-out" {
		t.Errorf("expected SessionID 'sess-out', got %q", result.SessionID)
	}
	if result.Channel != memory.WhatsAppMessage {
		t.Errorf("expected Channel WhatsAppMessage, got %q", result.Channel)
	}
	if !result.Delivered {
		t.Error("expected Delivered=true")
	}
	if result.Skipped {
		t.Error("expected Skipped=false")
	}
	if result.Reason != "" {
		t.Errorf("expected no reason, got %q", result.Reason)
	}
}

// --- Phase 4: SendReply Error Tests ---

func TestSendReply_HandlerError(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	handler := &mockChannelHandler{
		sendReplyErr: errors.New("network timeout"),
	}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: handler,
	}

	flow := NewSendReplyFlow(g, senders)

	_, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-1",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		Channel:   memory.WhatsAppMessage,
		Target:    HeartbeatTargetLast,
	})
	if err == nil {
		t.Fatal("expected error from SendReply handler")
	}
	if len(handler.sendReplyCalls) != 1 {
		t.Errorf("expected SendReply to be called once, got %d", len(handler.sendReplyCalls))
	}
}

func TestSendReply_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	g := newGenkitInstance(context.Background())

	handler := &mockChannelHandler{
		sendReplyErr: context.Canceled,
	}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: handler,
	}

	flow := NewSendReplyFlow(g, senders)

	cancel() // cancel before running

	_, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-1",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		Channel:   memory.WhatsAppMessage,
		Target:    HeartbeatTargetLast,
	})
	// Either the flow infrastructure rejects the cancelled context,
	// or the handler returns its error — both result in a non-nil error.
	if err == nil {
		t.Fatal("expected error with cancelled context")
	}
}

// --- Phase 5: SetupSenders Tests ---

func TestSetupSenders_AllSucceed(t *testing.T) {
	ctx := context.Background()

	h1 := &mockChannelHandler{}
	h2 := &mockChannelHandler{}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: h1,
		memory.ZoomMessage:     h2,
	}

	err := SetupSenders(ctx, "tenant-1", senders)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(h1.setupCalls) != 1 || h1.setupCalls[0] != "tenant-1" {
		t.Errorf("expected h1 Setup called with 'tenant-1', got %v", h1.setupCalls)
	}
	if len(h2.setupCalls) != 1 || h2.setupCalls[0] != "tenant-1" {
		t.Errorf("expected h2 Setup called with 'tenant-1', got %v", h2.setupCalls)
	}
}

func TestSetupSenders_OneFails(t *testing.T) {
	ctx := context.Background()

	h1 := &mockChannelHandler{}
	h2 := &mockChannelHandler{setupErr: errors.New("webhook creation failed")}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: h1,
		memory.ZoomMessage:     h2,
	}

	err := SetupSenders(ctx, "tenant-1", senders)
	if err == nil {
		t.Fatal("expected error when one sender fails")
	}
}

func TestSetupSenders_EmptyMap(t *testing.T) {
	ctx := context.Background()

	err := SetupSenders(ctx, "tenant-1", map[memory.MessageOrigin]ChannelHandler{})
	if err != nil {
		t.Fatalf("unexpected error with empty map: %v", err)
	}
}

// --- Phase 6: Multiple Handlers Tests ---

func TestSendReply_MultipleHandlers_CorrectOneSelected(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	whatsappHandler := &mockChannelHandler{}
	zoomHandler := &mockChannelHandler{}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: whatsappHandler,
		memory.ZoomMessage:     zoomHandler,
	}

	flow := NewSendReplyFlow(g, senders)

	_, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-1",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello zoom")}},
		Channel:   memory.ZoomMessage,
		Target:    HeartbeatTargetLast,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(zoomHandler.sendReplyCalls) != 1 {
		t.Errorf("expected zoom handler called once, got %d", len(zoomHandler.sendReplyCalls))
	}
	if len(whatsappHandler.sendReplyCalls) != 0 {
		t.Error("expected whatsapp handler not called")
	}
}

func TestSendReply_MultipleHandlers_EachGetsOwnInput(t *testing.T) {
	ctx := context.Background()
	g1 := newGenkitInstance(ctx)
	g2 := newGenkitInstance(ctx)

	whatsappHandler := &mockChannelHandler{}
	zoomHandler := &mockChannelHandler{}

	// Two separate flow instances to avoid name collision
	senders1 := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: whatsappHandler,
		memory.ZoomMessage:     zoomHandler,
	}
	senders2 := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: whatsappHandler,
		memory.ZoomMessage:     zoomHandler,
	}

	flow1 := NewSendReplyFlow(g1, senders1)
	flow2 := NewSendReplyFlow(g2, senders2)

	_, err := flow1.Run(ctx, &SendReplyInput{
		SessionID: "sess-wa",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("whatsapp msg")}},
		Channel:   memory.WhatsAppMessage,
		Target:    HeartbeatTargetLast,
	})
	if err != nil {
		t.Fatalf("unexpected error on flow1: %v", err)
	}

	_, err = flow2.Run(ctx, &SendReplyInput{
		SessionID: "sess-zoom",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("zoom msg")}},
		Channel:   memory.ZoomMessage,
		Target:    HeartbeatTargetLast,
	})
	if err != nil {
		t.Fatalf("unexpected error on flow2: %v", err)
	}

	if len(whatsappHandler.sendReplyCalls) != 1 {
		t.Errorf("expected 1 whatsapp call, got %d", len(whatsappHandler.sendReplyCalls))
	}
	if len(zoomHandler.sendReplyCalls) != 1 {
		t.Errorf("expected 1 zoom call, got %d", len(zoomHandler.sendReplyCalls))
	}
	if whatsappHandler.sendReplyCalls[0].SessionID != "sess-wa" {
		t.Errorf("expected whatsapp SessionID 'sess-wa', got %q", whatsappHandler.sendReplyCalls[0].SessionID)
	}
	if zoomHandler.sendReplyCalls[0].SessionID != "sess-zoom" {
		t.Errorf("expected zoom SessionID 'sess-zoom', got %q", zoomHandler.sendReplyCalls[0].SessionID)
	}
}

// --- Phase 7: Output Fields Tests ---

func TestSendReply_SkippedOutputFields(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	handler := &mockChannelHandler{}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: handler,
	}

	flow := NewSendReplyFlow(g, senders)

	result, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-skip",
		Sender:    Sender{TenantID: "tenant-skip"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		Channel:   memory.WhatsAppMessage,
		Target:    HeartbeatTargetNone,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.SessionID != "sess-skip" {
		t.Errorf("expected SessionID 'sess-skip', got %q", result.SessionID)
	}
	if result.Channel != memory.WhatsAppMessage {
		t.Errorf("expected Channel WhatsAppMessage, got %q", result.Channel)
	}
	if !result.Skipped {
		t.Error("expected Skipped=true")
	}
	if result.Delivered {
		t.Error("expected Delivered=false")
	}
}

func TestSendReply_NotFoundOutputFields(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	flow := NewSendReplyFlow(g, map[memory.MessageOrigin]ChannelHandler{})

	result, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-nf",
		Sender:    Sender{TenantID: "tenant-nf"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		Channel:   memory.EmailMessage,
		Target:    HeartbeatTargetLast,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.SessionID != "sess-nf" {
		t.Errorf("expected SessionID 'sess-nf', got %q", result.SessionID)
	}
	if result.Channel != memory.EmailMessage {
		t.Errorf("expected Channel EmailMessage, got %q", result.Channel)
	}
	if !result.Skipped {
		t.Error("expected Skipped=true")
	}
}

// --- Phase 8: WithReplyInThread Option Test ---

func TestSendReply_WithReplyInThreadOption(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	handler := &mockChannelHandler{}
	senders := map[memory.MessageOrigin]ChannelHandler{
		memory.WhatsAppMessage: handler,
	}

	// Verify the option doesn't panic and the flow still works
	flow := NewSendReplyFlow(g, senders, WithReplyInThread())

	result, err := flow.Run(ctx, &SendReplyInput{
		SessionID: "sess-thread",
		Sender:    Sender{TenantID: "tenant-1"},
		Content:   &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("threaded")}},
		Channel:   memory.WhatsAppMessage,
		Target:    HeartbeatTargetLast,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Delivered {
		t.Error("expected Delivered=true")
	}
}
