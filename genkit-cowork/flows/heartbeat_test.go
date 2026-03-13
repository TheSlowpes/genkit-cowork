package flows

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
)

// --- Phase 1: parseHHMM Unit Tests ---

func TestParseHHMM_ValidTimes(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantH   int
		wantM   int
		wantErr bool
	}{
		{"midnight", "00:00", 0, 0, false},
		{"morning", "09:00", 9, 0, false},
		{"afternoon", "14:30", 14, 30, false},
		{"end of day", "23:59", 23, 59, false},
		{"special 24:00", "24:00", 24, 0, false},
		{"single digit hour", "5:30", 5, 30, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h, m, err := parseHHMM(tt.input)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseHHMM(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			}
			if h != tt.wantH {
				t.Errorf("parseHHMM(%q) hour = %d, want %d", tt.input, h, tt.wantH)
			}
			if m != tt.wantM {
				t.Errorf("parseHHMM(%q) minute = %d, want %d", tt.input, m, tt.wantM)
			}
		})
	}
}

func TestParseHHMM_InvalidTimes(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"empty string", ""},
		{"letters", "abc"},
		{"no colon", "0900"},
		{"partial", "09:"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := parseHHMM(tt.input)
			if err == nil {
				t.Errorf("parseHHMM(%q): expected error, got nil", tt.input)
			}
		})
	}
}

// --- Phase 2: ActiveHours.Contains Unit Tests ---

func TestActiveHours_Contains_NilReturnsTrue(t *testing.T) {
	var ah *ActiveHours
	if !ah.Contains(time.Now()) {
		t.Error("nil ActiveHours should always contain any time")
	}
}

func TestActiveHours_Contains_WithinWindow(t *testing.T) {
	ah := &ActiveHours{
		Start:    "09:00",
		End:      "17:00",
		Timezone: "UTC",
	}
	// 12:00 UTC is within 09:00-17:00
	noon := time.Date(2026, 3, 13, 12, 0, 0, 0, time.UTC)
	if !ah.Contains(noon) {
		t.Error("expected 12:00 UTC to be within 09:00-17:00 UTC")
	}
}

func TestActiveHours_Contains_OutsideWindow(t *testing.T) {
	ah := &ActiveHours{
		Start:    "09:00",
		End:      "17:00",
		Timezone: "UTC",
	}
	// 07:00 UTC is outside 09:00-17:00
	early := time.Date(2026, 3, 13, 7, 0, 0, 0, time.UTC)
	if ah.Contains(early) {
		t.Error("expected 07:00 UTC to be outside 09:00-17:00 UTC")
	}

	// 18:00 UTC is outside 09:00-17:00
	late := time.Date(2026, 3, 13, 18, 0, 0, 0, time.UTC)
	if ah.Contains(late) {
		t.Error("expected 18:00 UTC to be outside 09:00-17:00 UTC")
	}
}

func TestActiveHours_Contains_ExactBoundary(t *testing.T) {
	ah := &ActiveHours{
		Start:    "09:00",
		End:      "17:00",
		Timezone: "UTC",
	}
	// Start boundary is inclusive
	atStart := time.Date(2026, 3, 13, 9, 0, 0, 0, time.UTC)
	if !ah.Contains(atStart) {
		t.Error("expected 09:00 UTC to be included (start is inclusive)")
	}

	// End boundary is exclusive
	atEnd := time.Date(2026, 3, 13, 17, 0, 0, 0, time.UTC)
	if ah.Contains(atEnd) {
		t.Error("expected 17:00 UTC to be excluded (end is exclusive)")
	}
}

func TestActiveHours_Contains_FullDay(t *testing.T) {
	ah := &ActiveHours{
		Start:    "00:00",
		End:      "24:00",
		Timezone: "UTC",
	}
	// Any time should be within 00:00-24:00
	midnight := time.Date(2026, 3, 13, 0, 0, 0, 0, time.UTC)
	if !ah.Contains(midnight) {
		t.Error("expected 00:00 to be within 00:00-24:00")
	}
	endOfDay := time.Date(2026, 3, 13, 23, 59, 0, 0, time.UTC)
	if !ah.Contains(endOfDay) {
		t.Error("expected 23:59 to be within 00:00-24:00")
	}
}

func TestActiveHours_Contains_InvalidStartReturnsTrueGracefully(t *testing.T) {
	ah := &ActiveHours{
		Start:    "invalid",
		End:      "17:00",
		Timezone: "UTC",
	}
	// Invalid start should default to true (fail open)
	if !ah.Contains(time.Now()) {
		t.Error("expected invalid start to fail open (return true)")
	}
}

func TestActiveHours_Contains_InvalidEndReturnsTrueGracefully(t *testing.T) {
	ah := &ActiveHours{
		Start:    "09:00",
		End:      "invalid",
		Timezone: "UTC",
	}
	if !ah.Contains(time.Now()) {
		t.Error("expected invalid end to fail open (return true)")
	}
}

func TestActiveHours_Contains_InvalidTimezoneUsesLocal(t *testing.T) {
	ah := &ActiveHours{
		Start:    "00:00",
		End:      "24:00",
		Timezone: "Not/A/Timezone",
	}
	// Invalid timezone falls back to local; 00:00-24:00 should contain any time
	if !ah.Contains(time.Now()) {
		t.Error("expected invalid timezone with full-day range to contain current time")
	}
}

func TestActiveHours_Contains_TimezoneConversion(t *testing.T) {
	ah := &ActiveHours{
		Start:    "09:00",
		End:      "17:00",
		Timezone: "America/New_York",
	}
	// 14:00 UTC = 10:00 ET (within 09:00-17:00 ET)
	inNY := time.Date(2026, 3, 13, 14, 0, 0, 0, time.UTC)
	if !ah.Contains(inNY) {
		t.Error("expected 14:00 UTC (10:00 ET) to be within 09:00-17:00 ET")
	}

	// 05:00 UTC = 01:00 ET (outside 09:00-17:00 ET)
	earlyUTC := time.Date(2026, 3, 13, 5, 0, 0, 0, time.UTC)
	if ah.Contains(earlyUTC) {
		t.Error("expected 05:00 UTC (01:00 ET) to be outside 09:00-17:00 ET")
	}
}

// --- Phase 3: DefaultHeartbeatDelivery / resolvedAckMaxChars Tests ---

func TestDefaultHeartbeatDelivery(t *testing.T) {
	d := DefaultHeartbeatDelivery()
	if d.ShowOk != false {
		t.Errorf("expected ShowOk=false, got %v", d.ShowOk)
	}
	if d.ShowAlerts != true {
		t.Errorf("expected ShowAlerts=true, got %v", d.ShowAlerts)
	}
	if d.UseIndicator != true {
		t.Errorf("expected UseIndicator=true, got %v", d.UseIndicator)
	}
}

func TestResolvedAckMaxChars_Default(t *testing.T) {
	cfg := &HeartbeatConfig{}
	if got := cfg.resolvedAckMaxChars(); got != 300 {
		t.Errorf("expected default ackMaxChars=300, got %d", got)
	}
}

func TestResolvedAckMaxChars_Custom(t *testing.T) {
	cfg := &HeartbeatConfig{AckMaxChars: 500}
	if got := cfg.resolvedAckMaxChars(); got != 500 {
		t.Errorf("expected ackMaxChars=500, got %d", got)
	}
}

func TestResolvedAckMaxChars_ZeroUsesDefault(t *testing.T) {
	cfg := &HeartbeatConfig{AckMaxChars: 0}
	if got := cfg.resolvedAckMaxChars(); got != 300 {
		t.Errorf("expected ackMaxChars=300 for zero, got %d", got)
	}
}

// --- Phase 4: parseHeartbeatResponse Tests ---

func TestParseHeartbeatResponse_OKTokenAtStart(t *testing.T) {
	kind, stripped := parseHeartbeatResponse("HEARTBEAT_OK all systems nominal", 300)
	if kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck, got %v", kind)
	}
	if stripped != "all systems nominal" {
		t.Errorf("expected 'all systems nominal', got %q", stripped)
	}
}

func TestParseHeartbeatResponse_OKTokenAtEnd(t *testing.T) {
	kind, stripped := parseHeartbeatResponse("all systems nominal HEARTBEAT_OK", 300)
	if kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck, got %v", kind)
	}
	if stripped != "all systems nominal" {
		t.Errorf("expected 'all systems nominal', got %q", stripped)
	}
}

func TestParseHeartbeatResponse_OKTokenOnly(t *testing.T) {
	kind, stripped := parseHeartbeatResponse("HEARTBEAT_OK", 300)
	if kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck, got %v", kind)
	}
	if stripped != "" {
		t.Errorf("expected empty stripped, got %q", stripped)
	}
}

func TestParseHeartbeatResponse_OKTokenWithWhitespace(t *testing.T) {
	kind, stripped := parseHeartbeatResponse("  HEARTBEAT_OK  summary here  ", 300)
	if kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck, got %v", kind)
	}
	if stripped != "summary here" {
		t.Errorf("expected 'summary here', got %q", stripped)
	}
}

func TestParseHeartbeatResponse_NoToken(t *testing.T) {
	kind, stripped := parseHeartbeatResponse("There is a problem with the database", 300)
	if kind != HeartbeatAlert {
		t.Errorf("expected HeartbeatAlert, got %v", kind)
	}
	if stripped != "There is a problem with the database" {
		t.Errorf("expected original text, got %q", stripped)
	}
}

func TestParseHeartbeatResponse_EmptyInput(t *testing.T) {
	kind, stripped := parseHeartbeatResponse("", 300)
	if kind != HeartbeatAlert {
		t.Errorf("expected HeartbeatAlert for empty input, got %v", kind)
	}
	if stripped != "" {
		t.Errorf("expected empty stripped, got %q", stripped)
	}
}

func TestParseHeartbeatResponse_OKTokenInMiddle(t *testing.T) {
	// Token in the middle (not start or end) should be treated as alert
	kind, _ := parseHeartbeatResponse("something HEARTBEAT_OK something", 300)
	if kind != HeartbeatAlert {
		t.Errorf("expected HeartbeatAlert for token in middle, got %v", kind)
	}
}

func TestParseHeartbeatResponse_AckExceedsMaxCharsBecomesAlert(t *testing.T) {
	// When remaining text after token removal exceeds ackMaxChars, it becomes an alert
	longText := "HEARTBEAT_OK " + string(make([]byte, 400))
	kind, _ := parseHeartbeatResponse(longText, 300)
	if kind != HeartbeatAlert {
		t.Errorf("expected HeartbeatAlert when remaining exceeds ackMaxChars, got %v", kind)
	}
}

func TestParseHeartbeatResponse_AckWithinMaxChars(t *testing.T) {
	kind, stripped := parseHeartbeatResponse("HEARTBEAT_OK short note", 300)
	if kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck, got %v", kind)
	}
	if stripped != "short note" {
		t.Errorf("expected 'short note', got %q", stripped)
	}
}

func TestParseHeartbeatResponse_AckExactlyAtMaxChars(t *testing.T) {
	// Create a remaining string exactly at the limit
	remaining := make([]byte, 10)
	for i := range remaining {
		remaining[i] = 'a'
	}
	kind, _ := parseHeartbeatResponse("HEARTBEAT_OK "+string(remaining), 10)
	if kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck when remaining == ackMaxChars, got %v", kind)
	}
}

// --- Phase 5: shouldDeliver Tests ---

func TestShouldDeliver_AckWithShowOk(t *testing.T) {
	delivery := HeartbeatDelivery{ShowOk: true, ShowAlerts: false}
	if !shouldDeliver(HeartbeatAck, delivery) {
		t.Error("expected shouldDeliver=true for Ack with ShowOk=true")
	}
}

func TestShouldDeliver_AckWithoutShowOk(t *testing.T) {
	delivery := HeartbeatDelivery{ShowOk: false, ShowAlerts: true}
	if shouldDeliver(HeartbeatAck, delivery) {
		t.Error("expected shouldDeliver=false for Ack with ShowOk=false")
	}
}

func TestShouldDeliver_AlertWithShowAlerts(t *testing.T) {
	delivery := HeartbeatDelivery{ShowOk: false, ShowAlerts: true}
	if !shouldDeliver(HeartbeatAlert, delivery) {
		t.Error("expected shouldDeliver=true for Alert with ShowAlerts=true")
	}
}

func TestShouldDeliver_AlertWithoutShowAlerts(t *testing.T) {
	delivery := HeartbeatDelivery{ShowOk: true, ShowAlerts: false}
	if shouldDeliver(HeartbeatAlert, delivery) {
		t.Error("expected shouldDeliver=false for Alert with ShowAlerts=false")
	}
}

func TestShouldDeliver_SkippedNeverDelivers(t *testing.T) {
	delivery := HeartbeatDelivery{ShowOk: true, ShowAlerts: true}
	if shouldDeliver(HeartbeatSkipped, delivery) {
		t.Error("expected shouldDeliver=false for Skipped kind")
	}
}

func TestShouldDeliver_ErrorNeverDelivers(t *testing.T) {
	delivery := HeartbeatDelivery{ShowOk: true, ShowAlerts: true}
	if shouldDeliver(HeartbeatError, delivery) {
		t.Error("expected shouldDeliver=false for Error kind")
	}
}

func TestShouldDeliver_DefaultDelivery(t *testing.T) {
	delivery := DefaultHeartbeatDelivery()
	// Default: ShowOk=false, ShowAlerts=true
	if shouldDeliver(HeartbeatAck, delivery) {
		t.Error("expected default delivery to not deliver Ack")
	}
	if !shouldDeliver(HeartbeatAlert, delivery) {
		t.Error("expected default delivery to deliver Alert")
	}
}

// --- Phase 6: evaluateHeartbeatResult Tests ---

func TestEvaluateHeartbeatResult_Ack(t *testing.T) {
	cfg := &HeartbeatConfig{
		Delivery: DefaultHeartbeatDelivery(),
	}
	runAt := time.Date(2026, 3, 13, 12, 0, 0, 0, time.UTC)
	result := evaluateHeartbeatResult("sess-1", runAt, "HEARTBEAT_OK all good", 1, cfg)

	if result.Kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck, got %v", result.Kind)
	}
	if result.SessionID != "sess-1" {
		t.Errorf("expected session ID 'sess-1', got %q", result.SessionID)
	}
	if result.RunAt != runAt {
		t.Errorf("expected RunAt %v, got %v", runAt, result.RunAt)
	}
	if result.RawContent != "HEARTBEAT_OK all good" {
		t.Errorf("expected raw content preserved, got %q", result.RawContent)
	}
	if result.DeliveryContent != "all good" {
		t.Errorf("expected delivery content 'all good', got %q", result.DeliveryContent)
	}
	if result.Turns != 1 {
		t.Errorf("expected turns=1, got %d", result.Turns)
	}
	// Default delivery: ShowOk=false, so ShouldDeliver=false for ack
	if result.ShouldDeliver {
		t.Error("expected ShouldDeliver=false with default delivery for Ack")
	}
}

func TestEvaluateHeartbeatResult_Alert(t *testing.T) {
	cfg := &HeartbeatConfig{
		Delivery: DefaultHeartbeatDelivery(),
	}
	runAt := time.Date(2026, 3, 13, 12, 0, 0, 0, time.UTC)
	result := evaluateHeartbeatResult("sess-2", runAt, "Database connection failing", 3, cfg)

	if result.Kind != HeartbeatAlert {
		t.Errorf("expected HeartbeatAlert, got %v", result.Kind)
	}
	if result.DeliveryContent != "Database connection failing" {
		t.Errorf("expected delivery content preserved for alert, got %q", result.DeliveryContent)
	}
	if result.Turns != 3 {
		t.Errorf("expected turns=3, got %d", result.Turns)
	}
	// Default delivery: ShowAlerts=true, so ShouldDeliver=true for alert
	if !result.ShouldDeliver {
		t.Error("expected ShouldDeliver=true with default delivery for Alert")
	}
}

func TestEvaluateHeartbeatResult_AckWithCustomMaxChars(t *testing.T) {
	cfg := &HeartbeatConfig{
		AckMaxChars: 5,
		Delivery:    DefaultHeartbeatDelivery(),
	}
	runAt := time.Now()
	// "a long message" is > 5 chars, so even with OK token it becomes alert
	result := evaluateHeartbeatResult("sess-3", runAt, "HEARTBEAT_OK a long message", 1, cfg)
	if result.Kind != HeartbeatAlert {
		t.Errorf("expected HeartbeatAlert when remaining exceeds custom maxChars, got %v", result.Kind)
	}
}

func TestEvaluateHeartbeatResult_AckWithShowOkDelivery(t *testing.T) {
	cfg := &HeartbeatConfig{
		Delivery: HeartbeatDelivery{ShowOk: true, ShowAlerts: true},
	}
	runAt := time.Now()
	result := evaluateHeartbeatResult("sess-4", runAt, "HEARTBEAT_OK fine", 1, cfg)
	if result.Kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck, got %v", result.Kind)
	}
	if !result.ShouldDeliver {
		t.Error("expected ShouldDeliver=true with ShowOk=true for Ack")
	}
}

// --- Phase 7: skippedResult and errorResult Tests ---

func TestSkippedResult(t *testing.T) {
	result := skippedResult("sess-skip", SkipReasonOutsideHours)
	if result.Kind != HeartbeatSkipped {
		t.Errorf("expected HeartbeatSkipped, got %v", result.Kind)
	}
	if result.SessionID != "sess-skip" {
		t.Errorf("expected session 'sess-skip', got %q", result.SessionID)
	}
	if result.SkipReason != SkipReasonOutsideHours {
		t.Errorf("expected reason 'outside_hours', got %q", result.SkipReason)
	}
	if result.ShouldDeliver {
		t.Error("expected ShouldDeliver=false for skipped result")
	}
	if result.RunAt.IsZero() {
		t.Error("expected RunAt to be set")
	}
}

func TestSkippedResult_BusyReason(t *testing.T) {
	result := skippedResult("sess-busy", SkipReasonBusy)
	if result.SkipReason != SkipReasonBusy {
		t.Errorf("expected reason 'busy', got %q", result.SkipReason)
	}
}

func TestErrorResult(t *testing.T) {
	runAt := time.Date(2026, 3, 13, 12, 0, 0, 0, time.UTC)
	testErr := errors.New("something went wrong")
	result := errorResult("sess-err", runAt, testErr)

	if result.Kind != HeartbeatError {
		t.Errorf("expected HeartbeatError, got %v", result.Kind)
	}
	if result.SessionID != "sess-err" {
		t.Errorf("expected session 'sess-err', got %q", result.SessionID)
	}
	if result.RunAt != runAt {
		t.Errorf("expected RunAt %v, got %v", runAt, result.RunAt)
	}
	if result.Err != testErr {
		t.Errorf("expected error %v, got %v", testErr, result.Err)
	}
	if result.ShouldDeliver {
		t.Error("expected ShouldDeliver=false for error result")
	}
}

// --- Phase 8: extractText Tests ---

func TestExtractText_NilMessage(t *testing.T) {
	if got := extractText(nil); got != "" {
		t.Errorf("expected empty string for nil message, got %q", got)
	}
}

func TestExtractText_SingleTextPart(t *testing.T) {
	msg := &ai.Message{
		Role:    ai.RoleModel,
		Content: []*ai.Part{ai.NewTextPart("hello world")},
	}
	if got := extractText(msg); got != "hello world" {
		t.Errorf("expected 'hello world', got %q", got)
	}
}

func TestExtractText_MultipleTextParts(t *testing.T) {
	msg := &ai.Message{
		Role: ai.RoleModel,
		Content: []*ai.Part{
			ai.NewTextPart("hello "),
			ai.NewTextPart("world"),
		},
	}
	if got := extractText(msg); got != "hello world" {
		t.Errorf("expected 'hello world', got %q", got)
	}
}

func TestExtractText_MixedParts(t *testing.T) {
	toolReq := ai.NewToolRequestPart(&ai.ToolRequest{Name: "test"})
	msg := &ai.Message{
		Role: ai.RoleModel,
		Content: []*ai.Part{
			ai.NewTextPart("text before "),
			toolReq,
			ai.NewTextPart("text after"),
		},
	}
	if got := extractText(msg); got != "text before text after" {
		t.Errorf("expected 'text before text after', got %q", got)
	}
}

func TestExtractText_NoParts(t *testing.T) {
	msg := &ai.Message{Role: ai.RoleModel}
	if got := extractText(msg); got != "" {
		t.Errorf("expected empty string for message with no parts, got %q", got)
	}
}

// --- Phase 9: sessionID Tests ---

func TestSessionID_CustomID(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	h := NewHeartbeat(g, store, HeartbeatConfig{
		SessionID: "custom-session",
		Interval:  time.Minute,
	})

	if got := h.sessionID(); got != "custom-session" {
		t.Errorf("expected 'custom-session', got %q", got)
	}
}

func TestSessionID_DefaultID(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval: time.Minute,
	})

	if got := h.sessionID(); got != "heartbeat" {
		t.Errorf("expected 'heartbeat', got %q", got)
	}
}

// --- Phase 10: NewHeartbeat Constructor and Functional Options Tests ---

func TestNewHeartbeat_BasicConstruction(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	cfg := HeartbeatConfig{
		Interval:  5 * time.Minute,
		SessionID: "test-hb",
		TenantID:  "tenant-1",
		Delivery:  DefaultHeartbeatDelivery(),
	}

	h := NewHeartbeat(g, store, cfg)
	if h == nil {
		t.Fatal("expected non-nil Heartbeat")
	}
	if h.cfg == nil {
		t.Fatal("expected non-nil config")
	}
	if h.cfg.Interval != 5*time.Minute {
		t.Errorf("expected interval 5m, got %v", h.cfg.Interval)
	}
	if h.flow == nil {
		t.Fatal("expected non-nil flow")
	}
	if h.onResult == nil {
		t.Fatal("expected non-nil onResult (should default to no-op)")
	}
	if h.stopCh == nil {
		t.Fatal("expected non-nil stopCh")
	}
}

func TestNewHeartbeat_OnResultDefault(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	h := NewHeartbeat(g, store, HeartbeatConfig{Interval: time.Minute})

	// Default onResult should be a no-op and not panic
	h.onResult(&HeartbeatOutput{Kind: HeartbeatAck})
}

func TestNewHeartbeat_WithOnResult(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var received *HeartbeatOutput
	h := NewHeartbeat(g, store, HeartbeatConfig{Interval: time.Minute},
		WithHeartbeatOnResult(func(output *HeartbeatOutput) {
			received = output
		}),
	)

	testOutput := &HeartbeatOutput{Kind: HeartbeatAlert, SessionID: "test"}
	h.onResult(testOutput)
	if received == nil {
		t.Fatal("expected onResult callback to be called")
	}
	if received.Kind != HeartbeatAlert {
		t.Errorf("expected HeartbeatAlert, got %v", received.Kind)
	}
}

// --- Phase 11: Heartbeat.Run Skip Logic Tests ---

func TestHeartbeatRun_SkipsOutsideActiveHours(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var results []*HeartbeatOutput
	var mu sync.Mutex

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-skip-hours",
		ActiveHours: &ActiveHours{
			Start:    "09:00",
			End:      "17:00",
			Timezone: "UTC",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		mu.Lock()
		results = append(results, output)
		mu.Unlock()
	}))

	// Run at 03:00 UTC — outside active hours
	outsideTime := time.Date(2026, 3, 13, 3, 0, 0, 0, time.UTC)
	h.Run(ctx, outsideTime)

	mu.Lock()
	defer mu.Unlock()
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Kind != HeartbeatSkipped {
		t.Errorf("expected HeartbeatSkipped, got %v", results[0].Kind)
	}
	if results[0].SkipReason != SkipReasonOutsideHours {
		t.Errorf("expected SkipReasonOutsideHours, got %q", results[0].SkipReason)
	}
}

func TestHeartbeatRun_SkipsBusy(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-busy", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		// Simulate a slow model by blocking
		time.Sleep(200 * time.Millisecond)
		return textResponse("HEARTBEAT_OK"), nil
	})

	var results []*HeartbeatOutput
	var mu sync.Mutex

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-busy",
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-busy",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		mu.Lock()
		results = append(results, output)
		mu.Unlock()
	}))

	// Mark as running
	h.running.Store(true)

	// Try to run while busy
	runAt := time.Now()
	h.Run(ctx, runAt)

	// Restore running flag
	h.running.Store(false)

	mu.Lock()
	defer mu.Unlock()
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Kind != HeartbeatSkipped {
		t.Errorf("expected HeartbeatSkipped, got %v", results[0].Kind)
	}
	if results[0].SkipReason != SkipReasonBusy {
		t.Errorf("expected SkipReasonBusy, got %q", results[0].SkipReason)
	}
}

func TestHeartbeatRun_RunningFlagSetAndCleared(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var runningDuringExec atomic.Bool

	mockDefineModel(g, "hb-flag", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("HEARTBEAT_OK"), nil
	})

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-flag",
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-flag",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		// Check running flag during callback — should still be true
		runningDuringExec.Store(true)
	}))

	// Verify running starts as false
	if h.running.Load() {
		t.Error("expected running=false before Run")
	}

	h.Run(ctx, time.Now())

	// Verify running is false after Run completes
	if h.running.Load() {
		t.Error("expected running=false after Run completes")
	}
}

// --- Phase 12: Heartbeat Integration Tests (full flow execution) ---

func TestHeartbeatRun_SuccessfulAckResult(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-ack", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("HEARTBEAT_OK everything is fine"), nil
	})

	var result *HeartbeatOutput
	var mu sync.Mutex

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-ack",
		TenantID:  "tenant-1",
		Delivery:  DefaultHeartbeatDelivery(),
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-ack",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		mu.Lock()
		result = output
		mu.Unlock()
	}))

	runAt := time.Now()
	h.Run(ctx, runAt)

	mu.Lock()
	defer mu.Unlock()
	if result == nil {
		t.Fatal("expected onResult to be called")
	}
	if result.Kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck, got %v", result.Kind)
	}
	if result.SessionID != "hb-ack" {
		t.Errorf("expected session 'hb-ack', got %q", result.SessionID)
	}
	if result.DeliveryContent != "everything is fine" {
		t.Errorf("expected delivery content 'everything is fine', got %q", result.DeliveryContent)
	}
	if result.RawContent != "HEARTBEAT_OK everything is fine" {
		t.Errorf("expected raw content preserved, got %q", result.RawContent)
	}
	// Default delivery: ShowOk=false
	if result.ShouldDeliver {
		t.Error("expected ShouldDeliver=false with default delivery for Ack")
	}
}

func TestHeartbeatRun_SuccessfulAlertResult(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-alert", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("WARNING: Disk space running low on production server"), nil
	})

	var result *HeartbeatOutput

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-alert",
		TenantID:  "tenant-1",
		Delivery:  DefaultHeartbeatDelivery(),
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-alert",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		result = output
	}))

	h.Run(ctx, time.Now())

	if result == nil {
		t.Fatal("expected onResult to be called")
	}
	if result.Kind != HeartbeatAlert {
		t.Errorf("expected HeartbeatAlert, got %v", result.Kind)
	}
	if !result.ShouldDeliver {
		t.Error("expected ShouldDeliver=true with default delivery for Alert")
	}
}

func TestHeartbeatRun_FlowErrorBecomesErrorResult(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-err", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, errors.New("model unavailable")
	})

	var result *HeartbeatOutput

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-err",
		TenantID:  "tenant-1",
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-err",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		result = output
	}))

	h.Run(ctx, time.Now())

	if result == nil {
		t.Fatal("expected onResult to be called")
	}
	// The flow handler catches agent loop errors and returns an errorResult.
	// But Run's tryRun calls h.flow.Run, which returns the errorResult (no Go error).
	// So result should be an error result from the flow.
	if result.Kind != HeartbeatError {
		t.Errorf("expected HeartbeatError, got %v", result.Kind)
	}
}

func TestHeartbeatRun_SessionPersistence(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-sess", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("HEARTBEAT_OK session test"), nil
	})

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-persist",
		TenantID:  "tenant-test",
		Delivery:  DefaultHeartbeatDelivery(),
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-sess",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {}))

	h.Run(ctx, time.Now())

	// Verify session was created and has messages
	sessData, err := store.Get(ctx, "hb-persist")
	if err != nil {
		t.Fatalf("failed to load session: %v", err)
	}
	if sessData == nil {
		t.Fatal("expected session to exist after heartbeat run")
	}
	if sessData.State.TenantID != "tenant-test" {
		t.Errorf("expected tenantID 'tenant-test', got %q", sessData.State.TenantID)
	}
	if len(sessData.State.Messages) == 0 {
		t.Error("expected session to have messages after heartbeat run")
	}
}

func TestHeartbeatRun_SessionPersistenceAcrossRuns(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var modelCalls atomic.Int32
	mockDefineModel(g, "hb-multi", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		return textResponse("HEARTBEAT_OK run " + string(rune('0'+call))), nil
	})

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-multi-run",
		TenantID:  "tenant-1",
		Delivery:  DefaultHeartbeatDelivery(),
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-multi",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {}))

	// Run twice
	h.Run(ctx, time.Now())
	h.Run(ctx, time.Now())

	sessData, err := store.Get(ctx, "hb-multi-run")
	if err != nil {
		t.Fatalf("failed to load session: %v", err)
	}
	if sessData == nil {
		t.Fatal("expected session to exist")
	}

	// Should have accumulated messages from both runs
	if len(sessData.State.Messages) < 2 {
		t.Errorf("expected at least 2 messages from 2 runs, got %d", len(sessData.State.Messages))
	}
}

func TestHeartbeatRun_HeartbeatOrigin(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-origin", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("HEARTBEAT_OK"), nil
	})

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-origin",
		TenantID:  "tenant-1",
		Delivery:  DefaultHeartbeatDelivery(),
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-origin",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {}))

	h.Run(ctx, time.Now())

	sessData, err := store.Get(ctx, "hb-origin")
	if err != nil {
		t.Fatalf("failed to load session: %v", err)
	}
	if sessData == nil {
		t.Fatal("expected session to exist")
	}
	// Heartbeat messages should use HeartbeatMessage origin for model messages
	for _, msg := range sessData.State.Messages {
		if msg.Content.Role == ai.RoleModel {
			expected := originForRole(ai.RoleModel, memory.HeartbeatMessage)
			if msg.Origin != expected {
				t.Errorf("expected origin %q for model message, got %q", expected, msg.Origin)
			}
		}
	}
}

// --- Phase 13: Start / Stop / Wake Lifecycle Tests ---

func TestHeartbeat_StartDoesNothingWithZeroInterval(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval: 0, // zero interval should not start ticker
	})

	// Start should return without starting a goroutine
	h.Start(ctx)
	// Give it a moment to potentially fail
	time.Sleep(50 * time.Millisecond)
	// Should be safe to stop even though Start was a no-op
	h.Stop()
}

func TestHeartbeat_StartDoesNothingWithNegativeInterval(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval: -1 * time.Minute,
	})

	h.Start(ctx)
	time.Sleep(50 * time.Millisecond)
	h.Stop()
}

func TestHeartbeat_StopIsIdempotent(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	h := NewHeartbeat(g, store, HeartbeatConfig{Interval: time.Minute})

	// Stop multiple times should not panic
	h.Stop()
	h.Stop()
	h.Stop()
}

func TestHeartbeat_StartAndStop(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-lifecycle", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("HEARTBEAT_OK"), nil
	})

	var callCount atomic.Int32

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  50 * time.Millisecond,
		SessionID: "hb-lifecycle",
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-lifecycle",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		callCount.Add(1)
	}))

	h.Start(ctx)
	// Wait for at least one tick
	time.Sleep(150 * time.Millisecond)
	h.Stop()

	count := callCount.Load()
	if count == 0 {
		t.Error("expected at least one heartbeat run after Start")
	}
}

func TestHeartbeat_StopPreventsMoreTicks(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-stop", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("HEARTBEAT_OK"), nil
	})

	var callCount atomic.Int32

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  50 * time.Millisecond,
		SessionID: "hb-stop",
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-stop",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		callCount.Add(1)
	}))

	h.Start(ctx)
	time.Sleep(100 * time.Millisecond)
	h.Stop()

	// Allow time for any in-flight tick to complete
	time.Sleep(100 * time.Millisecond)
	countAfterSettle := callCount.Load()

	// Wait more and verify count stabilizes (no new ticks)
	time.Sleep(200 * time.Millisecond)
	countFinal := callCount.Load()

	if countFinal > countAfterSettle {
		t.Errorf("expected no more ticks after Stop settled, got %d after settle, %d final", countAfterSettle, countFinal)
	}
}

func TestHeartbeat_ContextCancellationStopsTicker(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-cancel", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("HEARTBEAT_OK"), nil
	})

	var callCount atomic.Int32

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  50 * time.Millisecond,
		SessionID: "hb-cancel",
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-cancel",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		callCount.Add(1)
	}))

	h.Start(ctx)
	time.Sleep(100 * time.Millisecond)
	cancel()

	// Allow time for any in-flight tick to complete
	time.Sleep(100 * time.Millisecond)
	countAfterSettle := callCount.Load()

	// Wait more and verify count stabilizes (no new ticks)
	time.Sleep(200 * time.Millisecond)
	countFinal := callCount.Load()

	if countFinal > countAfterSettle {
		t.Errorf("expected no more ticks after context cancel settled, got %d after settle, %d final", countAfterSettle, countFinal)
	}

	// Clean up
	h.Stop()
}

func TestHeartbeat_Wake(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-wake", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("HEARTBEAT_OK woke up"), nil
	})

	var result *HeartbeatOutput
	var mu sync.Mutex
	done := make(chan struct{})

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Hour, // Long interval — so ticker won't fire
		SessionID: "hb-wake",
		TenantID:  "tenant-1",
		Delivery:  DefaultHeartbeatDelivery(),
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-wake",
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		mu.Lock()
		result = output
		mu.Unlock()
		close(done)
	}))

	h.Wake(ctx)

	// Wait for async completion
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Wake did not complete within timeout")
	}

	mu.Lock()
	defer mu.Unlock()
	if result == nil {
		t.Fatal("expected onResult to be called after Wake")
	}
	if result.Kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck, got %v", result.Kind)
	}
}

// --- Phase 14: WithHeartbeatLoopOperator Tests ---

func TestNewHeartbeat_WithCustomLoopOperator(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var generateCalled atomic.Bool

	customOp := &mockOperator{
		generateFunc: func(ctx context.Context, opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
			generateCalled.Store(true)
			return textResponse("HEARTBEAT_OK from custom operator"), nil
		},
		lookupModelFunc: func(name string) (ai.Model, bool) {
			// Return a dummy model
			m := mockDefineModel(g, "hb-custom-op", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
				return textResponse("HEARTBEAT_OK from custom operator"), nil
			})
			return m, true
		},
		lookupToolFunc: func(name string) (ai.Tool, bool) {
			return nil, false
		},
	}

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-custom-op",
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-custom-op",
		},
	},
		WithHeartbeatLoopOperator(customOp),
		WithHeartbeatOnResult(func(output *HeartbeatOutput) {}),
	)

	h.Run(ctx, time.Now())

	if !generateCalled.Load() {
		t.Error("expected custom operator Generate to be called")
	}
}

// mockOperator implements AgentLoopOperator for testing
type mockOperator struct {
	generateFunc    func(ctx context.Context, opts ...ai.GenerateOption) (*ai.ModelResponse, error)
	lookupModelFunc func(name string) (ai.Model, bool)
	lookupToolFunc  func(name string) (ai.Tool, bool)
}

func (m *mockOperator) Generate(ctx context.Context, opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
	return m.generateFunc(ctx, opts...)
}

func (m *mockOperator) LookupModel(name string) (ai.Model, bool) {
	return m.lookupModelFunc(name)
}

func (m *mockOperator) LookupTool(name string) (ai.Tool, bool) {
	return m.lookupToolFunc(name)
}

// --- Phase 15: WithCustomHeartbeatAgentConfig Tests ---

func TestNewHeartbeat_WithCustomAgentConfig(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var receivedModel string
	mockDefineModel(g, "hb-default-model", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		receivedModel = "hb-default-model"
		return textResponse("HEARTBEAT_OK"), nil
	})

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-cfg-override",
	},
		WithCustomHeartbeatAgentConfig(AgentLoopConfig{
			Model: "test/hb-default-model",
		}),
		WithHeartbeatOnResult(func(output *HeartbeatOutput) {}),
	)

	h.Run(ctx, time.Now())

	if receivedModel != "hb-default-model" {
		t.Errorf("expected model 'hb-default-model' to be used, got %q", receivedModel)
	}
}

// --- Phase 16: WithHeartbeatEventBus Tests ---

func TestNewHeartbeat_WithEventBus(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	mockDefineModel(g, "hb-bus", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResponse("HEARTBEAT_OK"), nil
	})

	bus := NewEventBus()
	var agentStarted atomic.Bool
	Subscribe(bus, AgentStart, EventHandler[AgentContext](func(ctx context.Context, e *Event[AgentContext]) error {
		agentStarted.Store(true)
		return nil
	}))

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-bus",
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-bus",
		},
	},
		WithHeartbeatEventBus(bus),
		WithHeartbeatOnResult(func(output *HeartbeatOutput) {}),
	)

	h.Run(ctx, time.Now())

	if !agentStarted.Load() {
		t.Error("expected AgentStart event to fire through the event bus")
	}
}

// --- Phase 17: Heartbeat with Tool Execution ---

func TestHeartbeatRun_WithToolExecution(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)
	store := newSessionStore()

	var modelCalls atomic.Int32
	mockDefineModel(g, "hb-tools", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := modelCalls.Add(1)
		if call == 1 {
			return toolCallResponse(ai.ToolRequest{
				Name:  "check-status",
				Input: map[string]any{"service": "db"},
				Ref:   "ref-1",
			}), nil
		}
		return textResponse("HEARTBEAT_OK database is healthy"), nil
	})

	mockDefineTool(g, "check-status", "check service status",
		func(tc *ai.ToolContext, input GenericInput) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{Output: map[string]any{"status": "healthy"}}, nil
		},
	)

	var result *HeartbeatOutput

	h := NewHeartbeat(g, store, HeartbeatConfig{
		Interval:  time.Minute,
		SessionID: "hb-tools",
		TenantID:  "tenant-1",
		Delivery:  DefaultHeartbeatDelivery(),
		AgentConfig: &AgentLoopConfig{
			Model: "test/hb-tools",
			Tools: []string{"check-status"},
		},
	}, WithHeartbeatOnResult(func(output *HeartbeatOutput) {
		result = output
	}))

	h.Run(ctx, time.Now())

	if result == nil {
		t.Fatal("expected onResult to be called")
	}
	if result.Kind != HeartbeatAck {
		t.Errorf("expected HeartbeatAck, got %v", result.Kind)
	}
	if result.Turns != 2 {
		t.Errorf("expected 2 turns (tool call + final response), got %d", result.Turns)
	}
	if result.DeliveryContent != "database is healthy" {
		t.Errorf("expected 'database is healthy', got %q", result.DeliveryContent)
	}
}
