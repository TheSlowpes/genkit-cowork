package msgraph

import (
	"context"
	"testing"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/fake"
)

// fakeClient is a no-op GraphClient used in tests to avoid real HTTP calls.
type fakeClient struct{}

func (f *fakeClient) GetMessages(_ context.Context, _ int) ([]Message, error) {
	return []Message{
		{ID: "1", Subject: "Hello", BodyPreview: "Hi there", IsRead: false},
	}, nil
}

func (f *fakeClient) GetCalendarEvents(_ context.Context, _ int) ([]CalendarEvent, error) {
	return []CalendarEvent{
		{ID: "1", Subject: "Team sync"},
	}, nil
}

func (f *fakeClient) GetDriveItem(_ context.Context, _, _ string) (*DriveItem, error) {
	return &DriveItem{
		ID:      "abc",
		Name:    "report.txt",
		Content: []byte("hello world"),
	}, nil
}

func (f *fakeClient) UpdateDriveItem(_ context.Context, _, _ string, _ []byte) error {
	return nil
}

// fakeFactory returns a fakeClient regardless of the access token.
func fakeFactory(_ string) GraphClient {
	return &fakeClient{}
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

func TestInit_WithAccessToken(t *testing.T) {
	m := &MSGraph{AccessToken: "tok"}
	m.Init(nil)

	if m.client == nil {
		t.Fatal("expected client to be set after Init")
	}
}

func TestInit_WithCredential(t *testing.T) {
	// Use azcore/fake.TokenCredential to avoid real network calls.
	cred := &fake.TokenCredential{}
	m := &MSGraph{Credential: cred}
	m.Init(nil)

	if m.client == nil {
		t.Fatal("expected client to be set after Init with Credential")
	}
}

func TestInit_WithClientFactory(t *testing.T) {
	m := &MSGraph{ClientFactory: fakeFactory}
	m.Init(nil)

	if m.client == nil {
		t.Fatal("expected client to be set after Init with ClientFactory")
	}
}

func TestInit_PanicsOnSecondCall(t *testing.T) {
	m := &MSGraph{AccessToken: "tok"}
	m.Init(nil)

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on second Init call, got none")
		}
	}()
	m.Init(nil)
}

func TestInit_PanicsWithoutAnyAuth(t *testing.T) {
	m := &MSGraph{}
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when AccessToken, Credential, and ClientFactory are all missing")
		}
	}()
	m.Init(nil)
}

func TestInit_FactoryTakesPrecedenceOverEmptyToken(t *testing.T) {
	// ClientFactory set but AccessToken empty — should not panic.
	m := &MSGraph{ClientFactory: fakeFactory}
	m.Init(nil) // must not panic

	if m.client == nil {
		t.Fatal("expected client to be set")
	}
}

func TestInit_CredentialTakesPrecedenceOverAccessToken(t *testing.T) {
	// When Credential is set, AccessToken is unused for SDK auth.
	cred := &fake.TokenCredential{}
	m := &MSGraph{AccessToken: "ignored", Credential: cred}
	m.Init(nil)

	if m.client == nil {
		t.Fatal("expected client to be set")
	}
}

// ---------------------------------------------------------------------------
// Name
// ---------------------------------------------------------------------------

func TestName(t *testing.T) {
	m := &MSGraph{}
	if got := m.Name(); got != "msgraph" {
		t.Errorf("Name() = %q, want %q", got, "msgraph")
	}
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

func TestClient_PanicsBeforeInit(t *testing.T) {
	m := &MSGraph{}
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when calling Client() before Init()")
		}
	}()
	m.Client()
}

func TestClient_ReturnsClientAfterInit(t *testing.T) {
	m := &MSGraph{ClientFactory: fakeFactory}
	m.Init(nil)

	if c := m.Client(); c == nil {
		t.Fatal("expected non-nil client after Init")
	}
}

// ---------------------------------------------------------------------------
// GraphClient — fake implementation exercises
// ---------------------------------------------------------------------------

func TestFakeClient_GetMessages(t *testing.T) {
	c := &fakeClient{}
	msgs, err := c.GetMessages(context.Background(), 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) == 0 {
		t.Fatal("expected at least one message")
	}
	if msgs[0].Subject != "Hello" {
		t.Errorf("expected subject %q, got %q", "Hello", msgs[0].Subject)
	}
}

func TestFakeClient_GetCalendarEvents(t *testing.T) {
	c := &fakeClient{}
	events, err := c.GetCalendarEvents(context.Background(), 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) == 0 {
		t.Fatal("expected at least one event")
	}
}

func TestFakeClient_GetDriveItem(t *testing.T) {
	c := &fakeClient{}
	item, err := c.GetDriveItem(context.Background(), "drive-id-123", "item-id-456")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if item.Name != "report.txt" {
		t.Errorf("expected name %q, got %q", "report.txt", item.Name)
	}
	if string(item.Content) != "hello world" {
		t.Errorf("expected content %q, got %q", "hello world", string(item.Content))
	}
}

func TestFakeClient_UpdateDriveItem(t *testing.T) {
	c := &fakeClient{}
	if err := c.UpdateDriveItem(context.Background(), "drive-id-123", "item-id-456", []byte("new content")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

