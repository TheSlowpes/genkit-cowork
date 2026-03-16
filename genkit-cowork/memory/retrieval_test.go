package memory

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
)

// --- mock VectorIndexerOperator ---

type indexCall struct {
	tenantID string
	docs     []*ai.Document
}

type retrieveCall struct {
	tenantID string
	req      *ai.RetrieverRequest
}

type mockVectorIndexer struct {
	indexCalls    []indexCall
	retrieveCalls []retrieveCall

	indexErr     error
	retrieveResp *ai.RetrieverResponse
	retrieveErr  error
}

func (m *mockVectorIndexer) Index(ctx context.Context, tenantID string, docs []*ai.Document) error {
	m.indexCalls = append(m.indexCalls, indexCall{tenantID: tenantID, docs: docs})
	return m.indexErr
}

func (m *mockVectorIndexer) Retrieve(ctx context.Context, tenantID string, q *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
	m.retrieveCalls = append(m.retrieveCalls, retrieveCall{tenantID: tenantID, req: q})
	return m.retrieveResp, m.retrieveErr
}

var _ VectorIndexerOperator = (*mockVectorIndexer)(nil)

// --- extractTextFromMessage ---

func TestExtractTextFromMessage_SingleTextPart(t *testing.T) {
	msg := makeMessage("m1", UIMessage, ai.RoleUser, "hello world")
	text := extractTextFromMessage(msg)
	if text != "hello world" {
		t.Errorf("expected 'hello world', got %q", text)
	}
}

func TestExtractTextFromMessage_MultipleTextParts(t *testing.T) {
	msg := SessionMessage{
		MessageID: "m1",
		Origin:    UIMessage,
		Content: ai.Message{
			Role: ai.RoleUser,
			Content: []*ai.Part{
				ai.NewTextPart("hello "),
				ai.NewTextPart("world"),
			},
		},
	}
	text := extractTextFromMessage(msg)
	if text != "hello world" {
		t.Errorf("expected 'hello world', got %q", text)
	}
}

func TestExtractTextFromMessage_NoTextParts(t *testing.T) {
	msg := SessionMessage{
		MessageID: "m1",
		Origin:    ToolMessage,
		Content: ai.Message{
			Role: ai.RoleTool,
			Content: []*ai.Part{
				ai.NewToolResponsePart(&ai.ToolResponse{
					Name:   "bash",
					Ref:    "ref-1",
					Output: &ai.MultipartToolResponse{Output: "result"},
				}),
			},
		},
	}
	text := extractTextFromMessage(msg)
	if text != "" {
		t.Errorf("expected empty string for non-text parts, got %q", text)
	}
}

func TestExtractTextFromMessage_MixedParts(t *testing.T) {
	msg := SessionMessage{
		MessageID: "m1",
		Origin:    ModelMessage,
		Content: ai.Message{
			Role: ai.RoleModel,
			Content: []*ai.Part{
				ai.NewTextPart("here is the result: "),
				ai.NewToolRequestPart(&ai.ToolRequest{
					Name:  "calculator",
					Input: map[string]any{"expr": "2+2"},
					Ref:   "calc-1",
				}),
				ai.NewTextPart("done"),
			},
		},
	}
	text := extractTextFromMessage(msg)
	if text != "here is the result: done" {
		t.Errorf("expected 'here is the result: done', got %q", text)
	}
}

func TestExtractTextFromMessage_EmptyContent(t *testing.T) {
	msg := SessionMessage{
		MessageID: "m1",
		Origin:    UIMessage,
		Content: ai.Message{
			Role:    ai.RoleUser,
			Content: nil,
		},
	}
	text := extractTextFromMessage(msg)
	if text != "" {
		t.Errorf("expected empty string for nil content, got %q", text)
	}
}

// --- IndexMessages ---

func TestIndexMessages_ConvertsMessagesToDocuments(t *testing.T) {
	mock := &mockVectorIndexer{}
	r := &MessageRetriever{indexer: mock}

	ts := time.Date(2025, 6, 15, 10, 30, 0, 0, time.UTC)
	messages := []SessionMessage{
		{
			MessageID: "msg-1",
			Origin:    UIMessage,
			Content: ai.Message{
				Role:    ai.RoleUser,
				Content: []*ai.Part{ai.NewTextPart("What is the weather?")},
			},
			Timestamp: ts,
		},
		{
			MessageID: "msg-2",
			Origin:    ModelMessage,
			Content: ai.Message{
				Role:    ai.RoleModel,
				Content: []*ai.Part{ai.NewTextPart("It's sunny today.")},
			},
			Timestamp: ts.Add(time.Second),
		},
	}

	err := r.IndexMessages(context.Background(), "tenant-1", messages)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(mock.indexCalls) != 1 {
		t.Fatalf("expected 1 index call, got %d", len(mock.indexCalls))
	}
	call := mock.indexCalls[0]
	if call.tenantID != "tenant-1" {
		t.Errorf("expected tenantID 'tenant-1', got %q", call.tenantID)
	}
	if len(call.docs) != 2 {
		t.Fatalf("expected 2 documents, got %d", len(call.docs))
	}

	// Verify first document.
	doc := call.docs[0]
	if doc.Metadata["messageID"] != "msg-1" {
		t.Errorf("doc[0] messageID: expected 'msg-1', got %v", doc.Metadata["messageID"])
	}
	if doc.Metadata["origin"] != string(UIMessage) {
		t.Errorf("doc[0] origin: expected %q, got %v", UIMessage, doc.Metadata["origin"])
	}
	if doc.Metadata["role"] != string(ai.RoleUser) {
		t.Errorf("doc[0] role: expected %q, got %v", ai.RoleUser, doc.Metadata["role"])
	}
	if doc.Metadata["timestamp"] != ts.Format(time.RFC3339) {
		t.Errorf("doc[0] timestamp: expected %q, got %v", ts.Format(time.RFC3339), doc.Metadata["timestamp"])
	}

	// Verify document text content.
	if len(doc.Content) != 1 || !doc.Content[0].IsText() {
		t.Fatalf("doc[0]: expected single text part")
	}
	if doc.Content[0].Text != "What is the weather?" {
		t.Errorf("doc[0] text: expected 'What is the weather?', got %q", doc.Content[0].Text)
	}
}

func TestIndexMessages_SkipsEmptyTextMessages(t *testing.T) {
	mock := &mockVectorIndexer{}
	r := &MessageRetriever{indexer: mock}

	messages := []SessionMessage{
		{
			MessageID: "msg-tool",
			Origin:    ToolMessage,
			Content: ai.Message{
				Role: ai.RoleTool,
				Content: []*ai.Part{
					ai.NewToolResponsePart(&ai.ToolResponse{
						Name:   "bash",
						Ref:    "ref-1",
						Output: &ai.MultipartToolResponse{Output: "ok"},
					}),
				},
			},
			Timestamp: time.Now(),
		},
	}

	err := r.IndexMessages(context.Background(), "tenant-1", messages)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// No index call should be made since no documents had text.
	if len(mock.indexCalls) != 0 {
		t.Errorf("expected 0 index calls for text-less messages, got %d", len(mock.indexCalls))
	}
}

func TestIndexMessages_EmptySlice(t *testing.T) {
	mock := &mockVectorIndexer{}
	r := &MessageRetriever{indexer: mock}

	err := r.IndexMessages(context.Background(), "tenant-1", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(mock.indexCalls) != 0 {
		t.Errorf("expected 0 index calls for nil messages, got %d", len(mock.indexCalls))
	}
}

func TestIndexMessages_MixedTextAndNonText(t *testing.T) {
	mock := &mockVectorIndexer{}
	r := &MessageRetriever{indexer: mock}

	messages := []SessionMessage{
		makeMessage("msg-1", UIMessage, ai.RoleUser, "hello"),
		{
			MessageID: "msg-tool",
			Origin:    ToolMessage,
			Content: ai.Message{
				Role: ai.RoleTool,
				Content: []*ai.Part{
					ai.NewToolResponsePart(&ai.ToolResponse{
						Name:   "bash",
						Ref:    "ref-1",
						Output: &ai.MultipartToolResponse{Output: "ok"},
					}),
				},
			},
			Timestamp: time.Now(),
		},
		makeMessage("msg-3", ModelMessage, ai.RoleModel, "world"),
	}

	err := r.IndexMessages(context.Background(), "tenant-1", messages)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(mock.indexCalls) != 1 {
		t.Fatalf("expected 1 index call, got %d", len(mock.indexCalls))
	}
	// Only 2 documents should be indexed (the tool-only message is skipped).
	if len(mock.indexCalls[0].docs) != 2 {
		t.Errorf("expected 2 documents (skipping tool-only), got %d", len(mock.indexCalls[0].docs))
	}
}

func TestIndexMessages_PropagatesError(t *testing.T) {
	mock := &mockVectorIndexer{
		indexErr: fmt.Errorf("storage unavailable"),
	}
	r := &MessageRetriever{indexer: mock}

	messages := []SessionMessage{makeMessage("m1", UIMessage, ai.RoleUser, "hello")}
	err := r.IndexMessages(context.Background(), "tenant-1", messages)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if err.Error() != "storage unavailable" {
		t.Errorf("expected 'storage unavailable', got %q", err.Error())
	}
}

// --- Recall ---

func TestRecall_BuildsRequestAndReturnsDocuments(t *testing.T) {
	expectedDocs := []*ai.Document{
		ai.DocumentFromText("relevant result", nil),
	}
	mock := &mockVectorIndexer{
		retrieveResp: &ai.RetrieverResponse{Documents: expectedDocs},
	}
	r := &MessageRetriever{indexer: mock}

	opts := map[string]any{"k": 5}
	docs, err := r.Recall(context.Background(), "tenant-1", "search query", nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify the request was constructed correctly.
	if len(mock.retrieveCalls) != 1 {
		t.Fatalf("expected 1 retrieve call, got %d", len(mock.retrieveCalls))
	}
	call := mock.retrieveCalls[0]
	if call.tenantID != "tenant-1" {
		t.Errorf("expected tenantID 'tenant-1', got %q", call.tenantID)
	}
	if call.req.Query == nil {
		t.Fatal("expected non-nil query document")
	}
	if len(call.req.Query.Content) != 1 || call.req.Query.Content[0].Text != "search query" {
		t.Errorf("expected query text 'search query', got %v", call.req.Query.Content)
	}

	// Verify documents returned.
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}
	if docs[0].Content[0].Text != "relevant result" {
		t.Errorf("expected 'relevant result', got %q", docs[0].Content[0].Text)
	}
}

func TestRecall_WithMetadata(t *testing.T) {
	mock := &mockVectorIndexer{
		retrieveResp: &ai.RetrieverResponse{Documents: nil},
	}
	r := &MessageRetriever{indexer: mock}

	meta := map[string]any{"topic": "weather"}
	_, err := r.Recall(context.Background(), "tenant-1", "forecast", meta, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	call := mock.retrieveCalls[0]
	if call.req.Query.Metadata["topic"] != "weather" {
		t.Errorf("expected metadata topic 'weather', got %v", call.req.Query.Metadata["topic"])
	}
}

func TestRecall_PropagatesError(t *testing.T) {
	mock := &mockVectorIndexer{
		retrieveErr: fmt.Errorf("retrieval failed"),
	}
	r := &MessageRetriever{indexer: mock}

	_, err := r.Recall(context.Background(), "tenant-1", "query", nil, nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if err.Error() != "recall: retrieval failed" {
		t.Errorf("expected wrapped error, got %q", err.Error())
	}
}

func TestRecall_EmptyResults(t *testing.T) {
	mock := &mockVectorIndexer{
		retrieveResp: &ai.RetrieverResponse{Documents: nil},
	}
	r := &MessageRetriever{indexer: mock}

	docs, err := r.Recall(context.Background(), "tenant-1", "query", nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if docs != nil {
		t.Errorf("expected nil documents for empty results, got %v", docs)
	}
}

func TestRecall_PassesOptionsThrough(t *testing.T) {
	mock := &mockVectorIndexer{
		retrieveResp: &ai.RetrieverResponse{Documents: nil},
	}
	r := &MessageRetriever{indexer: mock}

	customOpts := map[string]any{"k": 10, "threshold": 0.8}
	_, err := r.Recall(context.Background(), "tenant-1", "query", nil, customOpts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	call := mock.retrieveCalls[0]
	opts, ok := call.req.Options.(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any options, got %T", call.req.Options)
	}
	if opts["k"] != 10 {
		t.Errorf("expected k=10, got %v", opts["k"])
	}
}

// --- NewMessageRetriever ---

func TestNewMessageRetriever_WithCustomIndexer(t *testing.T) {
	mock := &mockVectorIndexer{}
	r := NewMessageRetriever(nil, nil, WithCustomVectorIndexer(mock))

	if r.indexer != mock {
		t.Error("expected custom indexer to be used")
	}
}
