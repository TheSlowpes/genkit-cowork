package memory

import (
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// --- test helpers ---

// mockEmbedder creates a deterministic embedder that returns a fixed-length
// vector for each document. The vector is derived from the byte values of
// the first text part, padded or truncated to the given dimension.
func mockEmbedder(g *genkit.Genkit, name string, dim int) ai.Embedder {
	return genkit.DefineEmbedder(g, "test/"+name, nil,
		func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
			embeddings := make([]*ai.Embedding, len(req.Input))
			for i, doc := range req.Input {
				vec := make([]float32, dim)
				text := ""
				if len(doc.Content) > 0 && doc.Content[0].IsText() {
					text = doc.Content[0].Text
				}
				for j := range dim {
					if j < len(text) {
						vec[j] = float32(text[j]) / 255.0
					}
				}
				embeddings[i] = &ai.Embedding{Embedding: vec}
			}
			return &ai.EmbedResponse{Embeddings: embeddings}, nil
		},
	)
}

// --- Constructor / Options ---

func TestNewDefaultVectorIndexer_DefaultDir(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	embedder := mockEmbedder(g, "embed-default", 4)

	vi := NewDefaultVectorIndexer(g, WithVectorEmbedder(embedder))
	dvi, ok := vi.(*defaultVectorIndexer)
	if !ok {
		t.Fatal("expected *defaultVectorIndexer")
	}
	if dvi.dir != os.TempDir() {
		t.Errorf("expected default dir %q, got %q", os.TempDir(), dvi.dir)
	}
	if dvi.embedder != embedder {
		t.Error("expected embedder to be set")
	}
	if dvi.stores == nil {
		t.Error("expected stores map to be initialized")
	}
}

func TestNewDefaultVectorIndexer_CustomDir(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	embedder := mockEmbedder(g, "embed-custom", 4)

	dir := t.TempDir()
	vi := NewDefaultVectorIndexer(g, WithVectorDir(dir), WithVectorEmbedder(embedder))
	dvi := vi.(*defaultVectorIndexer)
	if dvi.dir != dir {
		t.Errorf("expected dir %q, got %q", dir, dvi.dir)
	}
}

// --- Interface compliance ---

func TestDefaultVectorIndexer_ImplementsInterface(t *testing.T) {
	// Compile-time check is already in vectorstore.go via:
	//   var _ VectorIndexerOperator = (*defaultVectorIndexer)(nil)
	// This test just exercises it at runtime.
	var _ VectorIndexerOperator = &defaultVectorIndexer{}
}

// --- Per-tenant isolation via getOrCreate ---

func TestDefaultVectorIndexer_PerTenantIsolation(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	embedder := mockEmbedder(g, "embed-iso", 8)

	dir := t.TempDir()
	vi := NewDefaultVectorIndexer(g, WithVectorDir(dir), WithVectorEmbedder(embedder))

	// Index documents for tenant A.
	docsA := []*ai.Document{ai.DocumentFromText("tenant A data", nil)}
	if err := vi.Index(ctx, "tenant-a", docsA); err != nil {
		t.Fatalf("index tenant-a: %v", err)
	}

	// Index documents for tenant B.
	docsB := []*ai.Document{ai.DocumentFromText("tenant B data", nil)}
	if err := vi.Index(ctx, "tenant-b", docsB); err != nil {
		t.Fatalf("index tenant-b: %v", err)
	}

	// Retrieve for tenant A — should only find tenant A's data.
	respA, err := vi.Retrieve(ctx, "tenant-a", &ai.RetrieverRequest{
		Query: ai.DocumentFromText("tenant A data", nil),
	})
	if err != nil {
		t.Fatalf("retrieve tenant-a: %v", err)
	}
	if len(respA.Documents) != 1 {
		t.Fatalf("tenant-a: expected 1 document, got %d", len(respA.Documents))
	}
	if respA.Documents[0].Content[0].Text != "tenant A data" {
		t.Errorf("tenant-a: expected 'tenant A data', got %q", respA.Documents[0].Content[0].Text)
	}

	// Retrieve for tenant B — should only find tenant B's data.
	respB, err := vi.Retrieve(ctx, "tenant-b", &ai.RetrieverRequest{
		Query: ai.DocumentFromText("tenant B data", nil),
	})
	if err != nil {
		t.Fatalf("retrieve tenant-b: %v", err)
	}
	if len(respB.Documents) != 1 {
		t.Fatalf("tenant-b: expected 1 document, got %d", len(respB.Documents))
	}
	if respB.Documents[0].Content[0].Text != "tenant B data" {
		t.Errorf("tenant-b: expected 'tenant B data', got %q", respB.Documents[0].Content[0].Text)
	}
}

func TestDefaultVectorIndexer_GetOrCreateReusesStore(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	embedder := mockEmbedder(g, "embed-reuse", 4)

	dir := t.TempDir()
	vi := NewDefaultVectorIndexer(g, WithVectorDir(dir), WithVectorEmbedder(embedder))
	dvi := vi.(*defaultVectorIndexer)

	// First index creates the store.
	docs := []*ai.Document{ai.DocumentFromText("hello", nil)}
	if err := vi.Index(ctx, "tenant-x", docs); err != nil {
		t.Fatalf("first index: %v", err)
	}
	if len(dvi.stores) != 1 {
		t.Fatalf("expected 1 store, got %d", len(dvi.stores))
	}

	// Second index for the same tenant reuses the store.
	docs2 := []*ai.Document{ai.DocumentFromText("world", nil)}
	if err := vi.Index(ctx, "tenant-x", docs2); err != nil {
		t.Fatalf("second index: %v", err)
	}
	if len(dvi.stores) != 1 {
		t.Fatalf("expected still 1 store after second index, got %d", len(dvi.stores))
	}
}

func TestDefaultVectorIndexer_RetrieveFromEmptyStore(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	embedder := mockEmbedder(g, "embed-empty", 4)

	dir := t.TempDir()
	vi := NewDefaultVectorIndexer(g, WithVectorDir(dir), WithVectorEmbedder(embedder))

	resp, err := vi.Retrieve(ctx, "empty-tenant", &ai.RetrieverRequest{
		Query: ai.DocumentFromText("anything", nil),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Documents) != 0 {
		t.Errorf("expected 0 documents from empty store, got %d", len(resp.Documents))
	}
}

// --- Mock VectorIndexerOperator (for retrieval_test.go) compliance ---

func TestMockVectorIndexer_SatisfiesInterface(t *testing.T) {
	var _ VectorIndexerOperator = (*mockVectorIndexer)(nil)
}
