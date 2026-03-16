package memory

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/localvec"
)

type VectorIndexerOperator interface {
	Index(ctx context.Context, tenantID string, docs []*ai.Document) error
	Retrieve(ctx context.Context, tenantID string, q *ai.RetrieverRequest) (*ai.RetrieverResponse, error)
}

type VectorIndexerOption func(*vectorIndexerOptions)

type vectorIndexerOptions struct {
	dir      string
	embedder ai.Embedder
}

func WithVectorDir(dir string) VectorIndexerOption {
	return func(opts *vectorIndexerOptions) {
		opts.dir = dir
	}
}

func WithVectorEmbedder(embedder ai.Embedder) VectorIndexerOption {
	return func(opts *vectorIndexerOptions) {
		opts.embedder = embedder
	}
}

type defaultVectorIndexer struct {
	g        *genkit.Genkit
	dir      string
	embedder ai.Embedder

	mu     sync.Mutex
	stores map[string]*tenantStore
}

type tenantStore struct {
	docsStore *localvec.DocStore
	retriever ai.Retriever
}

func NewDefaultVectorIndexer(g *genkit.Genkit, opts ...VectorIndexerOption) VectorIndexerOperator {
	options := &vectorIndexerOptions{
		dir: os.TempDir(),
	}
	for _, opt := range opts {
		opt(options)
	}
	return &defaultVectorIndexer{
		g:        g,
		dir:      options.dir,
		embedder: options.embedder,
		stores:   make(map[string]*tenantStore),
	}
}

var _ VectorIndexerOperator = (*defaultVectorIndexer)(nil)

func (v *defaultVectorIndexer) getOrCreate(tenantID string) (*tenantStore, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if ts, ok := v.stores[tenantID]; ok {
		return ts, nil
	}

	name := "memory-" + tenantID
	ds, ret, err := localvec.DefineRetriever(v.g, name, localvec.Config{
		Dir:      v.dir,
		Embedder: v.embedder,
	}, nil)
	if err != nil {
		return nil, fmt.Errorf("create vector store for tenant %s: %w", tenantID, err)
	}

	ts := &tenantStore{
		docsStore: ds,
		retriever: ret,
	}
	v.stores[tenantID] = ts
	return ts, nil
}

func (v *defaultVectorIndexer) Index(ctx context.Context, tenantID string, docs []*ai.Document) error {
	ts, err := v.getOrCreate(tenantID)
	if err != nil {
		return err
	}

	return localvec.Index(ctx, docs, ts.docsStore)
}

func (v *defaultVectorIndexer) Retrieve(ctx context.Context, tenantID string, q *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
	ts, err := v.getOrCreate(tenantID)
	if err != nil {
		return nil, err
	}
	return ts.retriever.Retrieve(ctx, q)
}
