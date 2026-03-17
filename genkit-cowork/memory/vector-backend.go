package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/localvec"
)

type VectorBackend interface {
	Index(ctx context.Context, sessionID string, docs []*ai.Document) error
	Retrieve(ctx context.Context, sessionID, query string, topK int) ([]*ai.Document, error)
	Delete(ctx context.Context, sessionID string) error
}

type LocalVecConfig struct {
	Embedder        ai.Embedder
	OverFetchFactor int
}

type localVecBackend struct {
	docStore        *localvec.DocStore
	retriever       ai.Retriever
	overFetchFactor int
}

func NewLocalVecBackend(g *genkit.Genkit, name string, cfg LocalVecConfig) (VectorBackend, error) {
	if err := localvec.Init(); err != nil {
		return nil, fmt.Errorf("localvec init: %w", err)
	}

	ds, retriever, err := localvec.DefineRetriever(
		g,
		name,
		localvec.Config{Embedder: cfg.Embedder},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("localvec define retriever: %w", err)
	}

	overFetch := cfg.OverFetchFactor
	if overFetch <= 0 {
		overFetch = 10
	}

	return &localVecBackend{
		docStore:        ds,
		retriever:       retriever,
		overFetchFactor: overFetch,
	}, nil
}

func (b *localVecBackend) Index(ctx context.Context, sessionID string, docs []*ai.Document) error {
	for _, doc := range docs {
		if doc.Metadata == nil {
			doc.Metadata = make(map[string]any)
		}
		doc.Metadata["sessionID"] = sessionID
	}

	if err := localvec.Index(ctx, docs, b.docStore); err != nil {
		return fmt.Errorf("localvec index: %w", err)
	}

	return nil
}

func (b *localVecBackend) Delete(ctx context.Context, sessionID string) error {
	var toDelete []string
	for key, val := range b.docStore.Data {
		sid, ok := val.Doc.Metadata["sessionID"].(string)
		if ok && sid == sessionID {
			toDelete = append(toDelete, key)
		}
	}

	for _, key := range toDelete {
		delete(b.docStore.Data, key)
	}

	if len(toDelete) > 0 {
		if err := b.persistDocStore(); err != nil {
			return fmt.Errorf("localvec delete: persist: %w", err)
		}
	}

	return nil
}

func (b *localVecBackend) persistDocStore() error {
	data, err := json.Marshal(b.docStore.Data)
	if err != nil {
		return fmt.Errorf("marshal doc store: %w", err)
	}

	tmpFile := b.docStore.Filename + ".tmp"
	if err := os.WriteFile(tmpFile, data, 0644); err != nil {
		return fmt.Errorf("write temp file: %w", err)
	}

	if err := os.Rename(tmpFile, b.docStore.Filename); err != nil {
		return fmt.Errorf("rename temp file: %w", err)
	}

	return nil
}

func (b *localVecBackend) Retrieve(ctx context.Context, sessionID, query string, topK int) ([]*ai.Document, error) {
	fetchK := topK * b.overFetchFactor

	req := ai.RetrieverRequest{
		Query:   ai.DocumentFromText(query, nil),
		Options: localvec.RetrieverOptions{K: fetchK},
	}

	resp, err := b.retriever.Retrieve(ctx, &req)
	if err != nil {
		return nil, fmt.Errorf("localvec retrieve: %w", err)
	}

	var results []*ai.Document
	for _, doc := range resp.Documents {
		sid, ok := doc.Metadata["sessionID"].(string)
		if !ok || sid != sessionID {
			continue
		}
		results = append(results, doc)
		if len(results) >= topK {
			break
		}
	}
	return results, nil
}
