package memory

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type MessageRetriever struct {
	indexer  VectorIndexerOperator
	embedder ai.Embedder
}

type RetrieverOption func(*retrieverOptions)

type retrieverOptions struct {
	indexer VectorIndexerOperator
}

func WithCustomVectorIndexer(indexer VectorIndexerOperator) RetrieverOption {
	return func(opts *retrieverOptions) {
		opts.indexer = indexer
	}
}

func NewMessageRetriever(g *genkit.Genkit, embedder ai.Embedder, opts ...RetrieverOption) *MessageRetriever {
	options := &retrieverOptions{}
	for _, opt := range opts {
		opt(options)
	}

	indexer := options.indexer
	if indexer == nil {
		indexer = NewDefaultVectorIndexer(g, WithVectorEmbedder(embedder))
	}

	return &MessageRetriever{
		indexer:  indexer,
		embedder: embedder,
	}
}

func (r *MessageRetriever) IndexMessages(ctx context.Context, tenantId string, message []SessionMessage) error {
	var docs []*ai.Document
	for _, msg := range message {
		text := extractTextFromMessage(msg)
		if text == "" {
			continue
		}

		doc := ai.DocumentFromText(text, map[string]any{
			"messageID": msg.MessageID,
			"origin":    string(msg.Origin),
			"timestamp": msg.Timestamp.Format(time.RFC3339),
			"role":      string(msg.Content.Role),
		})
		docs = append(docs, doc)
	}

	if len(docs) == 0 {
		return nil
	}

	return r.indexer.Index(ctx, tenantId, docs)
}

func (r *MessageRetriever) Recall(ctx context.Context, tenantId string, query string, metadata map[string]any, options any) ([]*ai.Document, error) {
	req := &ai.RetrieverRequest{
		Query:   ai.DocumentFromText(query, metadata),
		Options: options,
	}

	resp, err := r.indexer.Retrieve(ctx, tenantId, req)
	if err != nil {
		return nil, fmt.Errorf("recall: %w", err)
	}

	return resp.Documents, nil
}

func extractTextFromMessage(msg SessionMessage) string {
	var b strings.Builder
	for _, part := range msg.Content.Content {
		if part.IsText() {
			b.WriteString(part.Text)
		}
	}
	return b.String()
}
