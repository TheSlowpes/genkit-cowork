package memory

import (
	"context"

	"github.com/firebase/genkit/go/ai"
)

type VectorStoreOperator interface {
	Index(ctx context.Context, docs []*ai.Document) error
	Retriever(ctx context.Context, q *ai.RetrieverRequest) (*ai.RetrieverResponse, error)
}
