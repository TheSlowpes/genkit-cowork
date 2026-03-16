package memory

import (
	"context"

	"github.com/firebase/genkit/go/ai"
)

type EmbedderOperator interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type genkitEmbedder struct {
	embedder ai.Embedder
}

func NewGenkitEmbedder(embedder ai.Embedder) *EmbedderOperator {
	return &genkitEmbedder{embedder: embedder}
}
