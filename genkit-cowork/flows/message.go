package flows

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

func HandleMessage(g *genkit.Genkit, opts ...ai.GenerateOption) *core.Flow[*ai.Message, *ai.Message, struct{}] {
	return genkit.DefineFlow(
		g,
		"handleMessage",
		func(ctx context.Context, input *ai.Message) (*ai.Message, error) {
			opts = append(opts, ai.WithMessages(input))
			response, err := genkit.Generate(
				ctx,
				g,
				opts...,
			)
			if err != nil {
				return nil, err
			}
			return response.Message, nil
		},
	)
}
