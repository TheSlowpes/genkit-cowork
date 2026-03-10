package flows

import (
	"context"
	"fmt"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/firebase/genkit/go/genkit"
)

type HandleMessageInput struct {
	SessionID string               `json:"sessionID"`
	TenantID  string               `json:"tenantID"`
	Origin    memory.MessageOrigin `json:"origin,omitempty"`
	Content   ai.Message           `json:"content"`
}

type HandleMessageOutput struct {
	SessionID string      `json:"sessionID"`
	Response  *ai.Message `json:"response"`
}

func HandleMessageFlow(
	g *genkit.Genkit,
	store *memory.Session,
	opts ...ai.GenerateOption,
) *core.Flow[*HandleMessageInput, *HandleMessageOutput, struct{}] {
	return genkit.DefineFlow(
		g,
		"handleMessage",
		func(ctx context.Context, input *HandleMessageInput) (*HandleMessageOutput, error) {
			sess, err := session.Load(ctx, store, input.SessionID)
			if err != nil {
				sess, err = session.New(ctx,
					session.WithID[memory.SessionState](input.SessionID),
					session.WithStore(store),
					session.WithInitialState(memory.SessionState{
						TenantID: input.TenantID,
					}),
				)
				if err != nil {
					return nil, fmt.Errorf("create new session: %w", err)
				}
			}

			ctx = session.NewContext(ctx, sess)

			var history []*ai.Message
			for _, msg := range sess.State().Messages {
				history = append(history, &msg.Content)
			}

			history = append(history, &input.Content)

			callOpts := make([]ai.GenerateOption, len(opts))
			copy(callOpts, opts)
			callOpts = append(callOpts, ai.WithMessages(history...))

			response, err := genkit.Generate(
				ctx,
				g,
				callOpts...,
			)

			if err != nil {
				return nil, fmt.Errorf("generate response: %w", err)
			}

			userMsg := memory.SessionMessage{
				Origin:  input.Origin,
				Content: input.Content,
			}

			responseMsg := memory.SessionMessage{
				Origin:  memory.ModelMessage,
				Content: *response.Message,
			}

			state := sess.State()
			state.Messages = append(state.Messages, userMsg, responseMsg)
			if err := sess.UpdateState(ctx, state); err != nil {
				return nil, fmt.Errorf("update session state: %w", err)
			}

			return &HandleMessageOutput{
				SessionID: sess.ID(),
				Response:  response.Message,
			}, nil
		},
	)
}
