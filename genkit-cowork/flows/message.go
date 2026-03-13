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
	SessionID     string               `json:"sessionID"`
	TenantID      string               `json:"tenantID"`
	Origin        memory.MessageOrigin `json:"origin,omitempty"`
	Content       ai.Message           `json:"content"`
	Config        *AgentLoopConfig     `json:"config,omitempty"`
	ToolResponses []*ai.Part           `json:"toolResponses,omitempty"`
	ToolRestarts  []*ai.Part           `json:"toolRestarts,omitempty"`
}

type HandleMessageOutput struct {
	SessionID    string          `json:"sessionID"`
	Response     *ai.Message     `json:"response"`
	History      []*ai.Message   `json:"history,omitempty"`
	Turns        int             `json:"turns"`
	FinishReason ai.FinishReason `json:"finishReason"`
	Interrupts   []*ai.Part      `json:"interrupts,omitempty"`
}

type handleMessageOptions struct {
	bus           *EventBus
	baseOpts      []ai.GenerateOption
	loopOperator  AgentLoopOperator
	defaultConfig *AgentLoopConfig
}

type HandleMessageOption func(*handleMessageOptions)

func WithHandleMessageEventBus(bus *EventBus) HandleMessageOption {
	return func(opts *handleMessageOptions) {
		opts.bus = bus
	}
}

func WithHandleMessageGenerateOptions(genOpts ...ai.GenerateOption) HandleMessageOption {
	return func(opts *handleMessageOptions) {
		opts.baseOpts = genOpts
	}
}

func WithHandleMessageLoopOperator(loopOperator AgentLoopOperator) HandleMessageOption {
	return func(opts *handleMessageOptions) {
		opts.loopOperator = loopOperator
	}
}

func WithCustomAgentConfig(config AgentLoopConfig) HandleMessageOption {
	return func(opts *handleMessageOptions) {
		opts.defaultConfig = &config
	}
}

func NewHandleMessageFlow(
	g *genkit.Genkit,
	store *memory.Session,
	opts ...HandleMessageOption,
) *core.Flow[*HandleMessageInput, *HandleMessageOutput, struct{}] {
	options := &handleMessageOptions{
		loopOperator: &defaultAgentLoopOperator{g: g},
	}
	for _, opt := range opts {
		opt(options)
	}
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
			priorHistoryLen := len(history)

			isResume := len(input.ToolResponses) > 0 || len(input.ToolRestarts) > 0
			if !isResume {
				history = append(history, &input.Content)
			}

			resolvedConfig := mergeAgentConfig(options.defaultConfig, input.Config)

			loopInput := &AgentLoopInput{
				SessionID:     input.SessionID,
				Messages:      history,
				Config:        resolvedConfig,
				ToolResponses: input.ToolResponses,
				ToolRestarts:  input.ToolRestarts,
			}

			agentLoop := NewAgentLoop(
				g,
				WithEventBus(options.bus),
				WithCustomGenerateOptions(options.baseOpts...),
				WithCustomAgentLoopOperator(options.loopOperator),
			)

			loopOutput, err := agentLoop.Run(ctx, loopInput)
			if err != nil {
				return nil, fmt.Errorf("agent loop: %w", err)
			}

			newMessages := loopOutput.History[priorHistoryLen:]

			var sessionMessages []memory.SessionMessage
			for _, msg := range newMessages {
				origin := originForRole(msg.Role, input.Origin)
				sessionMessages = append(sessionMessages, memory.SessionMessage{
					Origin:  origin,
					Content: *msg,
				})
			}

			state := sess.State()
			state.Messages = append(state.Messages, sessionMessages...)
			if err := sess.UpdateState(ctx, state); err != nil {
				return nil, fmt.Errorf("update session state: %w", err)
			}

			return &HandleMessageOutput{
				SessionID:    sess.ID(),
				Response:     loopOutput.Response,
				History:      loopOutput.History,
				Turns:        loopOutput.Turns,
				FinishReason: loopOutput.FinishReason,
				Interrupts:   loopOutput.Interrupts,
			}, nil
		},
	)
}

func mergeAgentConfig(base, override *AgentLoopConfig) AgentLoopConfig {
	if base == nil && override == nil {
		return AgentLoopConfig{}
	}
	if base == nil {
		return *override
	}
	if override == nil {
		return *base
	}

	merged := *base

	if override.Model != "" {
		merged.Model = override.Model
	}
	if override.Tools != nil {
		merged.Tools = override.Tools
	}
	if override.SystemPrompt != nil {
		merged.SystemPrompt = override.SystemPrompt
	}
	if override.MaxTurns != 0 {
		merged.MaxTurns = override.MaxTurns
	}
	return merged
}

func originForRole(role ai.Role, inputOrigin memory.MessageOrigin) memory.MessageOrigin {
	switch role {
	case ai.RoleUser:
		return inputOrigin
	case ai.RoleModel:
		return memory.ModelMessage
	case ai.RoleTool:
		return memory.ToolMessage
	default:
		return memory.ModelMessage
	}
}
