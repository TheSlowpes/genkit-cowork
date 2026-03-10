package flows

import (
	"context"
	"fmt"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

type AgentConfig struct {
	Model        string
	Tools        []string
	SystemPrompt ai.PromptFn
	MaxTurns     int
}

type agentLoopOptions struct {
	bus      *EventBus
	baseOpts []ai.GenerateOption
	operator AgentLoopOperator
}

type AgentLoopOption func(*agentLoopOptions)

type AgentLoopOperator interface {
	Generate(ctx context.Context, opts ...ai.GenerateOption) (*ai.ModelResponse, error)
	LookupModel(name string) (ai.Model, bool)
	LookupTool(name string) (ai.Tool, bool)
}

type defaultAgentLoopOperator struct {
	g *genkit.Genkit
}

var _ AgentLoopOperator = (*defaultAgentLoopOperator)(nil)

func (o *defaultAgentLoopOperator) Generate(ctx context.Context, opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
	return genkit.Generate(ctx, o.g, opts...)
}

func (o *defaultAgentLoopOperator) LookupModel(name string) (ai.Model, bool) {
	model := genkit.LookupModel(o.g, name)
	if model == nil {
		return nil, false
	}
	return model, true
}

func (o *defaultAgentLoopOperator) LookupTool(name string) (ai.Tool, bool) {
	tool := genkit.LookupTool(o.g, name)
	if tool == nil {
		return nil, false
	}
	return tool, true
}

func WithEventBus(bus *EventBus) AgentLoopOption {
	return func(opts *agentLoopOptions) {
		opts.bus = bus
	}
}

func WithCustomGenerateOptions(opts ...ai.GenerateOption) AgentLoopOption {
	return func(loopOpts *agentLoopOptions) {
		loopOpts.baseOpts = opts
	}
}

func WithCustomAgentLoopOperator(operator AgentLoopOperator) AgentLoopOption {
	return func(loopOpts *agentLoopOptions) {
		loopOpts.operator = operator
	}
}

type AgentLoopInput struct {
	SessionID string        `json:"sessionID"`
	Messages  []*ai.Message `json:"messages"`
	Config    AgentConfig   `json:"config"`
}

type AgentLoopOutput struct {
	SessionID string        `json:"sessionID"`
	Response  *ai.Message   `json:"response"`
	History   []*ai.Message `json:"history"`
	Turns     int           `json:"turns"`
}

func NewAgentLoop(
	g *genkit.Genkit,
	opts ...AgentLoopOption,
) *core.Flow[*AgentLoopInput, *AgentLoopOutput, struct{}] {
	options := &agentLoopOptions{
		operator: &defaultAgentLoopOperator{g: g},
	}
	for _, opt := range opts {
		opt(options)
	}

	return genkit.DefineFlow(
		g,
		"agentLoop",
		func(ctx context.Context, input *AgentLoopInput) (*AgentLoopOutput, error) {
			config := input.Config
			genOptions := make([]ai.GenerateOption, len(options.baseOpts))
			genOptions = append(genOptions, ai.WithReturnToolRequests(true))
			toolsRef := make([]ai.ToolRef, len(config.Tools))
			for _, toolName := range config.Tools {
				if tool, ok := options.operator.LookupTool(toolName); ok {
					toolsRef = append(toolsRef, tool)
				}
			}
			genOptions = append(genOptions, ai.WithTools(toolsRef...))
			if model, ok := options.operator.LookupModel(config.Model); ok {
				genOptions = append(genOptions, ai.WithModel(model))
			}
			if config.SystemPrompt != nil {
				genOptions = append(genOptions, ai.WithSystemFn(config.SystemPrompt))
			}
			if config.MaxTurns > 0 {
				genOptions = append(genOptions, ai.WithMaxTurns(config.MaxTurns))
			}

			history := input.Messages

			turnNumber := 0

			emitIfBus(options.bus, ctx, AgentStart, AgentContext{
				SessionID: input.SessionID,
				ModelName: input.Config.Model,
				Tools:     input.Config.Tools,
				Config:    input.Config,
			})

			for {
				turnNumber++

				if config.MaxTurns > 0 && turnNumber > config.MaxTurns {
					return nil, fmt.Errorf("agent loop exceeded max turns (%d)", config.MaxTurns)
				}

				emitIfBus(options.bus, ctx, TurnStart, TurnContext{
					SessionID:  input.SessionID,
					TurnNumber: turnNumber,
					Messages:   history,
				})

				callOpts := append(genOptions, ai.WithMessages(history...))
				response, err := options.operator.Generate(ctx, callOpts...)
				if err != nil {
					emitIfBus(options.bus, ctx, AgentEnd, AgentContext{
						SessionID: input.SessionID,
						ModelName: input.Config.Model,
						Tools:     input.Config.Tools,
						Config:    input.Config,
						Error:     err,
					})
					return nil, fmt.Errorf("generate response: %w", err)
				}

				emitIfBus(options.bus, ctx, MessageStart, MessageContext{
					SessionID: input.SessionID,
					Role:      response.Message.Role,
					Message:   response.Message,
				})
				emitIfBus(options.bus, ctx, MessageEnd, MessageContext{
					SessionID: input.SessionID,
					Role:      response.Message.Role,
					Message:   response.Message,
				})

				history = append(history, response.Message)

				toolRequests := response.ToolRequests()
				if len(toolRequests) > 0 && response.FinishReason == ai.FinishReasonStop {
					emitIfBus(options.bus, ctx, TurnEnd, TurnContext{
						SessionID:  input.SessionID,
						TurnNumber: turnNumber,
						Messages:   history,
						Response:   response.Message,
					})
					break
				}

				var toolResponseParts []*ai.Part
				var tooCallMessages []*ai.Message
				for _, toolReq := range toolRequests {
					startEvent, _ := emitIfBus(options.bus, ctx, ToolExecutionStart, ToolExecutionContext{
						SessionID: input.SessionID,
						ToolName:  toolReq.Name,
						Input:     toolReq.Input,
					})

					toolInput := toolReq.Input
					if startEvent != nil {
						toolInput = startEvent.Data.Input
					}

					start := time.Now()
					tool, ok := options.operator.LookupTool(toolReq.Name)
					if !ok {
						return nil, fmt.Errorf("tool not found: %s", toolReq.Name)
					}
					multipartOutput, toolErr := tool.RunRawMultipart(ctx, toolInput)
					duration := time.Since(start)

					emitIfBus(options.bus, ctx, ToolExecutionEnd, ToolExecutionContext{
						SessionID: input.SessionID,
						ToolName:  toolReq.Name,
						Input:     toolInput,
						Output:    multipartOutput.Output,
						Duration:  duration,
						Error:     toolErr,
					})

					toolResponseParts = append(toolResponseParts, ai.NewToolResponsePart(&ai.ToolResponse{
						Name:    toolReq.Name,
						Ref:     toolReq.Ref,
						Content: multipartOutput.Content,
						Output:  multipartOutput.Output,
					}))
				}

				toolMsg := &ai.Message{
					Role:    ai.RoleTool,
					Content: toolResponseParts,
				}
				history = append(history, toolMsg)
				tooCallMessages = append(tooCallMessages, toolMsg)

				emitIfBus(options.bus, ctx, TurnEnd, TurnContext{
					SessionID:  input.SessionID,
					TurnNumber: turnNumber,
					Messages:   history,
					Response:   response.Message,
					ToolCalls:  tooCallMessages,
				})
			}

			emitIfBus(options.bus, ctx, AgentEnd, AgentContext{
				SessionID: input.SessionID,
				ModelName: input.Config.Model,
				Tools:     input.Config.Tools,
				Config:    input.Config,
			})

			return &AgentLoopOutput{
				SessionID: input.SessionID,
				Response:  history[len(history)-1],
				History:   history,
				Turns:     turnNumber,
			}, nil
		},
	)
}

func emitIfBus[T any](bus *EventBus, ctx context.Context, eventType EventType, data T) (*Event[T], error) {
	if bus == nil {
		return nil, nil
	}
	return EmitEvent(bus, ctx, eventType, data)
}
