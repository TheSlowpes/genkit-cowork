package flows

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

type AgentConfig struct {
	Model        string      `json:"model,omitempty"`
	Tools        []string    `json:"tools,omitempty"`
	SystemPrompt ai.PromptFn `json:"-"`
	MaxTurns     int         `json:"maxTurns,omitempty"`
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
	SessionID     string        `json:"sessionID"`
	Messages      []*ai.Message `json:"messages"`
	Config        AgentConfig   `json:"config"`
	ToolResponses []*ai.Part    `json:"toolResponses,omitempty"`
	ToolRestarts  []*ai.Part    `json:"toolRestarts,omitempty"`
}

type AgentLoopOutput struct {
	SessionID    string          `json:"sessionID"`
	Response     *ai.Message     `json:"response"`
	History      []*ai.Message   `json:"history"`
	Turns        int             `json:"turns"`
	FinishReason ai.FinishReason `json:"finishReason"`
	Interrupts   []*ai.Part      `json:"interrupts,omitempty"`
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
			return agentLoopHandler(ctx, input, options)
		},
	)
}

func agentLoopHandler(ctx context.Context, input *AgentLoopInput, options *agentLoopOptions) (*AgentLoopOutput, error) {
	config := input.Config
	genOptions := make([]ai.GenerateOption, 0, len(options.baseOpts)+4)
	genOptions = append(genOptions, options.baseOpts...)
	genOptions = append(genOptions, ai.WithReturnToolRequests(true))
	var toolsRef []ai.ToolRef
	for _, toolName := range config.Tools {
		if tool, ok := options.operator.LookupTool(toolName); ok {
			toolsRef = append(toolsRef, tool)
		}
	}
	if len(toolsRef) > 0 {
		genOptions = append(genOptions, ai.WithTools(toolsRef...))
	}
	if model, ok := options.operator.LookupModel(config.Model); ok {
		genOptions = append(genOptions, ai.WithModel(model))
	}
	if config.SystemPrompt != nil {
		genOptions = append(genOptions, ai.WithSystemFn(config.SystemPrompt))
	}

	history := make([]*ai.Message, len(input.Messages))
	copy(history, input.Messages)

	turnNumber := 0

	emitIfBus(options.bus, ctx, AgentStart, AgentContext{
		SessionID: input.SessionID,
		ModelName: input.Config.Model,
		Tools:     input.Config.Tools,
		Config:    input.Config,
	})

	var resumeOpts []ai.GenerateOption
	if len(input.ToolResponses) > 0 {
		resumeOpts = append(resumeOpts, ai.WithToolResponses(input.ToolResponses...))
	}
	if len(input.ToolRestarts) > 0 {
		resumeOpts = append(resumeOpts, ai.WithToolRestarts(input.ToolRestarts...))
	}

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
		isResumeTurn := turnNumber == 1 && len(resumeOpts) > 0
		if isResumeTurn {
			callOpts = append(callOpts, resumeOpts...)
		}
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

		// After a resume turn, Genkit's handleResumeOption internally creates a
		// tool response message and prepends it to the messages sent to the model.
		// However, response.Request is not populated by Genkit's Generate in the
		// ReturnToolRequests path, so we reconstruct the tool response message
		// from the resume inputs and insert it into history ourselves.
		if isResumeTurn {
			var resumeParts []*ai.Part
			resumeParts = append(resumeParts, input.ToolResponses...)
			resumeParts = append(resumeParts, input.ToolRestarts...)
			if len(resumeParts) > 0 {
				toolMsg := &ai.Message{
					Role:     ai.RoleTool,
					Content:  resumeParts,
					Metadata: map[string]any{"resumed": true},
				}
				history = append(history, toolMsg)
			}
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
		if len(toolRequests) == 0 {
			emitIfBus(options.bus, ctx, TurnEnd, TurnContext{
				SessionID:  input.SessionID,
				TurnNumber: turnNumber,
				Messages:   history,
				Response:   response.Message,
			})
			break
		}

		// Build a map from (name, ref) to index in response.Message.Content
		// so we can annotate the correct parts on interrupt.
		type toolExecResult struct {
			output  any
			content []*ai.Part
		}
		completedTools := make(map[int]toolExecResult) // key: index in response.Message.Content

		var toolResponseParts []*ai.Part
		var toolCallMessages []*ai.Message
		interrupted := false

		// Build an ordered list of (toolRequest, contentIndex) pairs so we can
		// correlate tool requests back to their position in the model message.
		type toolReqEntry struct {
			request      *ai.ToolRequest
			contentIndex int
		}
		var toolReqEntries []toolReqEntry
		for i, part := range response.Message.Content {
			if part.IsToolRequest() {
				toolReqEntries = append(toolReqEntries, toolReqEntry{
					request:      part.ToolRequest,
					contentIndex: i,
				})
			}
		}

		for entryIdx, entry := range toolReqEntries {
			toolReq := entry.request

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

			isInterrupt, interruptMeta := ai.IsToolInterruptError(toolErr)
			if isInterrupt {
				// Emit ToolExecutionUpdate with interrupt info
				emitIfBus(options.bus, ctx, ToolExecutionUpdate, ToolExecutionContext{
					SessionID:         input.SessionID,
					ToolName:          toolReq.Name,
					Input:             toolInput,
					Duration:          duration,
					Interrupted:       true,
					InterruptMetadata: interruptMeta,
				})

				// Emit ToolExecutionEnd with the error
				emitIfBus(options.bus, ctx, ToolExecutionEnd, ToolExecutionContext{
					SessionID:   input.SessionID,
					ToolName:    toolReq.Name,
					Input:       toolInput,
					Duration:    duration,
					Error:       toolErr,
					Interrupted: true,
				})

				// Build annotated model message with interrupt/pendingOutput metadata
				annotatedMsg := cloneMessage(response.Message)

				// Annotate completed tools with pendingOutput
				for completedIdx, result := range completedTools {
					setPartMetadata(annotatedMsg.Content[completedIdx], "pendingOutput", result.output)
				}

				// Annotate the interrupted tool
				if interruptMeta != nil {
					setPartMetadata(annotatedMsg.Content[entry.contentIndex], "interrupt", interruptMeta)
				} else {
					setPartMetadata(annotatedMsg.Content[entry.contentIndex], "interrupt", true)
				}

				// Annotate remaining unexecuted tools
				for _, remaining := range toolReqEntries[entryIdx+1:] {
					setPartMetadata(annotatedMsg.Content[remaining.contentIndex], "interrupt", true)
				}

				// Replace the model message in history with the annotated version
				history[len(history)-1] = annotatedMsg

				// Collect interrupted parts
				var interruptParts []*ai.Part
				for _, part := range annotatedMsg.Content {
					if part.IsInterrupt() {
						interruptParts = append(interruptParts, part)
					}
				}

				// Emit TurnEnd and AgentEnd
				emitIfBus(options.bus, ctx, TurnEnd, TurnContext{
					SessionID:  input.SessionID,
					TurnNumber: turnNumber,
					Messages:   history,
					Response:   annotatedMsg,
				})

				emitIfBus(options.bus, ctx, AgentEnd, AgentContext{
					SessionID: input.SessionID,
					ModelName: input.Config.Model,
					Tools:     input.Config.Tools,
					Config:    input.Config,
				})

				return &AgentLoopOutput{
					SessionID:    input.SessionID,
					Response:     annotatedMsg,
					History:      history,
					Turns:        turnNumber,
					FinishReason: ai.FinishReasonInterrupted,
					Interrupts:   interruptParts,
				}, nil
			}

			var toolOutput any
			var toolContent []*ai.Part
			if multipartOutput != nil {
				toolOutput = multipartOutput.Output
				toolContent = multipartOutput.Content
			}

			// Track completed tool for potential pendingOutput annotation
			completedTools[entry.contentIndex] = toolExecResult{
				output:  toolOutput,
				content: toolContent,
			}

			emitIfBus(options.bus, ctx, ToolExecutionEnd, ToolExecutionContext{
				SessionID: input.SessionID,
				ToolName:  toolReq.Name,
				Input:     toolInput,
				Output:    toolOutput,
				Duration:  duration,
				Error:     toolErr,
			})

			toolResponseParts = append(toolResponseParts, ai.NewToolResponsePart(&ai.ToolResponse{
				Name:    toolReq.Name,
				Ref:     toolReq.Ref,
				Content: toolContent,
				Output:  toolOutput,
			}))
		}

		if interrupted {
			continue
		}

		toolMsg := &ai.Message{
			Role:    ai.RoleTool,
			Content: toolResponseParts,
		}
		history = append(history, toolMsg)
		toolCallMessages = append(toolCallMessages, toolMsg)

		emitIfBus(options.bus, ctx, TurnEnd, TurnContext{
			SessionID:  input.SessionID,
			TurnNumber: turnNumber,
			Messages:   history,
			Response:   response.Message,
			ToolCalls:  toolCallMessages,
		})
	}

	emitIfBus(options.bus, ctx, AgentEnd, AgentContext{
		SessionID: input.SessionID,
		ModelName: input.Config.Model,
		Tools:     input.Config.Tools,
		Config:    input.Config,
	})

	return &AgentLoopOutput{
		SessionID:    input.SessionID,
		Response:     history[len(history)-1],
		History:      history,
		Turns:        turnNumber,
		FinishReason: ai.FinishReasonStop,
	}, nil
}

// --- Helpers ---

func cloneMessage(msg *ai.Message) *ai.Message {
	if msg == nil {
		return nil
	}
	data, err := json.Marshal(msg)
	if err != nil {
		panic(fmt.Sprintf("cloneMessage: marshal failed: %v", err))
	}
	var cloned ai.Message
	if err := json.Unmarshal(data, &cloned); err != nil {
		panic(fmt.Sprintf("cloneMessage: unmarshal failed: %v", err))
	}
	return &cloned
}

func setPartMetadata(part *ai.Part, key string, value any) {
	if part.Metadata == nil {
		part.Metadata = make(map[string]any)
	}
	part.Metadata[key] = value
}

func emitIfBus[T any](bus *EventBus, ctx context.Context, eventType EventType, data T) (*Event[T], error) {
	if bus == nil {
		return nil, nil
	}
	return EmitEvent(bus, ctx, eventType, data)
}
