// Copyright 2026 Kevin Lopes
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package flows

import (
	"context"
	"fmt"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

const defaultCoworkMaxTurns = 50

// AgentLoopConfig configures model/tool behavior for each loop run.
type AgentLoopConfig struct {
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

// AgentLoopOption configures NewAgentLoop behavior.
type AgentLoopOption func(*agentLoopOptions)

// AgentLoopOperator abstracts model generation and registry lookups used by
// the agent loop so callers can inject custom execution strategies.
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

// WithEventBus attaches an event bus used to emit agent loop lifecycle events.
func WithEventBus(bus *EventBus) AgentLoopOption {
	return func(opts *agentLoopOptions) {
		opts.bus = bus
	}
}

// WithCustomGenerateOptions sets base generate options for the underlying
// Genkit Generate call.
func WithCustomGenerateOptions(opts ...ai.GenerateOption) AgentLoopOption {
	return func(loopOpts *agentLoopOptions) {
		loopOpts.baseOpts = opts
	}
}

// WithCustomAgentLoopOperator sets a custom operator for generation and
// model/tool lookups.
func WithCustomAgentLoopOperator(operator AgentLoopOperator) AgentLoopOption {
	return func(loopOpts *agentLoopOptions) {
		loopOpts.operator = operator
	}
}

// AgentLoopInput is the input payload for a single agent loop run.
type AgentLoopInput struct {
	SessionID     string          `json:"sessionID"`
	Messages      []*ai.Message   `json:"messages"`
	Config        AgentLoopConfig `json:"config"`
	ToolResponses []*ai.Part      `json:"toolResponses,omitempty"`
	ToolRestarts  []*ai.Part      `json:"toolRestarts,omitempty"`
}

// AgentLoopOutput is the final result of an agent loop run.
type AgentLoopOutput struct {
	SessionID    string          `json:"sessionID"`
	Response     *ai.Message     `json:"response"`
	History      []*ai.Message   `json:"history"`
	Turns        int             `json:"turns"`
	TurnRecords  []AgentLoopTurn `json:"turnRecords,omitempty"`
	FinishReason ai.FinishReason `json:"finishReason"`
	Interrupts   []*ai.Part      `json:"interrupts,omitempty"`
}

// ToolExecutionRecord captures one tool execution during a loop turn.
type ToolExecutionRecord struct {
	ToolName    string        `json:"toolName"`
	ToolRef     string        `json:"toolRef,omitempty"`
	StartedAt   time.Time     `json:"startedAt"`
	EndedAt     time.Time     `json:"endedAt"`
	Duration    time.Duration `json:"duration"`
	Interrupted bool          `json:"interrupted,omitempty"`
	Error       string        `json:"error,omitempty"`
}

// AgentLoopTurn captures turn-level execution metadata for one loop iteration.
type AgentLoopTurn struct {
	TurnNumber             int                   `json:"turnNumber"`
	StartedAt              time.Time             `json:"startedAt"`
	EndedAt                time.Time             `json:"endedAt"`
	ResponseRole           ai.Role               `json:"responseRole"`
	FinishReason           string                `json:"finishReason"`
	InputTokens            int                   `json:"inputTokens,omitempty"`
	OutputTokens           int                   `json:"outputTokens,omitempty"`
	TotalTokens            int                   `json:"totalTokens,omitempty"`
	PersistedMessageCount  int                   `json:"persistedMessageCount"`
	ToolRequestCount       int                   `json:"toolRequestCount,omitempty"`
	ToolResponsePartCount  int                   `json:"toolResponsePartCount,omitempty"`
	Interrupted            bool                  `json:"interrupted,omitempty"`
	ToolExecutionSummaries []ToolExecutionRecord `json:"toolExecutionSummaries,omitempty"`
}

// NewAgentLoop creates the core model/tool turn loop used by higher-level
// flows like message handling and heartbeat.
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

	return genkit.NewFlow(
		"agentLoop",
		func(ctx context.Context, input *AgentLoopInput) (*AgentLoopOutput, error) {
			return agentLoopHandler(ctx, input, options)
		},
	)
}

func agentLoopHandler(ctx context.Context, input *AgentLoopInput, options *agentLoopOptions) (*AgentLoopOutput, error) {
	config := input.Config
	recorder := newAgentLoopRecorder(input.SessionID, config, options.bus)
	genOptions := make([]ai.GenerateOption, 0, len(options.baseOpts)+8)
	genOptions = append(genOptions, options.baseOpts...)
	genOptions = append(genOptions, ai.WithUse(recorder.middleware()))
	if config.MaxTurns > 0 {
		genOptions = append(genOptions, ai.WithMaxTurns(config.MaxTurns))
	} else if len(options.baseOpts) == 0 {
		genOptions = append(genOptions, ai.WithMaxTurns(defaultCoworkMaxTurns))
	}

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

	messages := make([]*ai.Message, len(input.Messages))
	copy(messages, input.Messages)
	genOptions = append(genOptions, ai.WithMessages(messages...))
	if len(input.ToolResponses) > 0 {
		genOptions = append(genOptions, ai.WithToolResponses(input.ToolResponses...))
	}
	if len(input.ToolRestarts) > 0 {
		genOptions = append(genOptions, ai.WithToolRestarts(input.ToolRestarts...))
	}

	emitIfBus(options.bus, ctx, AgentStart, AgentContext{
		SessionID: input.SessionID,
		ModelName: input.Config.Model,
		Tools:     input.Config.Tools,
		Config:    input.Config,
	})

	response, err := options.operator.Generate(ctx, genOptions...)
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
	if response.Request == nil {
		response.Request = &ai.ModelRequest{Messages: messages}
	}
	annotateGenerationUsage(response)
	history := response.History()
	turnRecords := buildAgentLoopTurnRecords(history, len(messages), response, recorder)

	emitIfBus(options.bus, ctx, AgentEnd, AgentContext{
		SessionID: input.SessionID,
		ModelName: input.Config.Model,
		Tools:     input.Config.Tools,
		Config:    input.Config,
	})

	return &AgentLoopOutput{
		SessionID:    input.SessionID,
		Response:     response.Message,
		History:      history,
		Turns:        len(turnRecords),
		TurnRecords:  turnRecords,
		FinishReason: response.FinishReason,
		Interrupts:   response.Interrupts(),
	}, nil
}

// --- Helpers ---

func annotateGenerationUsage(response *ai.ModelResponse) {
	if response == nil || response.Message == nil || response.Usage == nil {
		return
	}
	if response.Message.Metadata == nil {
		response.Message.Metadata = make(map[string]any)
	}
	response.Message.Metadata["generationUsage"] = map[string]any{
		"inputTokens":  response.Usage.InputTokens,
		"outputTokens": response.Usage.OutputTokens,
		"totalTokens":  response.Usage.TotalTokens,
	}
}

func buildAgentLoopTurnRecords(history []*ai.Message, priorHistoryLen int, response *ai.ModelResponse, recorder *agentLoopRecorder) []AgentLoopTurn {
	modelRecords := recorder.snapshotModelRecords()
	toolRecords := recorder.snapshotToolRecords()
	records := make([]AgentLoopTurn, 0)
	modelIndex := 0
	toolIndex := 0
	resumedMessageCount := 0
	for i := priorHistoryLen; i < len(history); i++ {
		msg := history[i]
		if msg == nil {
			continue
		}
		if msg.Role == ai.RoleTool && isResumedToolMessage(msg) {
			resumedMessageCount++
			continue
		}
		if msg.Role != ai.RoleModel {
			continue
		}

		turn := AgentLoopTurn{
			TurnNumber:            len(records) + 1,
			StartedAt:             time.Now(),
			EndedAt:               time.Now(),
			ResponseRole:          msg.Role,
			PersistedMessageCount: 1 + resumedMessageCount,
			ToolRequestCount:      countToolRequests(msg),
			Interrupted:           countInterrupts(msg) > 0,
		}
		resumedMessageCount = 0
		if modelIndex < len(modelRecords) {
			modelRecord := modelRecords[modelIndex]
			turn.StartedAt = modelRecord.StartedAt
			turn.EndedAt = modelRecord.EndedAt
			turn.InputTokens = modelRecord.InputTokens
			turn.OutputTokens = modelRecord.OutputTokens
			turn.TotalTokens = modelRecord.TotalTokens
		}
		modelIndex++

		if i+1 < len(history) && history[i+1] != nil && history[i+1].Role == ai.RoleTool {
			toolMsg := history[i+1]
			turn.PersistedMessageCount++
			turn.ToolResponsePartCount = len(toolMsg.Content)
			toolCount := turn.ToolRequestCount
			toolCount = min(toolCount, len(toolRecords)-toolIndex)
			if toolCount > 0 {
				turn.ToolExecutionSummaries = append(turn.ToolExecutionSummaries, toolRecords[toolIndex:toolIndex+toolCount]...)
				turn.EndedAt = maxToolEnd(turn.EndedAt, turn.ToolExecutionSummaries)
				toolIndex += toolCount
			}
			i++
		}
		if turn.Interrupted && toolIndex < len(toolRecords) {
			turn.ToolExecutionSummaries = append(turn.ToolExecutionSummaries, toolRecords[toolIndex:]...)
			turn.EndedAt = maxToolEnd(turn.EndedAt, turn.ToolExecutionSummaries)
			toolIndex = len(toolRecords)
		}

		if turn.Interrupted {
			turn.FinishReason = string(ai.FinishReasonInterrupted)
		} else if i >= len(history)-1 {
			turn.FinishReason = string(response.FinishReason)
			if turn.FinishReason == "" {
				turn.FinishReason = string(ai.FinishReasonStop)
			}
		} else {
			turn.FinishReason = "continue"
		}
		records = append(records, turn)
	}
	return records
}

func isResumedToolMessage(msg *ai.Message) bool {
	if msg.Metadata == nil {
		return false
	}
	_, ok := msg.Metadata["resumed"]
	return ok
}

func countToolRequests(msg *ai.Message) int {
	count := 0
	for _, part := range msg.Content {
		if part.IsToolRequest() {
			count++
		}
	}
	return count
}

func countInterrupts(msg *ai.Message) int {
	count := 0
	for _, part := range msg.Content {
		if part.IsInterrupt() {
			count++
		}
	}
	return count
}

func maxToolEnd(end time.Time, records []ToolExecutionRecord) time.Time {
	for _, record := range records {
		if record.EndedAt.After(end) {
			end = record.EndedAt
		}
	}
	return end
}

func emitIfBus[T any](bus *EventBus, ctx context.Context, eventType EventType, data T) (*Event[T], error) {
	if bus == nil {
		return nil, nil
	}
	return EmitEvent(bus, ctx, eventType, data)
}
