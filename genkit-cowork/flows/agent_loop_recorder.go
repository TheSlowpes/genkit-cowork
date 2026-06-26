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
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
)

type agentLoopRecorder struct {
	sessionID string
	config    AgentLoopConfig
	bus       *EventBus

	mu           sync.Mutex
	modelRecords []agentLoopModelRecord
	toolRecords  []ToolExecutionRecord
}

type agentLoopModelRecord struct {
	StartedAt    time.Time
	EndedAt      time.Time
	Duration     time.Duration
	InputTokens  int
	OutputTokens int
	TotalTokens  int
	Error        string
}

func newAgentLoopRecorder(sessionID string, config AgentLoopConfig, bus *EventBus) *agentLoopRecorder {
	return &agentLoopRecorder{
		sessionID: sessionID,
		config:    config,
		bus:       bus,
	}
}

func (r *agentLoopRecorder) middleware() ai.Middleware {
	return ai.MiddlewareFunc(func(context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{
			WrapGenerate: r.wrapGenerate,
			WrapModel:    r.wrapModel,
			WrapTool:     r.wrapTool,
		}, nil
	})
}

func (r *agentLoopRecorder) wrapGenerate(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
	turnNumber := 0
	var messages []*ai.Message
	if params != nil {
		turnNumber = params.Iteration + 1
		if params.Request != nil {
			messages = params.Request.Messages
		}
	}

	emitIfBus(r.bus, ctx, TurnStart, TurnContext{
		SessionID:  r.sessionID,
		TurnNumber: turnNumber,
		Messages:   messages,
	})

	response, err := next(ctx, params)
	var responseMessage *ai.Message
	var history []*ai.Message
	if response != nil {
		responseMessage = response.Message
		history = response.History()
	}
	emitIfBus(r.bus, ctx, TurnEnd, TurnContext{
		SessionID:  r.sessionID,
		TurnNumber: turnNumber,
		Messages:   history,
		Response:   responseMessage,
		Error:      err,
	})

	return response, err
}

func (r *agentLoopRecorder) wrapModel(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
	startedAt := time.Now()
	response, err := next(ctx, params)
	duration := time.Since(startedAt)
	if response != nil && response.Request == nil && params != nil {
		response.Request = params.Request
	}
	if response != nil && response.Message != nil {
		emitIfBus(r.bus, ctx, MessageStart, MessageContext{
			SessionID: r.sessionID,
			Role:      response.Message.Role,
			Message:   response.Message,
		})
		emitIfBus(r.bus, ctx, MessageEnd, MessageContext{
			SessionID: r.sessionID,
			Role:      response.Message.Role,
			Message:   response.Message,
		})
	}

	record := agentLoopModelRecord{
		StartedAt: startedAt,
		EndedAt:   startedAt.Add(duration),
		Duration:  duration,
	}
	if response != nil && response.Usage != nil {
		record.InputTokens = response.Usage.InputTokens
		record.OutputTokens = response.Usage.OutputTokens
		record.TotalTokens = response.Usage.TotalTokens
	}
	if err != nil {
		record.Error = err.Error()
	}

	r.mu.Lock()
	r.modelRecords = append(r.modelRecords, record)
	r.mu.Unlock()

	return response, err
}

func (r *agentLoopRecorder) wrapTool(ctx context.Context, params *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
	toolName := ""
	var toolInput any
	var toolRef string
	if params != nil && params.Request != nil {
		toolName = params.Request.Name
		toolInput = params.Request.Input
		toolRef = params.Request.Ref
	}
	if toolName == "" && params != nil && params.Tool != nil {
		toolName = params.Tool.Name()
	}

	startEvent, _ := emitIfBus(r.bus, ctx, ToolExecutionStart, ToolExecutionContext{
		SessionID: r.sessionID,
		ToolName:  toolName,
		Input:     toolInput,
	})
	if startEvent != nil && params != nil && params.Request != nil {
		params.Request.Input = startEvent.Data.Input
		toolInput = startEvent.Data.Input
	}

	startedAt := time.Now()
	response, err := next(ctx, params)
	duration := time.Since(startedAt)

	var output any
	if response != nil {
		output = response.Output
	}

	interrupted, interruptMetadata := ai.IsToolInterruptError(err)
	if interrupted {
		emitIfBus(r.bus, ctx, ToolExecutionUpdate, ToolExecutionContext{
			SessionID:         r.sessionID,
			ToolName:          toolName,
			Input:             toolInput,
			Duration:          duration,
			Interrupted:       true,
			InterruptMetadata: interruptMetadata,
		})
	}

	emitIfBus(r.bus, ctx, ToolExecutionEnd, ToolExecutionContext{
		SessionID:   r.sessionID,
		ToolName:    toolName,
		Input:       toolInput,
		Output:      output,
		Duration:    duration,
		Error:       err,
		Interrupted: interrupted,
	})

	record := ToolExecutionRecord{
		ToolName:    toolName,
		ToolRef:     toolRef,
		StartedAt:   startedAt,
		EndedAt:     startedAt.Add(duration),
		Duration:    duration,
		Interrupted: interrupted,
	}
	if err != nil {
		record.Error = err.Error()
	}

	r.mu.Lock()
	r.toolRecords = append(r.toolRecords, record)
	r.mu.Unlock()

	return response, err
}

func (r *agentLoopRecorder) snapshotToolRecords() []ToolExecutionRecord {
	r.mu.Lock()
	defer r.mu.Unlock()

	records := make([]ToolExecutionRecord, len(r.toolRecords))
	copy(records, r.toolRecords)
	return records
}

func (r *agentLoopRecorder) snapshotModelRecords() []agentLoopModelRecord {
	r.mu.Lock()
	defer r.mu.Unlock()

	records := make([]agentLoopModelRecord, len(r.modelRecords))
	copy(records, r.modelRecords)
	return records
}
