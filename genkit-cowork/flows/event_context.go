// Copyright 2025 Kevin Lopes
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
	"time"

	"github.com/firebase/genkit/go/ai"
)

// AgentContext carries agent-level lifecycle event data.
type AgentContext struct {
	SessionID string          `json:"sessionID"`
	ModelName string          `json:"modelName"`
	Tools     []string        `json:"tools"`
	Config    AgentLoopConfig `json:"config"`
	Error     error           `json:"error,omitempty"` // populated on agent-end if failed
}

// TurnContext carries turn-level event data.
type TurnContext struct {
	SessionID  string        `json:"sessionID"`
	TurnNumber int           `json:"turnNumber"`
	Messages   []*ai.Message `json:"messages"`        // conversation history at turn-start
	Response   *ai.Message   `json:"response"`        // populated on turn-end
	ToolCalls  []*ai.Message `json:"toolcalls"`       // populated on turn-end
	Error      error         `json:"error,omitempty"` // populated on turn-end if failed
}

// MessageContext carries message-level event data.
type MessageContext struct {
	SessionID string      `json:"sessionID"`
	Role      ai.Role     `json:"role"`            // "user", "model", "tool"
	Message   *ai.Message `json:"message"`         // the full message (on start/end)
	Chunk     *ai.Part    `json:"chunk,omitempty"` // populated on message-update for streaming responses
	Index     int         `json:"index"`           // chunk index for updates
}

// ToolExecutionContext carries tool execution event data.
type ToolExecutionContext struct {
	SessionID         string         `json:"sessionID"`
	ToolName          string         `json:"toolName"`
	Input             any            `json:"input"`
	Output            any            `json:"output,omitempty"`
	InterruptMetadata map[string]any `json:"interruptMetadata,omitempty"` // populated on tool-execution-update if execution was interrupted, contains metadata about the interruption
	Interrupted       bool           `json:"interrupted,omitempty"`       // populated on tool-execution-update if execution was interrupted
	Duration          time.Duration  `json:"duration,omitempty"`          // populated on tool-execution-end
	Error             error          `json:"error,omitempty"`             // populated on tool-execution-end if failed
}
