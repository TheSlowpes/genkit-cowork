package flows

import (
	"time"

	"github.com/firebase/genkit/go/ai"
)

type AgentContext struct {
	SessionID string      `json:"sessionID"`
	ModelName string      `json:"modelName"`
	Tools     []string    `json:"tools"`
	Config    AgentConfig `json:"config"`
	Error     error       `json:"error,omitempty"` // populated on agent-end if failed
}

type TurnContext struct {
	SessionID  string        `json:"sessionID"`
	TurnNumber int           `json:"turnNumber"`
	Messages   []*ai.Message `json:"messages"`        // conversation history at turn-start
	Response   *ai.Message   `json:"response"`        // populated on turn-end
	ToolCalls  []*ai.Message `json:"toolcalls"`       // populated on turn-end
	Error      error         `json:"error,omitempty"` // populated on turn-end if failed
}

type MessageContext struct {
	SessionID string      `json:"sessionID"`
	Role      ai.Role     `json:"role"`            // "user", "model", "tool"
	Message   *ai.Message `json:"message"`         // the full message (on start/end)
	Chunk     *ai.Part    `json:"chunk,omitempty"` // populated on message-update for streaming responses
	Index     int         `json:"index"`           // chunk index for updates
}

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
