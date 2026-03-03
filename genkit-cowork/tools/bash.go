package tools

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type execOptions struct {
	OnData func(data []byte)
	Env    []string
}

type ExecOption func(*execOptions)

type bashToolOptions struct {
	operator      BashOperator
	commandPrefix string
	beforeSpawn   BashSpawnHook
}

type BashToolOption func(*bashToolOptions)

type BashOperator interface {
	Exec(ctx context.Context, cmd string, opts ...ExecOption) (int, error)
}

type BashSpawnContext struct {
	Cmd string
	Cwd string
	Env []string
}

type BashSpawnHook func(BashSpawnContext) BashSpawnContext

func WithCustomOperator(operator BashOperator) BashToolOption {
	return func(opts *bashToolOptions) {
		opts.operator = operator
	}
}

func WithCommandPrefix(prefix string) BashToolOption {
	return func(opts *bashToolOptions) {
		opts.commandPrefix = prefix
	}
}

func WithBeforeSpawnHook(hook BashSpawnHook) BashToolOption {
	return func(opts *bashToolOptions) {
		opts.beforeSpawn = hook
	}
}

type BashToolInput struct {
	Command string `json:"command" jsonschema_description:"Bash command to execute"`
	Timeout *int   `json:"timeout,omitempty" jsonschema_description:"Timeout in seconds(optional)"`
}

func NewBashTool(g *genkit.Genkit, opts ...BashToolOption) ai.Tool {
	options := bashToolOptions{}
	for _, opt := range opts {
		opt(&options)
	}
	return genkit.DefineTool(
		g,
		"bash",
		"Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last %d lines or %dKB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds",
		func(ctx *ai.ToolContext, input BashToolInput) (string, error) {

		}
	)
}
