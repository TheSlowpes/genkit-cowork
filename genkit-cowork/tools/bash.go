package tools

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/utils"
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
	Exec(ctx context.Context, cmd, cdw string, timeout *int, opts ...ExecOption) (string, int, error)
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

type BashToolOutput struct {
	Content  string
	ExitCode int
}

type defaultBashOperator struct{}

func (o *defaultBashOperator) Exec(ctx context.Context, cmd, cdw string, timeout *int, opts ...ExecOption) (string, int, error) {
	options := execOptions{}
	for _, opt := range opts {
		opt(&options)
	}
	var cmdCtx context.Context
	var cancel context.CancelFunc

	if timeout != nil {
		cmdCtx, cancel = context.WithTimeout(ctx, time.Duration(*timeout)*time.Second)
	} else {
		cmdCtx, cancel = context.WithCancel(ctx)
	}
	defer cancel()

	cmdExec := exec.CommandContext(cmdCtx, "bash", "-c", cmd)
	cmdExec.Dir = cdw

	var buf bytes.Buffer
	cmdExec.Stdout = &buf
	cmdExec.Stderr = &buf

	err := cmdExec.Run()

	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else if cmdCtx.Err() == context.DeadlineExceeded {
			return buf.String(), -1, fmt.Errorf("command time out after %d seconds", *timeout)
		} else if cmdCtx.Err() == context.Canceled {
			return buf.String(), -1, fmt.Errorf("command canceled")
		} else {
			return "", -1, err
		}
	}

	return buf.String(), exitCode, nil
}

func NewBashTool(g *genkit.Genkit, cwd string, opts ...BashToolOption) ai.Tool {
	options := bashToolOptions{}
	for _, opt := range opts {
		opt(&options)
	}
	return genkit.DefineTool(
		g,
		"bash",
		"Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last %d lines or %dKB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds",
		func(ctx *ai.ToolContext, input BashToolInput) (BashToolOutput, error) {
			if _, err := os.Stat(cwd); os.IsNotExist(err) {
				return BashToolOutput{}, fmt.Errorf("working directory does not exist: %s", cwd)
			}
			cmd := input.Command
			if options.commandPrefix != "" {
				cmd = options.commandPrefix + "\n" + cmd
			}

			execCwd := cwd
			var env []string

			spawnContext := BashSpawnContext{}

			if options.beforeSpawn != nil {
				spawnContext = resolveSpawnContext(cmd, execCwd, &options.beforeSpawn)
			}

			return BashToolOutput{}, nil
		},
	)
}

func resolveSpawnContext(cmd, cwd string, spawnHook *BashSpawnHook) BashSpawnContext {
	baseContext := BashSpawnContext{
		Cmd: cmd,
		Cwd: cwd,
		Env: utils.GetShellEnv(),
	}
	if spawnHook != nil {
		return (*spawnHook)(baseContext)
	}
	return baseContext
}
