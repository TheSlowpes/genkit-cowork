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

package tools

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/utils"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type execOptions struct {
	Env []string
}

// ExecOption configures command execution behaviour.
type ExecOption func(*execOptions)

type bashToolOptions struct {
	operator      BashOperator
	commandPrefix string
	beforeSpawn   BashSpawnHook
}

// BashToolOption configures a BashTool via functional options.
type BashToolOption func(*bashToolOptions)

// BashOperator abstracts command execution so implementations can be
// swapped for testing, sandboxing, or remote execution.
type BashOperator interface {
	Exec(ctx context.Context, cmd, cwd string, timeout *int, opts ...ExecOption) (string, int, error)
}

// BashSpawnContext carries the command, working directory, and environment
// that will be used to spawn a process.
type BashSpawnContext struct {
	Cmd string
	Cwd string
	Env []string
}

// BashSpawnHook allows callers to intercept and modify the spawn context
// before a command is executed.
type BashSpawnHook func(BashSpawnContext) BashSpawnContext

// WithCustomBashOperator injects a custom BashOperator implementation.
func WithCustomBashOperator(operator BashOperator) BashToolOption {
	return func(opts *bashToolOptions) {
		opts.operator = operator
	}
}

// WithCommandPrefix prepends a string to every command before execution.
// Useful for environment setup (e.g., "shopt -s expand_aliases").
func WithCommandPrefix(prefix string) BashToolOption {
	return func(opts *bashToolOptions) {
		opts.commandPrefix = prefix
	}
}

// WithBeforeSpawnHook registers a hook to modify the spawn context
// (command, cwd, env) before execution.
func WithBeforeSpawnHook(hook BashSpawnHook) BashToolOption {
	return func(opts *bashToolOptions) {
		opts.beforeSpawn = hook
	}
}

// WithCustomEnv overrides the process environment used for command execution.
func WithCustomEnv(env []string) ExecOption {
	return func(opts *execOptions) {
		opts.Env = env
	}
}

// BashToolInput is the schema for the bash tool's input parameters.
type BashToolInput struct {
	Command string `json:"command" jsonschema_description:"Bash command to execute"`
	Timeout *int   `json:"timeout,omitempty" jsonschema_description:"Timeout in seconds (optional)"`
}

// BashToolOutput is the structured result of a bash tool invocation.
type BashToolOutput struct {
	Content        string `json:"content"`
	ExitCode       int    `json:"exitCode"`
	FullOutputPath string `json:"fullOutputPath,omitempty"`
}

type defaultBashOperator struct{}

func (o *defaultBashOperator) Exec(ctx context.Context, cmd, cwd string, timeout *int, opts ...ExecOption) (string, int, error) {
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

	shell, shellArg := shellCommand()
	cmdExec := exec.CommandContext(cmdCtx, shell, shellArg, cmd)
	cmdExec.Dir = cwd

	if options.Env != nil {
		cmdExec.Env = options.Env
	}

	var buf bytes.Buffer
	cmdExec.Stdout = &buf
	cmdExec.Stderr = &buf

	err := cmdExec.Run()

	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else if cmdCtx.Err() == context.DeadlineExceeded {
			return buf.String(), -1, fmt.Errorf("command timed out after %d seconds", *timeout)
		} else if cmdCtx.Err() == context.Canceled {
			return buf.String(), -1, fmt.Errorf("command canceled")
		} else {
			return "", -1, err
		}
	}

	return buf.String(), exitCode, nil
}

// shellCommand returns the shell binary and flag to use for command execution,
// selected based on the current OS.
func shellCommand() (string, string) {
	if runtime.GOOS == "windows" {
		return "cmd", "/C"
	}
	return "bash", "-c"
}

// NewBashTool creates a Genkit tool that executes shell commands.
// The cwd parameter sets the working directory for command execution.
func NewBashTool(g *genkit.Genkit, cwd string, opts ...BashToolOption) ai.Tool {
	options := bashToolOptions{
		operator: &defaultBashOperator{},
	}
	for _, opt := range opts {
		opt(&options)
	}

	description := fmt.Sprintf(
		"Execute a bash command in the current working directory. Returns stdout and stderr. "+
			"Output is truncated to last %d lines or %s (whichever is hit first). "+
			"If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
		DEFAULT_MAX_LINES, FormatSize(DEFAULT_MAX_BYTES),
	)

	return genkit.DefineTool(
		g,
		"bash",
		description,
		func(ctx *ai.ToolContext, input BashToolInput) (BashToolOutput, error) {
			if _, err := os.Stat(cwd); os.IsNotExist(err) {
				return BashToolOutput{}, fmt.Errorf("working directory does not exist: %s", cwd)
			}

			cmd := input.Command
			if options.commandPrefix != "" {
				cmd = options.commandPrefix + "\n" + cmd
			}

			execCwd := cwd
			var execOpts []ExecOption

			if options.beforeSpawn != nil {
				spawnContext := resolveSpawnContext(cmd, execCwd, &options.beforeSpawn)
				cmd = spawnContext.Cmd
				execCwd = spawnContext.Cwd
				if len(spawnContext.Env) > 0 {
					execOpts = append(execOpts, func(o *execOptions) {
						o.Env = spawnContext.Env
					})
				}
			}

			output, exitCode, execErr := options.operator.Exec(ctx, cmd, execCwd, input.Timeout, execOpts...)

			// Process the output through truncation.
			result := processBashOutput(output, exitCode)
			if execErr != nil {
				return result, fmt.Errorf("command failed: %w", execErr)
			}
			return result, nil
		},
	)
}

// processBashOutput applies tail truncation to command output. When
// the output exceeds the truncation threshold, the full output is saved
// to a temp file and the returned content contains only the tail with
// an actionable notice pointing to the full output.
func processBashOutput(output string, exitCode int) BashToolOutput {
	if output == "" {
		content := "(no output)"
		if exitCode != 0 {
			content += fmt.Sprintf("\n\nCommand exited with code %d", exitCode)
		}
		return BashToolOutput{Content: content, ExitCode: exitCode}
	}

	truncation := TruncateTail(output, nil)
	outputText := truncation.Content
	var fullOutputPath string

	if truncation.Truncated {
		// Save full output to a temp file.
		tempPath, saveErr := saveTempOutput(output)
		if saveErr == nil {
			fullOutputPath = tempPath
		}

		// Build actionable truncation notice.
		startLine := truncation.TotalLines - truncation.OutputLines + 1
		endLine := truncation.TotalLines

		if truncation.LastLinePartial {
			lastLine := lastLineOf(output)
			lastLineSize := FormatSize(len(lastLine))
			outputText += fmt.Sprintf(
				"\n\n[Showing last %s of line %d (line is %s).",
				FormatSize(truncation.OutputBytes), endLine, lastLineSize,
			)
		} else if truncation.TruncatedBy == "lines" {
			outputText += fmt.Sprintf(
				"\n\n[Showing lines %d-%d of %d.",
				startLine, endLine, truncation.TotalLines,
			)
		} else {
			outputText += fmt.Sprintf(
				"\n\n[Showing lines %d-%d of %d (%s limit).",
				startLine, endLine, truncation.TotalLines, FormatSize(DEFAULT_MAX_BYTES),
			)
		}

		if fullOutputPath != "" {
			outputText += fmt.Sprintf(" Full output: %s]", fullOutputPath)
		} else {
			outputText += "]"
		}
	}

	if exitCode != 0 {
		outputText += fmt.Sprintf("\n\nCommand exited with code %d", exitCode)
	}

	return BashToolOutput{
		Content:        outputText,
		ExitCode:       exitCode,
		FullOutputPath: fullOutputPath,
	}
}

// saveTempOutput writes content to a temporary file and returns its path.
// Uses os.CreateTemp which works across platforms (Linux, macOS, Windows).
func saveTempOutput(content string) (string, error) {
	id := make([]byte, 8)
	if _, err := rand.Read(id); err != nil {
		return "", err
	}
	name := fmt.Sprintf("cowork-bash-%s-*.log", hex.EncodeToString(id))

	f, err := os.CreateTemp("", name)
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer f.Close()

	if _, err := f.WriteString(content); err != nil {
		os.Remove(f.Name())
		return "", fmt.Errorf("failed to write temp file: %w", err)
	}

	// Return a clean, absolute path.
	absPath, err := filepath.Abs(f.Name())
	if err != nil {
		return f.Name(), nil
	}
	return absPath, nil
}

// lastLineOf returns the last line of a string without allocating a full
// split. Returns the whole string if there are no newlines.
func lastLineOf(s string) string {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '\n' {
			return s[i+1:]
		}
	}
	return s
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
