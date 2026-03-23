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

package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type writeToolOptions struct {
	operator WriteOperator
}

// WriteToolOption configures Write tool behavior.
type WriteToolOption func(*writeToolOptions)

// WriteOperator abstracts file writing operations.
type WriteOperator interface {
	// WriteFile writes content to a target file path.
	WriteFile(ctx context.Context, absolutePath, content string) error
	// MkdirAll ensures the target directory exists.
	MkdirAll(ctx context.Context, absolutePath string) error
}

// WithCustomWriteOperator injects a custom write operator.
func WithCustomWriteOperator(operator WriteOperator) WriteToolOption {
	return func(opts *writeToolOptions) {
		opts.operator = operator
	}
}

type defaultWriteOperator struct{}

func (o *defaultWriteOperator) WriteFile(ctx context.Context, absolutePath, content string) error {
	permissions := os.FileMode(0644)
	return os.WriteFile(absolutePath, []byte(content), permissions)
}

func (o *defaultWriteOperator) MkdirAll(ctx context.Context, absolutePath string) error {
	return os.MkdirAll(absolutePath, os.ModePerm)
}

// WriteToolInput is the input payload for write tool invocations.
type WriteToolInput struct {
	Path    string `json:"path" jsonschema_description:"Path to the file to write (relative or absolute)"`
	Content string `json:"content" jsonschema_description:"Content to write to the file"`
}

// NewWriteTool creates a Genkit tool that writes or overwrites files.
func NewWriteTool(g *genkit.Genkit, cwd string, opts ...WriteToolOption) ai.Tool {
	options := writeToolOptions{
		operator: &defaultWriteOperator{},
	}
	for _, opt := range opts {
		opt(&options)
	}

	description := "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories."
	return genkit.DefineTool(
		g,
		"write",
		description,
		func(ctx *ai.ToolContext, input WriteToolInput) (string, error) {
			return performWrite(ctx, input, cwd, &options)
		},
	)
}

func performWrite(ctx context.Context, input WriteToolInput, cwd string, options *writeToolOptions) (string, error) {
	ops := options.operator

	// Resolve path relative to cwd
	absolutePath := resolveToCwd(input.Path, cwd)

	dir := filepath.Dir(absolutePath)

	if err := ctx.Err(); err != nil {
		return "", fmt.Errorf("operation cancelled")
	}

	if err := ops.MkdirAll(ctx, dir); err != nil {
		return "", fmt.Errorf("failed to create directories: %w", err)
	}

	if err := ctx.Err(); err != nil {
		return "", fmt.Errorf("operation cancelled")
	}

	if err := ops.WriteFile(ctx, absolutePath, input.Content); err != nil {
		return "", fmt.Errorf("failed to write file: %w", err)
	}

	return fmt.Sprintf("Successfully wrote %d bytes to %s", len(input.Content), absolutePath), nil
}
