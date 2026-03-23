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
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type editToolOptions struct {
	operator EditOperator
}

// EditToolOption configures Edit tool behavior.
type EditToolOption func(*editToolOptions)

// EditOperator abstracts file editing operations so implementations can be
// swapped for testing, sandboxing, or remote execution (e.g., SSH).
type EditOperator interface {
	ReadFile(ctx context.Context, absolutePath string) ([]byte, error)
	WriteFile(ctx context.Context, absolutePath, content string) error
	Access(ctx context.Context, absolutePath string) error
}

// WithCustomEditOperator injects a custom file editing operator.
func WithCustomEditOperator(operator EditOperator) EditToolOption {
	return func(opts *editToolOptions) {
		opts.operator = operator
	}
}

type defaultEditOperator struct{}

func (o *defaultEditOperator) ReadFile(ctx context.Context, absolutePath string) ([]byte, error) {
	return os.ReadFile(absolutePath)
}

func (o *defaultEditOperator) WriteFile(ctx context.Context, absolutePath, content string) error {
	fileInfo, _ := os.Stat(absolutePath)
	permissions := fileInfo.Mode()
	return os.WriteFile(absolutePath, []byte(content), permissions)
}

func (o *defaultEditOperator) Access(ctx context.Context, absolutePath string) error {
	_, err := os.Stat(absolutePath)
	return err
}

// EditToolInput is the schema for the input to the EditTool. It specifies
// the file to edit and the text to find and replace.
type EditToolInput struct {
	Path    string `json:"path" jsonschema_description:"Path to the file to edit (relative or absolute)"`
	OldText string `json:"old_text" jsonschema_description:"Exact text to find and replace (must match exactly)"`
	NewText string `json:"new_text" jsonschema_description:"Text to replace the old text with"`
}

// EditToolDetails contains detailed diff metadata for an edit operation.
type EditToolDetails struct {
	// Diff is a unified-style diff for the applied edit.
	Diff string
	// FirstChangedLine is the first changed line number in the new content.
	FirstChangedLine int
}

// EditToolOutput is the structured result returned by NewEditTool.
type EditToolOutput struct {
	Content string
	Details EditToolDetails
}

// NewEditTool creates a Genkit tool that executes find-and-replace edits on text files.
func NewEditTool(g *genkit.Genkit, cwd string, opts ...EditToolOption) ai.Tool {
	options := editToolOptions{
		operator: &defaultEditOperator{},
	}
	for _, opt := range opts {
		opt(&options)
	}

	description := "Edit a file by replacing exact text. The oldText must match exactly (including whitespace). Use this for precise, surgical edits."

	return genkit.DefineTool(
		g,
		"edit",
		description,
		func(ctx *ai.ToolContext, input EditToolInput) (EditToolOutput, error) {
			return performEdit(ctx, input, cwd, &options)
		},
	)
}

// performEdit contains the core logic for the edit tool, separated from the
// Genkit registration for testability.
func performEdit(ctx context.Context, input EditToolInput, cwd string, options *editToolOptions) (EditToolOutput, error) {
	ops := options.operator

	// Resolve path relative to cwd
	absolutePath := resolveToCwd(input.Path, cwd)

	if err := ops.Access(ctx, absolutePath); err != nil {
		return EditToolOutput{}, fmt.Errorf("cannot access file: %w", err)
	}

	if err := ctx.Err(); err != nil {
		return EditToolOutput{}, fmt.Errorf("operation cancelled")
	}

	buffer, err := ops.ReadFile(ctx, absolutePath)
	if err != nil {
		return EditToolOutput{}, fmt.Errorf("failed to read file: %w", err)
	}

	rawContent := string(buffer)

	if err := ctx.Err(); err != nil {
		return EditToolOutput{}, fmt.Errorf("operation cancelled")
	}

	bom, content := stripBom(rawContent)

	originalEnding := detectLineEnding(content)
	normalizedContent := normalizeToLF(content)
	normalizedOldText := normalizeToLF(input.OldText)
	normalizedNewText := normalizeToLF(input.NewText)

	matchResult := fuzzyFindText(normalizedContent, normalizedOldText)

	if !matchResult.Found {
		return EditToolOutput{}, fmt.Errorf("Could not find the exact text in %s. The old text must match exactly including all whitespace and newlines.", input.Path)
	}

	fuzzyContent := normalizeForFuzzyMatch(normalizedContent)
	fuzzyOldText := normalizeForFuzzyMatch(normalizedOldText)
	occurrences := len(strings.Split(fuzzyContent, fuzzyOldText)) - 1

	if occurrences > 1 {
		return EditToolOutput{}, fmt.Errorf("Found %d occurences of the text in %s. The text must be unique. Please provide more context to make it unique.", occurrences, input.Path)
	}

	if err := ctx.Err(); err != nil {
		return EditToolOutput{}, fmt.Errorf("operation cancelled")
	}

	// Perform replacement using the matched text position
	// When fuzzy matching was used, ContentForReplacement is the normalized version
	baseContent := matchResult.ContentForReplacement
	newContent := baseContent[:matchResult.Index] + normalizedNewText + baseContent[matchResult.Index+matchResult.MatchLength:]

	if baseContent == newContent {
		return EditToolOutput{}, fmt.Errorf("No changes made to %s. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected", input.Path)
	}

	if err := ctx.Err(); err != nil {
		return EditToolOutput{}, fmt.Errorf("operation cancelled")
	}

	finalContent := bom + restoreLineEndings(newContent, originalEnding)
	if err := ops.WriteFile(ctx, absolutePath, finalContent); err != nil {
		return EditToolOutput{}, fmt.Errorf("failed to write file: %w", err)
	}

	contextLines := CONTEXT_LINES

	diffResult, firstLineChanged := generateDiffString(baseContent, newContent, &contextLines)

	if firstLineChanged == nil {
		firstLineChanged = new(int)
	}

	return EditToolOutput{
		Content: fmt.Sprintf("Successfully replaced text in %s.", input.Path),
		Details: EditToolDetails{
			Diff:             diffResult,
			FirstChangedLine: *firstLineChanged,
		},
	}, nil
}
