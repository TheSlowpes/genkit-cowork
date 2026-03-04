package tools

import (
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type editToolOptions struct {
	operator EditOperator
}

type EditToolOption func(*editToolOptions)

// EditOperator abstracs file editing operations so implementations can be
// swapped for testing, sandboxing, or remote execution (e.g., SSH).
type EditOperator interface {
	ReadFile(absolutePath string) ([]byte, error)
	WriteFile(absolutePath, content string) error
	Access(absolutePath string) error
}

func WithCustomEditOperator(operator EditOperator) EditToolOption {
	return func(opts *editToolOptions) {
		opts.operator = operator
	}
}

// EditToolInput is the schema for the input to the EditTool. It specifies
// the file to edit and the text to find and replace.
type EditToolInput struct {
	Path    string `json:"path" jsonschema_description:"Path to the file to edit (relative or absolute)"`
	OldText string `json:"old_text" jsonschema_description:"Exact text to find and replace (must match exactly)"`
	NewText string `json:"new_text" jsonschema_description:"Text to replace the old text with"`
}

type EditToolDetails struct {
	Diff             string
	FirstChangedLine int
}

type EditToolOutput struct {
	Content string
	Details EditToolDetails
}

// NewEditTool creates a Genkit tool that executes find-and-replace edits on text files.
func NewEditTool(g *genkit.Genkit, cwd string, opts ...EditToolOption) ai.Tool {
	options := editToolOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	description := "Edit a file by replacing exact text. The oldText must match exactly (including whitespace). Use this for precise, surgical edits."

	return genkit.DefineTool(
		g,
		"edit",
		description,
		func(ctx *ai.ToolContext, input EditToolInput) (EditToolOutput, error) {
			return performEdit(input, cwd, &options)
		},
	)
}

// performEdit contains the core logic for the edit tool, separated from the
// Genkit registration for testability.
func performEdit(input EditToolInput, cwd string, options *editToolOptions) (EditToolOutput, error) {
	ops := options.operator

	// Resolve path relative to cwd

}
