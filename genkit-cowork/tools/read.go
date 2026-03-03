package tools

import (
	"encoding/base64"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type ReadToolInput struct {
	Path   string `json:"path" jsonschema_description:"Path to the file to read (relative or absolute)"`
	Offset int    `json:"offset,omitempty" jsonschema_description:"Line number to start reading from (1-indexed)"`
	Limit  int    `json:"limit,omitempty" jsonschema_description:"Maximum number of lines to read"`
}

type ReadOperator interface {
	ReadFile(absolutePath string) ([]byte, error)
	DetectImageMimeType(absolutePath string) string
}

type readToolOptions struct {
	autoResizeImages bool
	operations       ReadOperator
}

type ReadToolOption func(*readToolOptions)

func WithoutAutoResizeImages() ReadToolOption {
	return func(opts *readToolOptions) {
		opts.autoResizeImages = false
	}
}

func WithCustomReadOperator(operations ReadOperator) ReadToolOption {
	return func(opts *readToolOptions) {
		opts.operations = operations
	}
}

func NewReadTool(g *genkit.Genkit, opts ...ReadToolOption) ai.Tool {
	options := readToolOptions{
		autoResizeImages: true,
	}
	for _, opt := range opts {
		opt(&options)
	}
	return genkit.DefineMultipartTool(
		g,
		"read",
		fmt.Sprintf("Read the contents of a file. Supports text files and images (jpg, png, gif, webp). Images are sent as attachments. For text files, output is truncated to %d lines or %dKB (whichever is hit first) Use offset/limit for large files.", DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES),
		func(ctx *ai.ToolContext, input ReadToolInput) (*ai.MultipartToolResponse, error) {
			absolutePath, err := resolveReadPath(input.Path)
			if err != nil {
				return nil, fmt.Errorf("failed to resolve path: %w", err)
			}
			mimeType := options.operations.DetectImageMimeType(absolutePath)

			if mimeType != "" {
				buffer, err := options.operations.ReadFile(absolutePath)
				encondedBuffer := base64.StdEncoding.EncodeToString(buffer)

				if options.autoResizeImages {

				}
			}

			return nil, nil
		},
	)
}
