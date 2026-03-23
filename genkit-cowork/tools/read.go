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
	"encoding/base64"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/media"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// ReadToolInput is the schema for the read tool's input parameters.
type ReadToolInput struct {
	Path   string `json:"path" jsonschema_description:"Path to the file to read (relative or absolute)"`
	Offset int    `json:"offset,omitempty" jsonschema_description:"Line number to start reading from (1-indexed)"`
	Limit  int    `json:"limit,omitempty" jsonschema_description:"Maximum number of lines to read"`
}

// ReadOperator abstracts file I/O operations so implementations can be
// swapped for testing, sandboxing, or remote execution (e.g., SSH).
type ReadOperator interface {
	// ReadFile reads the entire contents of a file.
	ReadFile(ctx context.Context, absolutePath string) ([]byte, error)
	// Access checks whether a file is readable. Returns an error if not.
	Access(ctx context.Context, absolutePath string) error
	// DetectImageMimeType returns the image MIME type if the file is a
	// supported image format, or "" for non-images.
	DetectImageMimeType(absolutePath string) string
}

type readToolOptions struct {
	autoResizeImages bool
	operator         ReadOperator
}

// ReadToolOption configures a ReadTool via functional options.
type ReadToolOption func(*readToolOptions)

// WithoutAutoResizeImages disables automatic image resizing.
func WithoutAutoResizeImages() ReadToolOption {
	return func(opts *readToolOptions) {
		opts.autoResizeImages = false
	}
}

// WithCustomReadOperator injects a custom ReadOperator implementation.
func WithCustomReadOperator(operator ReadOperator) ReadToolOption {
	return func(opts *readToolOptions) {
		opts.operator = operator
	}
}

// defaultReadOperator is the standard filesystem-backed implementation.
type defaultReadOperator struct{}

func (o *defaultReadOperator) ReadFile(ctx context.Context, absolutePath string) ([]byte, error) {
	return os.ReadFile(absolutePath)
}

func (o *defaultReadOperator) Access(ctx context.Context, absolutePath string) error {
	_, err := os.Stat(absolutePath)
	return err
}

func (o *defaultReadOperator) DetectImageMimeType(absolutePath string) string {
	return media.DetectImageMimeType(absolutePath)
}

// NewReadTool creates a Genkit multipart tool that reads files and images.
// The cwd parameter sets the working directory for resolving relative paths.
func NewReadTool(g *genkit.Genkit, cwd string, opts ...ReadToolOption) ai.Tool {
	options := readToolOptions{
		autoResizeImages: true,
		operator:         &defaultReadOperator{},
	}
	for _, opt := range opts {
		opt(&options)
	}

	description := fmt.Sprintf(
		"Read the contents of a file. Supports text files and images (jpg, png, gif, webp). "+
			"Images are sent as attachments. For text files, output is truncated to %d lines or %sKB "+
			"(whichever is hit first). Use offset/limit for large files. When you need the full file, "+
			"continue with offset until complete.",
		DEFAULT_MAX_LINES, FormatSize(DEFAULT_MAX_BYTES),
	)

	return genkit.DefineMultipartTool(
		g,
		"read",
		description,
		func(ctx *ai.ToolContext, input ReadToolInput) (*ai.MultipartToolResponse, error) {
			return handleRead(ctx, input, cwd, &options)
		},
	)
}

// handleRead contains the core logic for the read tool, separated from the
// Genkit registration for testability.
func handleRead(ctx context.Context, input ReadToolInput, cwd string, options *readToolOptions) (*ai.MultipartToolResponse, error) {
	ops := options.operator

	// Resolve path relative to cwd.
	absolutePath, err := resolveReadPath(input.Path, cwd)
	if err != nil {
		return nil, err
	}

	// Check file is accessible.
	if err := ops.Access(ctx, absolutePath); err != nil {
		return nil, fmt.Errorf("cannot read file: %w", err)
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("operation cancelled")
	}

	// Detect whether this is an image.
	mimeType := ops.DetectImageMimeType(absolutePath)

	if mimeType != "" {
		return handleImageRead(ctx, absolutePath, mimeType, ops, options.autoResizeImages)
	}
	return handleTextRead(ctx, absolutePath, input, ops)
}

// handleImageRead reads an image file, optionally resizes it, and returns
// a multipart response containing a text note and the image data.
func handleImageRead(ctx context.Context, absolutePath, mimeType string, ops ReadOperator, autoResize bool) (*ai.MultipartToolResponse, error) {
	buffer, err := ops.ReadFile(ctx, absolutePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read image file: %w", err)
	}

	if autoResize {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("operation cancelled")
		}

		resized, err := media.AutoResizeImage(buffer, mimeType)
		if err != nil {
			return nil, fmt.Errorf("failed to process image: %w", err)
		}

		textNote := fmt.Sprintf("Read image file [%s]", resized.MimeType)
		dimNote := media.FormatDimensionNote(resized)
		if dimNote != "" {
			textNote += "\n" + dimNote
		}

		dataURI := "data:" + resized.MimeType + ";base64," + resized.Base64
		return &ai.MultipartToolResponse{
			Output: textNote,
			Content: []*ai.Part{
				ai.NewMediaPart(resized.MimeType, dataURI),
			},
		}, nil
	}

	// No auto-resize: return image as-is.
	encoded := base64.StdEncoding.EncodeToString(buffer)
	textNote := fmt.Sprintf("Read image file [%s]", mimeType)
	dataURI := "data:" + mimeType + ";base64," + encoded

	return &ai.MultipartToolResponse{
		Output: textNote,
		Content: []*ai.Part{
			ai.NewMediaPart(mimeType, dataURI),
		},
	}, nil
}

// handleTextRead reads a text file with offset/limit pagination, applies
// truncation, prefixes lines with their file line numbers, and returns
// an actionable message when content is truncated.
func handleTextRead(ctx context.Context, absolutePath string, input ReadToolInput, ops ReadOperator) (*ai.MultipartToolResponse, error) {
	buffer, err := ops.ReadFile(ctx, absolutePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("operation cancelled")
	}

	textContent := string(buffer)
	allLines := splitLines(textContent)
	totalFileLines := len(allLines)

	// Apply offset (1-indexed to 0-indexed).
	startLine := 0
	if input.Offset > 0 {
		startLine = input.Offset - 1
	}
	startLineDisplay := startLine + 1 // for user-facing messages (1-indexed)

	// Validate offset bounds.
	if startLine >= totalFileLines {
		return nil, fmt.Errorf("offset %d is beyond end of file (%d lines total)", input.Offset, totalFileLines)
	}

	// Apply limit.
	var selectedLines []string
	var userLimitedLines int
	hasUserLimit := input.Limit > 0

	if hasUserLimit {
		endLine := startLine + input.Limit
		endLine = min(endLine, totalFileLines) // don't go past end of file
		selectedLines = allLines[startLine:endLine]
		userLimitedLines = endLine - startLine
	} else {
		selectedLines = allLines[startLine:]
	}

	selectedContent := joinLines(selectedLines)

	// Apply truncation (respects both line and byte limits).
	truncation := TruncateHead(selectedContent, nil)

	var outputText string

	if truncation.FirstLineExceedsLimit {
		// First line at offset exceeds the byte limit.
		firstLineSize := FormatSize(len(allLines[startLine]))
		maxSize := FormatSize(DEFAULT_MAX_BYTES)
		outputText = fmt.Sprintf(
			"[Line %d is %s, exceeds %s limit. Use bash: sed -n '%dp' %s | head -c %d]",
			startLineDisplay, firstLineSize, maxSize, startLineDisplay, input.Path, DEFAULT_MAX_BYTES,
		)
	} else if truncation.Truncated {
		// Truncation occurred — prefix lines with numbers and add continuation notice.
		truncatedLines := splitLines(truncation.Content)
		outputText = prefixLineNumbers(truncatedLines, startLineDisplay)

		endLineDisplay := startLineDisplay + truncation.OutputLines - 1
		nextOffset := endLineDisplay + 1

		if truncation.TruncatedBy == "lines" {
			outputText += fmt.Sprintf(
				"\n\n[Showing lines %d-%d of %d. Use offset=%d to continue.]",
				startLineDisplay, endLineDisplay, totalFileLines, nextOffset,
			)
		} else {
			outputText += fmt.Sprintf(
				"\n\n[Showing lines %d-%d of %d (%s limit). Use offset=%d to continue.]",
				startLineDisplay, endLineDisplay, totalFileLines, FormatSize(DEFAULT_MAX_BYTES), nextOffset,
			)
		}
	} else if hasUserLimit && startLine+userLimitedLines < totalFileLines {
		// User specified limit, there's more content, but no truncation.
		outputText = prefixLineNumbers(selectedLines, startLineDisplay)

		remaining := totalFileLines - (startLine + userLimitedLines)
		nextOffset := startLine + userLimitedLines + 1
		outputText += fmt.Sprintf("\n\n[%d more lines in file. Use offset=%d to continue.]", remaining, nextOffset)
	} else {
		// No truncation, no user limit exceeded.
		outputText = prefixLineNumbers(selectedLines, startLineDisplay)
	}

	return &ai.MultipartToolResponse{
		Output: outputText,
	}, nil
}

// prefixLineNumbers prepends each line with its 1-indexed file line number.
// startNum is the line number of the first element in lines.
//
// Example output:
//
//	42: func main() {
//	43:     fmt.Println("hello")
//	44: }
func prefixLineNumbers(lines []string, startNum int) string {
	if len(lines) == 0 {
		return ""
	}

	// Calculate the width needed for the largest line number for alignment.
	maxNum := startNum + len(lines) - 1
	width := len(strconv.Itoa(maxNum))
	format := "%" + strconv.Itoa(width) + "d: %s"

	var b strings.Builder
	// Pre-allocate: each line gets ~width+2 chars for prefix plus content plus newline.
	b.Grow(len(lines) * (width + 4))

	for i, line := range lines {
		if i > 0 {
			b.WriteByte('\n')
		}
		fmt.Fprintf(&b, format, startNum+i, line)
	}
	return b.String()
}
