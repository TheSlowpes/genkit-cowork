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

package media

import (
	"context"
	"fmt"
	"os"

	"github.com/firebase/genkit/go/ai"
)

type ProcessorRegistry struct {
	processors map[string]DocumentProcessor
	fallback   DocumentProcessor
}

type processorOptions struct {
	fallbackProcessor DocumentProcessor
}

type ProcessorOption func(*processorOptions)

func WithFallbackProcessor(processor DocumentProcessor) ProcessorOption {
	return func(opts *processorOptions) {
		opts.fallbackProcessor = processor
	}
}

func NewProcessorRegistry(opts ...ProcessorOption) *ProcessorRegistry {
	options := &processorOptions{
		fallbackProcessor: &defaultDocumentProcessor{},
	}
	for _, opt := range opts {
		opt(options)
	}
	r := &ProcessorRegistry{
		processors: make(map[string]DocumentProcessor),
		fallback:   options.fallbackProcessor,
	}
	r.Register("text/plain", plainTextProcessor{})
	r.Register("text/markdown", markdownProcessor{})
	r.Register("application/json", jsonProcessor{})
	r.Register("text/csv", csvProcessor{})
	r.Register("text/html", htmlProcessor{})
	r.Register("image/jpeg", imageProcessor{mimeType: "image/jpeg"})
	r.Register("image/png", imageProcessor{mimeType: "image/png"})
	r.Register("image/gif", imageProcessor{mimeType: "image/gif"})
	r.Register("image/webp", imageProcessor{mimeType: "image/webp"})
	return r
}

func (r *ProcessorRegistry) Register(mimeType string, processor DocumentProcessor) {
	r.processors[mimeType] = processor
}

func (r *ProcessorRegistry) Get(mimeType string) DocumentProcessor {
	processor, ok := r.processors[mimeType]
	if !ok {
		return r.fallback
	}
	return processor
}

func (r *ProcessorRegistry) ProcessDocument(ctx context.Context, path string) ([]*ai.Document, error) {
	mimeType := DetectMimeType(path)
	processor := r.Get(mimeType)
	if processor == nil {
		return nil, fmt.Errorf("no processor found for MIME type: %s", mimeType)
	}
	file, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}
	documents, err := processor.Process(ctx, file)
	if err != nil {
		return nil, fmt.Errorf("process document: %w", err)
	}
	return documents, nil
}

type defaultDocumentProcessor struct{}

func (p *defaultDocumentProcessor) Process(ctx context.Context, data []byte) ([]*ai.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	_ = data
	return nil, errFileTypeNotSupported
}
