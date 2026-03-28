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
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"mime"
	"net/http"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	// ErrUnsupportedTextMimeType indicates there is no extractor registered for
	// the provided MIME type.
	ErrUnsupportedTextMimeType = errors.New("unsupported text mime type")

	htmlTagRegexp = regexp.MustCompile(`<[^>]+>`)
)

// TextExtractInput contains raw file bytes and metadata used by MIME-aware
// text extraction.
type TextExtractInput struct {
	FileName string
	MimeType string
	Data     []byte
}

// TextSection represents one extracted logical section from a document.
type TextSection struct {
	Title string
	Text  string
}

// TextExtractResult contains normalized text content and optional structure.
type TextExtractResult struct {
	Text     string
	Title    string
	Sections []TextSection
	Metadata map[string]string
}

// TextExtractor extracts canonical text from one MIME type family.
type TextExtractor interface {
	Extract(ctx context.Context, input TextExtractInput) (TextExtractResult, error)
}

// MultiTextExtractor dispatches extraction to MIME-specific handlers.
type MultiTextExtractor struct {
	extractors map[string]TextExtractor
}

// NewDefaultTextExtractor returns a MIME-aware extractor for text and
// structured formats used by tenant-global file memory ingestion.
func NewDefaultTextExtractor() *MultiTextExtractor {
	return &MultiTextExtractor{
		extractors: map[string]TextExtractor{
			"text/plain":       plainTextExtractor{},
			"text/markdown":    plainTextExtractor{},
			"application/json": jsonTextExtractor{},
			"text/csv":         csvTextExtractor{},
			"text/html":        htmlTextExtractor{},
		},
	}
}

// Extract extracts canonical text from input using the configured MIME
// extractors.
func (e *MultiTextExtractor) Extract(ctx context.Context, input TextExtractInput) (TextExtractResult, error) {
	if err := ctx.Err(); err != nil {
		return TextExtractResult{}, err
	}

	mimeType := normalizeMimeType(input.MimeType)
	if mimeType == "" {
		mimeType = DetectMimeType(input.Data, input.FileName)
	}

	extractor, ok := e.extractors[mimeType]
	if !ok {
		return TextExtractResult{}, fmt.Errorf("%w: %s", ErrUnsupportedTextMimeType, mimeType)
	}

	result, err := extractor.Extract(ctx, input)
	if err != nil {
		return TextExtractResult{}, err
	}
	if result.Metadata == nil {
		result.Metadata = make(map[string]string)
	}
	result.Metadata["mimeType"] = mimeType
	return result, nil
}

// DetectMimeType determines MIME type from content and filename extension.
//
// It returns an empty string when no supported text/structured MIME can be
// determined.
func DetectMimeType(data []byte, fileName string) string {
	ext := strings.ToLower(filepath.Ext(fileName))
	extMime := ""
	switch ext {
	case ".md", ".markdown":
		extMime = "text/markdown"
	case ".txt":
		extMime = "text/plain"
	case ".json":
		extMime = "application/json"
	case ".csv":
		extMime = "text/csv"
	case ".html", ".htm":
		extMime = "text/html"
	}

	if len(data) > 0 {
		detected := normalizeMimeType(http.DetectContentType(data))
		if detected == "text/plain" && extMime != "" && extMime != "text/plain" {
			return extMime
		}
		if isSupportedTextMimeType(detected) {
			return detected
		}
	}

	return extMime
}

func normalizeMimeType(value string) string {
	if value == "" {
		return ""
	}
	mediaType, _, err := mime.ParseMediaType(value)
	if err != nil {
		return strings.TrimSpace(strings.ToLower(value))
	}
	return strings.TrimSpace(strings.ToLower(mediaType))
}

func isSupportedTextMimeType(value string) bool {
	switch normalizeMimeType(value) {
	case "text/plain", "text/markdown", "application/json", "text/csv", "text/html":
		return true
	default:
		return false
	}
}

type plainTextExtractor struct{}

func (plainTextExtractor) Extract(ctx context.Context, input TextExtractInput) (TextExtractResult, error) {
	if err := ctx.Err(); err != nil {
		return TextExtractResult{}, err
	}
	text := strings.TrimSpace(string(input.Data))
	return TextExtractResult{Text: text}, nil
}

type jsonTextExtractor struct{}

func (jsonTextExtractor) Extract(ctx context.Context, input TextExtractInput) (TextExtractResult, error) {
	if err := ctx.Err(); err != nil {
		return TextExtractResult{}, err
	}

	var payload any
	if err := json.Unmarshal(input.Data, &payload); err != nil {
		return TextExtractResult{}, fmt.Errorf("parse json: %w", err)
	}

	formatted, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return TextExtractResult{}, fmt.Errorf("format json: %w", err)
	}

	return TextExtractResult{
		Text: strings.TrimSpace(string(formatted)),
	}, nil
}

type csvTextExtractor struct{}

func (csvTextExtractor) Extract(ctx context.Context, input TextExtractInput) (TextExtractResult, error) {
	if err := ctx.Err(); err != nil {
		return TextExtractResult{}, err
	}

	r := csv.NewReader(bytes.NewReader(input.Data))
	records, err := r.ReadAll()
	if err != nil {
		return TextExtractResult{}, fmt.Errorf("parse csv: %w", err)
	}

	var b strings.Builder
	for i, record := range records {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(strings.Join(record, "\t"))
	}

	return TextExtractResult{
		Text: strings.TrimSpace(b.String()),
	}, nil
}

type htmlTextExtractor struct{}

func (htmlTextExtractor) Extract(ctx context.Context, input TextExtractInput) (TextExtractResult, error) {
	if err := ctx.Err(); err != nil {
		return TextExtractResult{}, err
	}

	raw := string(input.Data)
	withoutTags := htmlTagRegexp.ReplaceAllString(raw, " ")
	plain := strings.Join(strings.Fields(html.UnescapeString(withoutTags)), " ")

	return TextExtractResult{
		Text: plain,
	}, nil
}
