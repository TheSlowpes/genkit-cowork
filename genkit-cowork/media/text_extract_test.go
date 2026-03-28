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
	"errors"
	"strings"
	"testing"
)

func TestDetectMimeType_ByExtension(t *testing.T) {
	tests := []struct {
		name     string
		fileName string
		want     string
	}{
		{name: "markdown", fileName: "notes.md", want: "text/markdown"},
		{name: "plain text", fileName: "notes.txt", want: "text/plain"},
		{name: "json", fileName: "payload.json", want: "application/json"},
		{name: "csv", fileName: "table.csv", want: "text/csv"},
		{name: "html", fileName: "index.html", want: "text/html"},
		{name: "unsupported", fileName: "archive.bin", want: ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := DetectMimeType(nil, tc.fileName); got != tc.want {
				t.Fatalf("DetectMimeType(_, %q) = %q, want %q", tc.fileName, got, tc.want)
			}
		})
	}
}

func TestDefaultTextExtractor_Extract(t *testing.T) {
	extractor := NewDefaultTextExtractor()
	ctx := context.Background()

	tests := []struct {
		name     string
		input    TextExtractInput
		contains string
	}{
		{
			name: "plain",
			input: TextExtractInput{
				FileName: "a.txt",
				Data:     []byte("hello world"),
			},
			contains: "hello world",
		},
		{
			name: "json",
			input: TextExtractInput{
				FileName: "a.json",
				Data:     []byte(`{"a":1,"b":"x"}`),
			},
			contains: `"a": 1`,
		},
		{
			name: "csv",
			input: TextExtractInput{
				FileName: "a.csv",
				Data:     []byte("col1,col2\nva,vb\n"),
			},
			contains: "col1\tcol2",
		},
		{
			name: "html",
			input: TextExtractInput{
				FileName: "a.html",
				Data:     []byte("<h1>Title</h1><p>Hello</p>"),
			},
			contains: "Title Hello",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := extractor.Extract(ctx, tc.input)
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			if result.Text == "" {
				t.Fatal("Extract() returned empty text")
			}
			if !strings.Contains(result.Text, tc.contains) {
				t.Fatalf("Extract() text = %q, expected to contain %q", result.Text, tc.contains)
			}
			if result.Metadata["mimeType"] == "" {
				t.Fatal("Extract() metadata mimeType was empty")
			}
		})
	}
}

func TestDefaultTextExtractor_UnsupportedMime(t *testing.T) {
	extractor := NewDefaultTextExtractor()
	_, err := extractor.Extract(context.Background(), TextExtractInput{
		FileName: "a.xml",
		MimeType: "application/xml",
		Data:     []byte("<root />"),
	})
	if err == nil {
		t.Fatal("expected unsupported mime type error, got nil")
	}
	if !errors.Is(err, ErrUnsupportedTextMimeType) {
		t.Fatalf("errors.Is(err, ErrUnsupportedTextMimeType) = false, err = %v", err)
	}
}

func TestDefaultTextExtractor_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	extractor := NewDefaultTextExtractor()
	_, err := extractor.Extract(ctx, TextExtractInput{
		FileName: "a.txt",
		Data:     []byte("hello"),
	})
	if err == nil {
		t.Fatal("expected context cancellation error, got nil")
	}
}
