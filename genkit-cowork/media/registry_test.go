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
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func TestProcessorRegistry_ProcessDocument_TextMarkdown(t *testing.T) {
	ctx := context.Background()
	r := NewProcessorRegistry()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "policy.md")
	if err := os.WriteFile(path, []byte("# Policy\nInvoice policy text."), 0644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	docs, err := r.ProcessDocument(ctx, path)
	if err != nil {
		t.Fatalf("ProcessDocument() error = %v", err)
	}
	if len(docs) == 0 {
		t.Fatal("ProcessDocument() returned no docs")
	}

	mimeType, _ := docs[0].Metadata["mimeType"].(string)
	if mimeType != "text/markdown" {
		t.Fatalf("metadata mimeType = %q, want %q", mimeType, "text/markdown")
	}
}

func TestProcessorRegistry_ProcessDocument_UnsupportedReturnsExpectedError(t *testing.T) {
	ctx := context.Background()
	r := NewProcessorRegistry()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "blob.bin")
	if err := os.WriteFile(path, []byte{0x00, 0x01, 0x02, 0x03}, 0644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	_, err := r.ProcessDocument(ctx, path)
	if err == nil {
		t.Fatal("ProcessDocument() expected error, got nil")
	}
	if got, want := err.Error(), "process document: file type not supported"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func TestProcessorRegistry_ProcessDocument_JSONProducesDocs(t *testing.T) {
	ctx := context.Background()
	r := NewProcessorRegistry()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "data.json")
	if err := os.WriteFile(path, []byte(`{"invoice":"A-100","amount":42}`), 0644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	docs, err := r.ProcessDocument(ctx, path)
	if err != nil {
		t.Fatalf("ProcessDocument() error = %v", err)
	}
	if len(docs) == 0 {
		t.Fatal("ProcessDocument() returned no docs")
	}
	mimeType, _ := docs[0].Metadata["mimeType"].(string)
	if mimeType != "application/json" {
		t.Fatalf("metadata mimeType = %q, want %q", mimeType, "application/json")
	}
}

func TestProcessorRegistry_ProcessDocument_ImageReturnsMediaPart(t *testing.T) {
	ctx := context.Background()
	r := NewProcessorRegistry()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "image.png")

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	if err := png.Encode(f, img); err != nil {
		_ = f.Close()
		t.Fatalf("png.Encode() error = %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	docs, err := r.ProcessDocument(ctx, path)
	if err != nil {
		t.Fatalf("ProcessDocument() error = %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("len(docs) = %d, want 1", len(docs))
	}
	if len(docs[0].Content) != 1 {
		t.Fatalf("len(docs[0].Content) = %d, want 1", len(docs[0].Content))
	}
	part := docs[0].Content[0]
	if part == nil || !part.IsMedia() {
		t.Fatal("expected media part")
	}
	if part.ContentType != "image/png" && part.ContentType != "image/jpeg" {
		t.Fatalf("media contentType = %q, want image/png or image/jpeg", part.ContentType)
	}
}
