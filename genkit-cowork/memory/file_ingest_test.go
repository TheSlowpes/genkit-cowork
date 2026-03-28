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

package memory

import (
	"context"
	"fmt"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

type stubVectorBackend struct {
	indexed map[string]map[string][]*ai.Document
}

func newStubVectorBackend() *stubVectorBackend {
	return &stubVectorBackend{indexed: make(map[string]map[string][]*ai.Document)}
}

func (b *stubVectorBackend) Index(ctx context.Context, tenantID, sessionID string, docs []*ai.Document) error {
	if _, ok := b.indexed[tenantID]; !ok {
		b.indexed[tenantID] = make(map[string][]*ai.Document)
	}
	b.indexed[tenantID][sessionID] = append(b.indexed[tenantID][sessionID], docs...)
	return nil
}

func (b *stubVectorBackend) RetrieveTenant(ctx context.Context, tenantID, query string, topK int) ([]*ai.Document, error) {
	var out []*ai.Document
	for _, docs := range b.indexed[tenantID] {
		out = append(out, docs...)
	}
	if topK > 0 && len(out) > topK {
		out = out[:topK]
	}
	return out, nil
}

func (b *stubVectorBackend) RetrieveSession(ctx context.Context, tenantID, sessionID, query string, topK int) ([]*ai.Document, error) {
	docs := b.indexed[tenantID][sessionID]
	if topK > 0 && len(docs) > topK {
		docs = docs[:topK]
	}
	return docs, nil
}

func (b *stubVectorBackend) Delete(ctx context.Context, tenantID, sessionID string) error {
	if _, ok := b.indexed[tenantID]; ok {
		delete(b.indexed[tenantID], sessionID)
	}
	return nil
}

func TestFileIngestService_IngestAndSearchTenantFiles(t *testing.T) {
	ctx := context.Background()
	files := NewDefaultFileOperator()
	blobs := NewFileBlobDiskStore(t.TempDir())
	backend := newStubVectorBackend()

	service := NewFileIngestService(files, blobs, nil, NewVectorFileIndexer(backend))

	if _, err := service.Ingest(ctx, FileIngestInput{
		TenantID:      "tenant-1",
		SessionID:     "session-a",
		SourceChannel: UIMessage,
		FileName:      "policy.md",
		Data:          []byte("# Policy\nInvoice dates are on the 1st."),
	}); err != nil {
		t.Fatalf("Ingest(session-a) error = %v", err)
	}

	if _, err := service.Ingest(ctx, FileIngestInput{
		TenantID:      "tenant-1",
		SessionID:     "session-b",
		SourceChannel: UIMessage,
		FileName:      "faq.txt",
		Data:          []byte("Invoices are downloadable as PDF."),
	}); err != nil {
		t.Fatalf("Ingest(session-b) error = %v", err)
	}

	results, err := service.SearchTenantFiles(ctx, FileChunkSearchInput{
		TenantID: "tenant-1",
		Query:    "invoice",
		TopK:     5,
	})
	if err != nil {
		t.Fatalf("SearchTenantFiles() error = %v", err)
	}
	if len(results) == 0 {
		t.Fatal("SearchTenantFiles() returned no results")
	}

	seenSessionA := false
	seenSessionB := false
	for _, result := range results {
		switch result.Chunk.SessionID {
		case "session-a":
			seenSessionA = true
		case "session-b":
			seenSessionB = true
		}
	}
	if !seenSessionA || !seenSessionB {
		t.Fatalf("expected cross-session tenant recall, seenSessionA=%v seenSessionB=%v", seenSessionA, seenSessionB)
	}
}

func TestFileIngestService_TenantIsolation(t *testing.T) {
	ctx := context.Background()
	files := NewDefaultFileOperator()
	blobs := NewFileBlobDiskStore(t.TempDir())
	backend := newStubVectorBackend()
	service := NewFileIngestService(files, blobs, nil, NewVectorFileIndexer(backend))

	if _, err := service.Ingest(ctx, FileIngestInput{
		TenantID: "tenant-a",
		FileName: "notes.txt",
		Data:     []byte("tenant A only"),
	}); err != nil {
		t.Fatalf("Ingest() error = %v", err)
	}

	results, err := service.SearchTenantFiles(ctx, FileChunkSearchInput{
		TenantID: "tenant-b",
		Query:    "tenant",
		TopK:     5,
	})
	if err != nil {
		t.Fatalf("SearchTenantFiles() error = %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("cross-tenant search returned %d results, want 0", len(results))
	}
}

func TestBuildFileChunks_Deterministic(t *testing.T) {
	in := fileChunkBuildInput{
		TenantID:    "tenant-1",
		FileID:      "file-1",
		FileName:    "doc.txt",
		MimeType:    "text/plain",
		Checksum:    "abc",
		Text:        "one two three four five six seven eight nine ten",
		ChunkSize:   18,
		OverlapSize: 3,
	}

	a := buildFileChunks(in)
	b := buildFileChunks(in)

	if len(a) != len(b) {
		t.Fatalf("chunk count mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].ChunkID != b[i].ChunkID {
			t.Fatalf("chunkID mismatch at %d: %q vs %q", i, a[i].ChunkID, b[i].ChunkID)
		}
		if a[i].Text != b[i].Text {
			t.Fatalf("chunk text mismatch at %d: %q vs %q", i, a[i].Text, b[i].Text)
		}
	}
}

func TestFileIngestService_UnsupportedMimeFails(t *testing.T) {
	ctx := context.Background()
	files := NewDefaultFileOperator()
	blobs := NewFileBlobDiskStore(t.TempDir())
	service := NewFileIngestService(files, blobs, nil, nil)

	_, err := service.Ingest(ctx, FileIngestInput{
		TenantID: "tenant-1",
		FileName: "blob.bin",
		Data:     []byte{0x00, 0x01, 0x02},
	})
	if err == nil {
		t.Fatal("expected unsupported mime error, got nil")
	}
}

func TestSearchTenantFileChunksHelper(t *testing.T) {
	ctx := context.Background()
	files := NewDefaultFileOperator()
	blobs := NewFileBlobDiskStore(t.TempDir())
	backend := newStubVectorBackend()
	service := NewFileIngestService(files, blobs, nil, NewVectorFileIndexer(backend))

	if _, err := service.Ingest(ctx, FileIngestInput{
		TenantID: "tenant-1",
		FileName: "doc.txt",
		Data:     []byte("invoice details"),
	}); err != nil {
		t.Fatalf("Ingest() error = %v", err)
	}

	results, err := SearchTenantFileChunks(ctx, service, "tenant-1", "invoice", 3)
	if err != nil {
		t.Fatalf("SearchTenantFileChunks() error = %v", err)
	}
	if len(results) == 0 {
		t.Fatal("SearchTenantFileChunks() returned no results")
	}
}

func TestFileIngestService_RequiresBlobStore(t *testing.T) {
	service := NewFileIngestService(NewDefaultFileOperator(), nil, nil, nil)
	_, err := service.Ingest(context.Background(), FileIngestInput{
		TenantID: "tenant-1",
		FileName: "x.txt",
		Data:     []byte("hello"),
	})
	if err == nil {
		t.Fatal("expected error when blob store is missing")
	}
	if got := err.Error(); got == "" {
		t.Fatal("expected non-empty error")
	}
}

func TestFileIngestService_SearchRequiresIndexer(t *testing.T) {
	service := NewFileIngestService(NewDefaultFileOperator(), NewFileBlobDiskStore(t.TempDir()), nil, nil)
	_, err := service.SearchTenantFiles(context.Background(), FileChunkSearchInput{
		TenantID: "tenant-1",
		Query:    "invoice",
		TopK:     5,
	})
	if err == nil {
		t.Fatal("expected error when indexer is missing")
	}
	if got := err.Error(); got == "" {
		t.Fatal("expected non-empty error")
	}
}

func TestBuildFileChunks_ProducesExpectedRanges(t *testing.T) {
	text := "abcdefghijklmnopqrstuvwxyz"
	chunks := buildFileChunks(fileChunkBuildInput{
		TenantID:    "tenant-1",
		FileID:      "file-1",
		FileName:    "alpha.txt",
		MimeType:    "text/plain",
		Checksum:    "sum",
		Text:        text,
		ChunkSize:   10,
		OverlapSize: 2,
	})

	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(chunks))
	}
	if chunks[0].CharStart != 0 {
		t.Fatalf("first chunk start = %d, want 0", chunks[0].CharStart)
	}
	for i := 1; i < len(chunks); i++ {
		if chunks[i].CharStart > chunks[i-1].CharEnd {
			t.Fatalf("expected overlap/adjacency between chunks %d and %d", i-1, i)
		}
	}
}

func TestEstimateTextTokens(t *testing.T) {
	tests := []struct {
		text string
		want int
	}{
		{text: "", want: 0},
		{text: "hello", want: 1},
		{text: "hello world from tests", want: 4},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%q", tc.text), func(t *testing.T) {
			if got := estimateTextTokens(tc.text); got != tc.want {
				t.Fatalf("estimateTextTokens(%q) = %d, want %d", tc.text, got, tc.want)
			}
		})
	}
}
