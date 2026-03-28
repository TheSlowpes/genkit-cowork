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
	"errors"
	"testing"
	"time"
)

func TestDefaultFileOperator_SaveAndLoadRecord(t *testing.T) {
	op := NewDefaultFileOperator()
	ctx := context.Background()

	record := FileRecord{
		FileID:       "file-1",
		TenantID:     "tenant-a",
		Name:         "notes.md",
		MimeType:     "text/markdown",
		UploadedAt:   time.Now().UTC(),
		IngestStatus: FileIngestPending,
	}

	if err := op.SaveFileRecord(ctx, "tenant-a", record); err != nil {
		t.Fatalf("SaveFileRecord() error = %v", err)
	}

	got, err := op.LoadFileRecord(ctx, "tenant-a", "file-1")
	if err != nil {
		t.Fatalf("LoadFileRecord() error = %v", err)
	}
	if got == nil {
		t.Fatal("LoadFileRecord() returned nil record")
	}
	if got.TenantID != "tenant-a" {
		t.Fatalf("TenantID = %q, want %q", got.TenantID, "tenant-a")
	}
}

func TestDefaultFileOperator_TenantIsolation(t *testing.T) {
	op := NewDefaultFileOperator()
	ctx := context.Background()

	record := FileRecord{FileID: "file-1", Name: "a.txt", MimeType: "text/plain", UploadedAt: time.Now().UTC()}
	if err := op.SaveFileRecord(ctx, "tenant-a", record); err != nil {
		t.Fatalf("SaveFileRecord() error = %v", err)
	}

	got, err := op.LoadFileRecord(ctx, "tenant-b", "file-1")
	if err != nil {
		t.Fatalf("LoadFileRecord() error = %v", err)
	}
	if got != nil {
		t.Fatalf("cross-tenant LoadFileRecord() = %+v, want nil", got)
	}
}

func TestDefaultFileOperator_SaveChunksDeduplicates(t *testing.T) {
	op := NewDefaultFileOperator()
	ctx := context.Background()

	chunks := []FileChunkRecord{{ChunkID: "c1", Index: 0, Text: "a"}, {ChunkID: "c1", Index: 0, Text: "a"}}
	if err := op.SaveFileChunks(ctx, "tenant-a", "file-1", chunks); err != nil {
		t.Fatalf("SaveFileChunks() error = %v", err)
	}

	got, err := op.ListFileChunks(ctx, "tenant-a", "file-1")
	if err != nil {
		t.Fatalf("ListFileChunks() error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("len(ListFileChunks()) = %d, want 1", len(got))
	}
}

func TestDefaultFileOperator_ContextCancelled(t *testing.T) {
	op := NewDefaultFileOperator()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := op.SaveFileRecord(ctx, "tenant-a", FileRecord{FileID: "f1"})
	if err == nil {
		t.Fatal("SaveFileRecord() expected context cancellation error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("errors.Is(err, context.Canceled) = false, err = %v", err)
	}
}

func TestFileRecord_MetadataRoundTrip(t *testing.T) {
	record := FileRecord{}
	if err := record.SetMetadata(map[string]string{"source": "upload"}); err != nil {
		t.Fatalf("SetMetadata() error = %v", err)
	}

	got, err := record.Metadata()
	if err != nil {
		t.Fatalf("Metadata() error = %v", err)
	}
	if got["source"] != "upload" {
		t.Fatalf("Metadata()[source] = %q, want %q", got["source"], "upload")
	}
}
