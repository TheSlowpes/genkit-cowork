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
	"testing"
	"time"
)

func TestFileRecordOperator_SaveAndList(t *testing.T) {
	root := t.TempDir()
	op := NewFileRecordOperator(root)
	ctx := context.Background()

	recordA := FileRecord{FileID: "f-b", Name: "b.txt", MimeType: "text/plain", UploadedAt: time.Now().UTC()}
	recordB := FileRecord{FileID: "f-a", Name: "a.txt", MimeType: "text/plain", UploadedAt: time.Now().UTC()}

	if err := op.SaveFileRecord(ctx, "tenant-1", recordA); err != nil {
		t.Fatalf("SaveFileRecord(recordA) error = %v", err)
	}
	if err := op.SaveFileRecord(ctx, "tenant-1", recordB); err != nil {
		t.Fatalf("SaveFileRecord(recordB) error = %v", err)
	}

	files, err := op.ListFiles(ctx, "tenant-1")
	if err != nil {
		t.Fatalf("ListFiles() error = %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("len(ListFiles()) = %d, want 2", len(files))
	}
	if files[0].FileID != "f-a" || files[1].FileID != "f-b" {
		t.Fatalf("ListFiles() order = [%s %s], want [f-a f-b]", files[0].FileID, files[1].FileID)
	}
}

func TestFileRecordOperator_SaveAndListChunks(t *testing.T) {
	root := t.TempDir()
	op := NewFileRecordOperator(root)
	ctx := context.Background()

	chunks := []FileChunkRecord{
		{ChunkID: "c2", Index: 2, Text: "world"},
		{ChunkID: "c1", Index: 1, Text: "hello"},
		{ChunkID: "c1", Index: 1, Text: "hello"},
	}

	if err := op.SaveFileChunks(ctx, "tenant-1", "f1", chunks); err != nil {
		t.Fatalf("SaveFileChunks() error = %v", err)
	}

	got, err := op.ListFileChunks(ctx, "tenant-1", "f1")
	if err != nil {
		t.Fatalf("ListFileChunks() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(ListFileChunks()) = %d, want 2", len(got))
	}
	if got[0].ChunkID != "c1" || got[1].ChunkID != "c2" {
		t.Fatalf("ListFileChunks() order = [%s %s], want [c1 c2]", got[0].ChunkID, got[1].ChunkID)
	}
}

func TestFileRecordOperator_TenantIsolation(t *testing.T) {
	root := t.TempDir()
	op := NewFileRecordOperator(root)
	ctx := context.Background()

	if err := op.SaveFileRecord(ctx, "tenant-a", FileRecord{FileID: "f1", Name: "a.txt", MimeType: "text/plain"}); err != nil {
		t.Fatalf("SaveFileRecord() error = %v", err)
	}

	got, err := op.LoadFileRecord(ctx, "tenant-b", "f1")
	if err != nil {
		t.Fatalf("LoadFileRecord() error = %v", err)
	}
	if got != nil {
		t.Fatalf("cross-tenant LoadFileRecord() = %+v, want nil", got)
	}
}
