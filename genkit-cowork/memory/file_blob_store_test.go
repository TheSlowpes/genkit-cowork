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
)

func TestFileBlobDiskStore_PutFile_RejectsPathTraversal(t *testing.T) {
	ctx := context.Background()
	store := NewFileBlobDiskStore(t.TempDir())

	cases := []struct {
		name              string
		tenantID, fileID string
	}{
		{"dotdot tenantID", "..", "file-1"},
		{"slash tenantID", "tenant/evil", "file-1"},
		{"dotdot fileID", "tenant-a", ".."},
		{"slash fileID", "tenant-a", "file/evil"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := store.PutFile(ctx, tc.tenantID, tc.fileID, "doc.txt", "text/plain", []byte("x")); err == nil {
				t.Fatalf("PutFile(%q, %q) expected error, got nil", tc.tenantID, tc.fileID)
			}
		})
	}
}

func TestFileBlobDiskStore_ReadFile_ValidatesTenantScope(t *testing.T) {
	ctx := context.Background()
	store := NewFileBlobDiskStore(t.TempDir())

	path, err := store.PutFile(ctx, "tenant-a", "file-1", "doc.txt", "text/plain", []byte("hello"))
	if err != nil {
		t.Fatalf("PutFile() error = %v", err)
	}

	data, err := store.ReadFile(ctx, "tenant-a", "file-1", path)
	if err != nil {
		t.Fatalf("ReadFile(same tenant) error = %v", err)
	}
	if got, want := string(data), "hello"; got != want {
		t.Fatalf("ReadFile() data = %q, want %q", got, want)
	}

	if _, err := store.ReadFile(ctx, "tenant-b", "file-1", path); err == nil {
		t.Fatal("ReadFile(cross-tenant) expected error, got nil")
	}
}

func TestFileBlobDiskStore_PutFile_RequiresTenantAndFileID(t *testing.T) {
	ctx := context.Background()
	store := NewFileBlobDiskStore(t.TempDir())

	if _, err := store.PutFile(ctx, "", "file-1", "doc.txt", "text/plain", []byte("hello")); err == nil {
		t.Fatal("PutFile() expected error for empty tenantID, got nil")
	}
	if _, err := store.PutFile(ctx, "tenant-a", "", "doc.txt", "text/plain", []byte("hello")); err == nil {
		t.Fatal("PutFile() expected error for empty fileID, got nil")
	}
}

func TestFileBlobDiskStore_ReadFile_RequiresTenantAndFileID(t *testing.T) {
	ctx := context.Background()
	store := NewFileBlobDiskStore(t.TempDir())

	path, err := store.PutFile(ctx, "tenant-a", "file-1", "doc.txt", "text/plain", []byte("hello"))
	if err != nil {
		t.Fatalf("PutFile() error = %v", err)
	}

	if _, err := store.ReadFile(ctx, "", "file-1", path); err == nil {
		t.Fatal("ReadFile() expected error for empty tenantID, got nil")
	}
	if _, err := store.ReadFile(ctx, "tenant-a", "", path); err == nil {
		t.Fatal("ReadFile() expected error for empty fileID, got nil")
	}
}
