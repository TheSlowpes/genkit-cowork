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
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileMediaAssetStore_PutTenantScopedPathAndExtension(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store := NewFileMediaAssetStore(root)

	path, err := store.Put(ctx, "tenant-1", "session-1", "asset-1", "image/png", []byte("hello"))
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	wantDir := filepath.Join("tenant-1", "session-1", "assets")
	if !strings.Contains(path, wantDir) {
		t.Fatalf("Put() path = %q, want to contain %q", path, wantDir)
	}
	if !strings.HasSuffix(path, ".png") {
		t.Fatalf("Put() path = %q, want suffix .png", path)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile(%q) error = %v", path, err)
	}
	if got, want := string(data), "hello"; got != want {
		t.Fatalf("stored bytes = %q, want %q", got, want)
	}
}

func TestFileMediaAssetStore_DeleteSessionAssetsTenantScoped(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store := NewFileMediaAssetStore(root)

	pathA, err := store.Put(ctx, "tenant-a", "session-1", "asset-a", "text/plain", []byte("a"))
	if err != nil {
		t.Fatalf("Put(tenant-a) error = %v", err)
	}
	pathB, err := store.Put(ctx, "tenant-b", "session-1", "asset-b", "text/plain", []byte("b"))
	if err != nil {
		t.Fatalf("Put(tenant-b) error = %v", err)
	}

	if err := store.DeleteSessionAssets(ctx, "tenant-a", "session-1"); err != nil {
		t.Fatalf("DeleteSessionAssets() error = %v", err)
	}

	if _, err := os.Stat(pathA); !os.IsNotExist(err) {
		t.Fatalf("tenant-a asset should be deleted, stat err = %v", err)
	}
	if _, err := os.Stat(pathB); err != nil {
		t.Fatalf("tenant-b asset should still exist, stat err = %v", err)
	}
}
