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
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileMediaAssetStore_Put_RejectsPathTraversal(t *testing.T) {
	ctx := context.Background()
	store := NewFileMediaAssetStore(t.TempDir())

	cases := []struct {
		name                         string
		tenantID, sessionID, assetID string
	}{
		{"empty tenantID", "", "session-1", "asset-1"},
		{"dotdot tenantID", "..", "session-1", "asset-1"},
		{"slash tenantID", "tenant/evil", "session-1", "asset-1"},
		{"empty sessionID", "tenant-1", "", "asset-1"},
		{"dotdot sessionID", "tenant-1", "..", "asset-1"},
		{"slash sessionID", "tenant-1", "session/evil", "asset-1"},
		{"empty assetID", "tenant-1", "session-1", ""},
		{"dotdot assetID", "tenant-1", "session-1", ".."},
		{"slash assetID", "tenant-1", "session-1", "asset/evil"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := store.Put(ctx, tc.tenantID, tc.sessionID, tc.assetID, "text/plain", []byte("x")); err == nil {
				t.Fatalf("Put(%q, %q, %q) expected error, got nil", tc.tenantID, tc.sessionID, tc.assetID)
			}
		})
	}
}

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

func TestFileMediaAssetStore_PutWritesAssetManifest(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store := NewFileMediaAssetStore(root)

	if _, err := store.Put(ctx, "tenant-1", "session-1", "asset-1", "text/plain", []byte("hello world")); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	assets, err := store.ListAssets(ctx, "tenant-1", "session-1")
	if err != nil {
		t.Fatalf("ListAssets() error = %v", err)
	}
	if len(assets) != 1 {
		t.Fatalf("len(ListAssets()) = %d, want 1", len(assets))
	}
	if assets[0].AssetID != "asset-1" {
		t.Fatalf("assetID = %q, want %q", assets[0].AssetID, "asset-1")
	}
	if assets[0].IngestStatus != AssetIngestPending {
		t.Fatalf("ingestStatus = %q, want %q", assets[0].IngestStatus, AssetIngestPending)
	}

	indexPath := filepath.Join(root, "tenant-1", "session-1", "assets", "index.json")
	b, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatalf("ReadFile(index.json) error = %v", err)
	}
	if len(b) == 0 {
		t.Fatal("index.json was empty")
	}
}

func TestFileMediaAssetStore_LoadWritesChunkIndexJSON(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store := NewFileMediaAssetStore(root)

	if _, err := store.Put(ctx, "tenant-1", "session-1", "asset-1", "text/plain", []byte(strings.Repeat("abc ", 500))); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	docs, err := store.Load(ctx, "tenant-1", "session-1", "asset-1")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if len(docs) == 0 {
		t.Fatal("Load() returned no docs")
	}

	chunkPath := filepath.Join(root, "tenant-1", "session-1", "assets", "asset-1.index.json")
	b, err := os.ReadFile(chunkPath)
	if err != nil {
		t.Fatalf("ReadFile(chunk index) error = %v", err)
	}

	var index []map[string]any
	if err := json.Unmarshal(b, &index); err != nil {
		t.Fatalf("Unmarshal(chunk index) error = %v", err)
	}
	if len(index) != len(docs) {
		t.Fatalf("chunk index len = %d, want %d", len(index), len(docs))
	}

	assets, err := store.ListAssets(ctx, "tenant-1", "session-1")
	if err != nil {
		t.Fatalf("ListAssets() error = %v", err)
	}
	if assets[0].IngestStatus != AssetIngestCompleted {
		t.Fatalf("ingestStatus = %q, want %q", assets[0].IngestStatus, AssetIngestCompleted)
	}
}

func TestFileMediaAssetStore_LoadUnsupportedTypeReturnsExpectedError(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store := NewFileMediaAssetStore(root)

	if _, err := store.Put(ctx, "tenant-1", "session-1", "asset-1", "application/pdf", []byte("%PDF-1.7")); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	_, err := store.Load(ctx, "tenant-1", "session-1", "asset-1")
	if err == nil {
		t.Fatal("Load() expected error, got nil")
	}
	if got, want := err.Error(), "process document: file type not supported"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}
