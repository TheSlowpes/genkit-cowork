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
	"mime"
	"os"
	"path/filepath"
)

type FileMediaAssetStore struct {
	RootDir string
}

// NewFileMediaAssetStore creates a filesystem-backed media asset store.
func NewFileMediaAssetStore(rootDir string) *FileMediaAssetStore {
	return &FileMediaAssetStore{RootDir: rootDir}
}

// Put stores raw media bytes under the session assets directory and returns the
// absolute file path.
func (s *FileMediaAssetStore) Put(ctx context.Context, tenantID, sessionID, assetID, mimeType string, data []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	dir := filepath.Join(s.RootDir, tenantID, sessionID, "assets")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("mkdir assets dir: %w", err)
	}

	assetExt := extensionForMimeType(mimeType)
	path := filepath.Join(dir, assetID+assetExt)
	if err := atomicWriteFile(path, data, 0644); err != nil {
		return "", fmt.Errorf("write media asset: %w", err)
	}

	abs, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("absolute path: %w", err)
	}
	return abs, nil
}

// DeleteSessionAssets removes all persisted assets for the provided session.
func (s *FileMediaAssetStore) DeleteSessionAssets(ctx context.Context, tenantID, sessionID string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return os.RemoveAll(filepath.Join(s.RootDir, tenantID, sessionID, "assets"))
}

func extensionForMimeType(mimeType string) string {
	exts, err := mime.ExtensionsByType(mimeType)
	if err != nil || len(exts) == 0 {
		return ""
	}
	return exts[0]
}
