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
	"os"
	"path/filepath"
	"strings"
)

// FileBlobStore stores raw tenant-global file bytes.
type FileBlobStore interface {
	PutFile(ctx context.Context, tenantID, fileID, fileName, mimeType string, data []byte) (absolutePath string, err error)
	ReadFile(ctx context.Context, tenantID, fileID, path string) ([]byte, error)
}

// FileBlobDiskStore stores raw file bytes on disk under
// rootDir/{tenantID}/files/raw/{fileID}{ext}.
type FileBlobDiskStore struct {
	RootDir string
}

// NewFileBlobDiskStore creates a filesystem-backed tenant file blob store.
func NewFileBlobDiskStore(rootDir string) *FileBlobDiskStore {
	return &FileBlobDiskStore{RootDir: rootDir}
}

var _ FileBlobStore = (*FileBlobDiskStore)(nil)

func (s *FileBlobDiskStore) PutFile(ctx context.Context, tenantID, fileID, fileName, _ string, data []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if strings.TrimSpace(tenantID) == "" {
		return "", fmt.Errorf("write tenant file blob: tenantID is required")
	}
	if strings.TrimSpace(fileID) == "" {
		return "", fmt.Errorf("write tenant file blob: fileID is required")
	}

	dir := filepath.Join(s.RootDir, tenantID, "files", "raw")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("mkdir raw files dir: %w", err)
	}

	ext := strings.ToLower(filepath.Ext(fileName))
	path := filepath.Join(dir, fileID+ext)
	if err := atomicWriteFile(path, data, 0644); err != nil {
		return "", fmt.Errorf("write tenant file blob: %w", err)
	}

	abs, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("absolute path: %w", err)
	}

	return abs, nil
}

func (s *FileBlobDiskStore) ReadFile(ctx context.Context, tenantID, fileID, path string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(tenantID) == "" {
		return nil, fmt.Errorf("read tenant file blob: tenantID is required")
	}
	if strings.TrimSpace(fileID) == "" {
		return nil, fmt.Errorf("read tenant file blob: fileID is required")
	}

	rootAbs, err := filepath.Abs(s.RootDir)
	if err != nil {
		return nil, fmt.Errorf("read tenant file blob: resolve root path: %w", err)
	}

	tenantRoot := filepath.Join(rootAbs, tenantID)
	pathAbs, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("read tenant file blob: resolve file path: %w", err)
	}

	rel, err := filepath.Rel(tenantRoot, pathAbs)
	if err != nil {
		return nil, fmt.Errorf("read tenant file blob: validate file path: %w", err)
	}
	if rel == "." || rel == "" || strings.HasPrefix(rel, "..") {
		return nil, fmt.Errorf("read tenant file blob: path %q is outside tenant scope", path)
	}

	b, err := os.ReadFile(pathAbs)
	if err != nil {
		return nil, fmt.Errorf("read tenant file blob: %w", err)
	}
	return b, nil
}
