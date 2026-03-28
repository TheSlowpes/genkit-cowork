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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

type fileRecordOperator struct {
	rootDir string
	mu      sync.Mutex
}

// NewFileRecordOperator returns a filesystem-backed FileOperator.
//
// Records are persisted under rootDir/{tenantID}/files.
func NewFileRecordOperator(rootDir string) FileOperator {
	return &fileRecordOperator{rootDir: rootDir}
}

var _ FileOperator = (*fileRecordOperator)(nil)

func (o *fileRecordOperator) SaveFileRecord(ctx context.Context, tenantID string, record FileRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("file record operator: context cancelled: %w", err)
	}
	if tenantID == "" {
		return fmt.Errorf("file record operator: tenantID is required")
	}
	if record.FileID == "" {
		return fmt.Errorf("file record operator: fileID is required")
	}
	if record.TenantID != "" && record.TenantID != tenantID {
		return fmt.Errorf("file record operator: tenant mismatch: got %q want %q", record.TenantID, tenantID)
	}
	record.TenantID = tenantID

	o.mu.Lock()
	defer o.mu.Unlock()

	records, err := o.readRecords(tenantID)
	if err != nil {
		return err
	}
	records[record.FileID] = record
	return o.writeRecords(tenantID, records)
}

func (o *fileRecordOperator) LoadFileRecord(ctx context.Context, tenantID, fileID string) (*FileRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file record operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	records, err := o.readRecords(tenantID)
	if err != nil {
		return nil, err
	}
	record, ok := records[fileID]
	if !ok {
		return nil, nil
	}
	copy := record
	return &copy, nil
}

func (o *fileRecordOperator) ListFiles(ctx context.Context, tenantID string) ([]FileRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file record operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	records, err := o.readRecords(tenantID)
	if err != nil {
		return nil, err
	}

	keys := make([]string, 0, len(records))
	for key := range records {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	out := make([]FileRecord, 0, len(keys))
	for _, key := range keys {
		out = append(out, records[key])
	}
	return out, nil
}

func (o *fileRecordOperator) SaveFileChunks(ctx context.Context, tenantID, fileID string, chunks []FileChunkRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("file record operator: context cancelled: %w", err)
	}
	if tenantID == "" {
		return fmt.Errorf("file record operator: tenantID is required")
	}
	if fileID == "" {
		return fmt.Errorf("file record operator: fileID is required")
	}
	if len(chunks) == 0 {
		return nil
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	existing, err := o.readChunks(tenantID, fileID)
	if err != nil {
		return err
	}

	seen := make(map[string]bool, len(existing))
	for _, chunk := range existing {
		seen[chunk.ChunkID] = true
	}

	for _, chunk := range chunks {
		if chunk.ChunkID == "" {
			return fmt.Errorf("file record operator: chunkID is required")
		}
		if chunk.TenantID != "" && chunk.TenantID != tenantID {
			return fmt.Errorf("file record operator: chunk tenant mismatch: got %q want %q", chunk.TenantID, tenantID)
		}
		if chunk.FileID != "" && chunk.FileID != fileID {
			return fmt.Errorf("file record operator: chunk file mismatch: got %q want %q", chunk.FileID, fileID)
		}
		if seen[chunk.ChunkID] {
			continue
		}
		chunk.TenantID = tenantID
		chunk.FileID = fileID
		existing = append(existing, chunk)
		seen[chunk.ChunkID] = true
	}

	sort.SliceStable(existing, func(i, j int) bool {
		return existing[i].Index < existing[j].Index
	})

	return o.writeChunks(tenantID, fileID, existing)
}

func (o *fileRecordOperator) ListFileChunks(ctx context.Context, tenantID, fileID string) ([]FileChunkRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file record operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	chunks, err := o.readChunks(tenantID, fileID)
	if err != nil {
		return nil, err
	}
	out := make([]FileChunkRecord, len(chunks))
	copy(out, chunks)
	return out, nil
}

func (o *fileRecordOperator) recordsPath(tenantID string) string {
	return filepath.Join(o.rootDir, tenantID, "files", "records.json")
}

func (o *fileRecordOperator) chunksPath(tenantID, fileID string) string {
	return filepath.Join(o.rootDir, tenantID, "files", fileID, "chunks.json")
}

func (o *fileRecordOperator) readRecords(tenantID string) (map[string]FileRecord, error) {
	data, err := os.ReadFile(o.recordsPath(tenantID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return make(map[string]FileRecord), nil
		}
		return nil, fmt.Errorf("read records file: %w", err)
	}

	var records []FileRecord
	if err := json.Unmarshal(data, &records); err != nil {
		return nil, fmt.Errorf("unmarshal records file: %w", err)
	}

	result := make(map[string]FileRecord, len(records))
	for _, record := range records {
		result[record.FileID] = record
	}
	return result, nil
}

func (o *fileRecordOperator) writeRecords(tenantID string, records map[string]FileRecord) error {
	keys := make([]string, 0, len(records))
	for key := range records {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	ordered := make([]FileRecord, 0, len(keys))
	for _, key := range keys {
		ordered = append(ordered, records[key])
	}

	data, err := json.Marshal(ordered)
	if err != nil {
		return fmt.Errorf("marshal records file: %w", err)
	}

	path := o.recordsPath(tenantID)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create records directory: %w", err)
	}
	if err := atomicWriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write records file: %w", err)
	}
	return nil
}

func (o *fileRecordOperator) readChunks(tenantID, fileID string) ([]FileChunkRecord, error) {
	data, err := os.ReadFile(o.chunksPath(tenantID, fileID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []FileChunkRecord{}, nil
		}
		return nil, fmt.Errorf("read chunks file: %w", err)
	}

	var chunks []FileChunkRecord
	if err := json.Unmarshal(data, &chunks); err != nil {
		return nil, fmt.Errorf("unmarshal chunks file: %w", err)
	}
	return chunks, nil
}

func (o *fileRecordOperator) writeChunks(tenantID, fileID string, chunks []FileChunkRecord) error {
	data, err := json.Marshal(chunks)
	if err != nil {
		return fmt.Errorf("marshal chunks file: %w", err)
	}

	path := o.chunksPath(tenantID, fileID)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create chunks directory: %w", err)
	}
	if err := atomicWriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write chunks file: %w", err)
	}
	return nil
}
