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
	"fmt"
	"maps"
	"slices"
	"sort"
	"sync"
	"time"
)

// FileIngestStatus tracks the state of a tenant-global file ingestion.
type FileIngestStatus string

const (
	// FileIngestPending indicates the file is persisted but not yet fully ingested.
	FileIngestPending FileIngestStatus = "pending"
	// FileIngestCompleted indicates parse/chunk/index completed successfully.
	FileIngestCompleted FileIngestStatus = "completed"
	// FileIngestFailed indicates ingestion failed and Error contains details.
	FileIngestFailed FileIngestStatus = "failed"
)

// FileRecord stores immutable tenant-owned file metadata.
type FileRecord struct {
	FileID         string           `json:"fileID"`
	TenantID       string           `json:"tenantID"`
	SessionID      string           `json:"sessionID,omitempty"`
	SourceChannel  MessageOrigin    `json:"sourceChannel,omitempty"`
	Name           string           `json:"name"`
	MimeType       string           `json:"mimeType"`
	SizeBytes      int              `json:"sizeBytes"`
	SHA256         string           `json:"sha256"`
	StoragePath    string           `json:"storagePath"`
	UploadedAt     time.Time        `json:"uploadedAt"`
	IngestStatus   FileIngestStatus `json:"ingestStatus"`
	IngestedAt     time.Time        `json:"ingestedAt,omitempty"`
	Error          string           `json:"error,omitempty"`
	ExtractedTitle string           `json:"extractedTitle,omitempty"`
	MetadataJSON   string           `json:"metadataJSON,omitempty"`
}

// FileChunkRecord stores chunk-level metadata for a file memory entry.
type FileChunkRecord struct {
	ChunkID        string        `json:"chunkID"`
	FileID         string        `json:"fileID"`
	TenantID       string        `json:"tenantID"`
	SessionID      string        `json:"sessionID,omitempty"`
	SourceChannel  MessageOrigin `json:"sourceChannel,omitempty"`
	Index          int           `json:"index"`
	Text           string        `json:"text"`
	CharStart      int           `json:"charStart"`
	CharEnd        int           `json:"charEnd"`
	TokenEstimate  int           `json:"tokenEstimate"`
	VectorDocID    string        `json:"vectorDocID,omitempty"`
	UploadedAt     time.Time     `json:"uploadedAt"`
	FileName       string        `json:"fileName"`
	FileMimeType   string        `json:"fileMimeType"`
	FileSHA256     string        `json:"fileSHA256"`
	ExtractionMode string        `json:"extractionMode"`
}

// FileChunkSearchResult is a retrieval row returned by tenant/global file
// recall APIs.
type FileChunkSearchResult struct {
	Chunk FileChunkRecord `json:"chunk"`
	File  FileRecord      `json:"file"`
}

// FileOperator defines tenant-global file metadata persistence.
type FileOperator interface {
	SaveFileRecord(ctx context.Context, tenantID string, record FileRecord) error
	LoadFileRecord(ctx context.Context, tenantID, fileID string) (*FileRecord, error)
	ListFiles(ctx context.Context, tenantID string) ([]FileRecord, error)
	SaveFileChunks(ctx context.Context, tenantID, fileID string, chunks []FileChunkRecord) error
	ListFileChunks(ctx context.Context, tenantID, fileID string) ([]FileChunkRecord, error)
}

// DefaultFileOperator is the in-memory implementation used for tests and
// lightweight single-process usage.
type DefaultFileOperator struct {
	mu      sync.Mutex
	files   map[string]map[string]FileRecord
	chunks  map[string]map[string][]FileChunkRecord
	chunkIx map[string]map[string]map[string]bool
}

// NewDefaultFileOperator returns an in-memory file operator.
func NewDefaultFileOperator() *DefaultFileOperator {
	return &DefaultFileOperator{
		files:   make(map[string]map[string]FileRecord),
		chunks:  make(map[string]map[string][]FileChunkRecord),
		chunkIx: make(map[string]map[string]map[string]bool),
	}
}

var _ FileOperator = (*DefaultFileOperator)(nil)

// SaveFileRecord inserts or updates one tenant-owned file record.
func (o *DefaultFileOperator) SaveFileRecord(ctx context.Context, tenantID string, record FileRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("file operator: context cancelled: %w", err)
	}
	if tenantID == "" {
		return fmt.Errorf("file operator: tenantID is required")
	}
	if record.FileID == "" {
		return fmt.Errorf("file operator: fileID is required")
	}
	if record.TenantID != "" && record.TenantID != tenantID {
		return fmt.Errorf("file operator: tenant mismatch: got %q want %q", record.TenantID, tenantID)
	}
	record.TenantID = tenantID

	o.mu.Lock()
	defer o.mu.Unlock()

	if _, ok := o.files[tenantID]; !ok {
		o.files[tenantID] = make(map[string]FileRecord)
	}
	o.files[tenantID][record.FileID] = copyFileRecord(record)
	return nil
}

// LoadFileRecord returns one tenant-scoped file record.
func (o *DefaultFileOperator) LoadFileRecord(ctx context.Context, tenantID, fileID string) (*FileRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	tenantFiles, ok := o.files[tenantID]
	if !ok {
		return nil, nil
	}
	record, ok := tenantFiles[fileID]
	if !ok {
		return nil, nil
	}
	copy := copyFileRecord(record)
	return &copy, nil
}

// ListFiles returns all tenant file records sorted by upload time.
func (o *DefaultFileOperator) ListFiles(ctx context.Context, tenantID string) ([]FileRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	tenantFiles, ok := o.files[tenantID]
	if !ok {
		return []FileRecord{}, nil
	}

	keys := slices.Collect(maps.Keys(tenantFiles))
	sort.Strings(keys)
	out := make([]FileRecord, 0, len(keys))
	for _, key := range keys {
		out = append(out, copyFileRecord(tenantFiles[key]))
	}
	return out, nil
}

// SaveFileChunks appends new chunks for one tenant file.
func (o *DefaultFileOperator) SaveFileChunks(ctx context.Context, tenantID, fileID string, chunks []FileChunkRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("file operator: context cancelled: %w", err)
	}
	if len(chunks) == 0 {
		return nil
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	if _, ok := o.chunks[tenantID]; !ok {
		o.chunks[tenantID] = make(map[string][]FileChunkRecord)
	}
	if _, ok := o.chunkIx[tenantID]; !ok {
		o.chunkIx[tenantID] = make(map[string]map[string]bool)
	}
	if _, ok := o.chunkIx[tenantID][fileID]; !ok {
		o.chunkIx[tenantID][fileID] = make(map[string]bool)
	}

	for _, chunk := range chunks {
		if chunk.ChunkID == "" {
			return fmt.Errorf("file operator: chunkID is required")
		}
		if chunk.FileID != "" && chunk.FileID != fileID {
			return fmt.Errorf("file operator: chunk file mismatch: got %q want %q", chunk.FileID, fileID)
		}
		if chunk.TenantID != "" && chunk.TenantID != tenantID {
			return fmt.Errorf("file operator: chunk tenant mismatch: got %q want %q", chunk.TenantID, tenantID)
		}
		if o.chunkIx[tenantID][fileID][chunk.ChunkID] {
			continue
		}
		chunk.TenantID = tenantID
		chunk.FileID = fileID
		o.chunks[tenantID][fileID] = append(o.chunks[tenantID][fileID], copyFileChunk(chunk))
		o.chunkIx[tenantID][fileID][chunk.ChunkID] = true
	}

	sort.SliceStable(o.chunks[tenantID][fileID], func(i, j int) bool {
		return o.chunks[tenantID][fileID][i].Index < o.chunks[tenantID][fileID][j].Index
	})

	return nil
}

// ListFileChunks returns all chunks for one tenant file.
func (o *DefaultFileOperator) ListFileChunks(ctx context.Context, tenantID, fileID string) ([]FileChunkRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("file operator: context cancelled: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	tenantChunks, ok := o.chunks[tenantID]
	if !ok {
		return []FileChunkRecord{}, nil
	}
	chunks := tenantChunks[fileID]
	out := make([]FileChunkRecord, len(chunks))
	for i := range chunks {
		out[i] = copyFileChunk(chunks[i])
	}
	return out, nil
}

func copyFileRecord(in FileRecord) FileRecord {
	if in.MetadataJSON == "" {
		in.MetadataJSON = "{}"
	}
	return in
}

func copyFileChunk(in FileChunkRecord) FileChunkRecord {
	return in
}

// MetadataJSON stores optional provider-specific metadata as JSON string.
//
// This field lets callers attach additional provenance without broadening the
// strongly-typed core schema for every scenario.
func (r *FileRecord) SetMetadata(metadata map[string]string) error {
	if metadata == nil {
		r.MetadataJSON = "{}"
		return nil
	}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("marshal file metadata: %w", err)
	}
	r.MetadataJSON = string(encoded)
	return nil
}

// Metadata decodes the optional metadata payload.
func (r FileRecord) Metadata() (map[string]string, error) {
	if r.MetadataJSON == "" {
		return map[string]string{}, nil
	}
	var decoded map[string]string
	if err := json.Unmarshal([]byte(r.MetadataJSON), &decoded); err != nil {
		return nil, fmt.Errorf("unmarshal file metadata: %w", err)
	}
	if decoded == nil {
		return map[string]string{}, nil
	}
	return decoded, nil
}
