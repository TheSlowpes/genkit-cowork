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
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/media"
	"github.com/firebase/genkit/go/ai"
	"github.com/google/uuid"
)

const (
	defaultChunkSize    = 1200
	defaultChunkOverlap = 200
)

// FileIngestInput is the request shape for tenant-global file ingestion.
type FileIngestInput struct {
	TenantID      string
	SessionID     string
	SourceChannel MessageOrigin
	FileName      string
	MimeType      string
	Data          []byte
}

// FileIngestOutput contains the canonical file record and persisted chunks.
type FileIngestOutput struct {
	File   FileRecord
	Chunks []FileChunkRecord
}

// FileChunkSearchInput is the query shape for tenant file recall.
type FileChunkSearchInput struct {
	TenantID string
	Query    string
	TopK     int
}

// FileMemoryIndexer indexes file chunks for semantic retrieval.
type FileMemoryIndexer interface {
	IndexFileChunks(ctx context.Context, tenantID, fileID string, chunks []FileChunkRecord) error
	SearchFileChunks(ctx context.Context, tenantID, query string, topK int) ([]FileChunkRecord, error)
}

// FileIngestService stores tenant-global files, extracts text from supported
// text/structured MIME types, chunks content, and indexes chunks.
type FileIngestService struct {
	files       FileOperator
	blobs       FileBlobStore
	extractor   media.TextExtractor
	indexer     FileMemoryIndexer
	chunkSize   int
	overlapSize int
}

// NewFileIngestService constructs a file ingestion service.
func NewFileIngestService(files FileOperator, blobs FileBlobStore, extractor media.TextExtractor, indexer FileMemoryIndexer) *FileIngestService {
	if files == nil {
		files = NewDefaultFileOperator()
	}
	if extractor == nil {
		extractor = media.NewDefaultTextExtractor()
	}

	return &FileIngestService{
		files:       files,
		blobs:       blobs,
		extractor:   extractor,
		indexer:     indexer,
		chunkSize:   defaultChunkSize,
		overlapSize: defaultChunkOverlap,
	}
}

// Ingest stores and indexes one tenant-global file record.
func (s *FileIngestService) Ingest(ctx context.Context, input FileIngestInput) (*FileIngestOutput, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(input.TenantID) == "" {
		return nil, fmt.Errorf("ingest file: tenantID is required")
	}
	if strings.TrimSpace(input.FileName) == "" {
		return nil, fmt.Errorf("ingest file: fileName is required")
	}
	if len(input.Data) == 0 {
		return nil, fmt.Errorf("ingest file: data is required")
	}
	if s.blobs == nil {
		return nil, fmt.Errorf("ingest file: blob store is not configured")
	}

	now := time.Now().UTC()
	fileID := uuid.New().String()
	sha := sha256.Sum256(input.Data)
	checksum := hex.EncodeToString(sha[:])

	mimeType := strings.TrimSpace(input.MimeType)
	if mimeType == "" {
		mimeType = media.DetectMimeType(input.Data, input.FileName)
	}
	if mimeType == "" {
		return nil, fmt.Errorf("ingest file: unsupported mime type")
	}

	storagePath, err := s.blobs.PutFile(ctx, input.TenantID, fileID, input.FileName, mimeType, input.Data)
	if err != nil {
		return nil, fmt.Errorf("ingest file: store blob: %w", err)
	}

	record := FileRecord{
		FileID:        fileID,
		TenantID:      input.TenantID,
		SessionID:     input.SessionID,
		SourceChannel: input.SourceChannel,
		Name:          filepath.Base(input.FileName),
		MimeType:      mimeType,
		SizeBytes:     len(input.Data),
		SHA256:        checksum,
		StoragePath:   storagePath,
		UploadedAt:    now,
		IngestStatus:  FileIngestPending,
	}

	if err := s.files.SaveFileRecord(ctx, input.TenantID, record); err != nil {
		return nil, fmt.Errorf("ingest file: save record: %w", err)
	}

	extracted, err := s.extractor.Extract(ctx, media.TextExtractInput{
		FileName: input.FileName,
		MimeType: mimeType,
		Data:     input.Data,
	})
	if err != nil {
		record.IngestStatus = FileIngestFailed
		record.Error = err.Error()
		record.IngestedAt = now
		_ = s.files.SaveFileRecord(ctx, input.TenantID, record)
		return nil, fmt.Errorf("ingest file: extract text: %w", err)
	}

	record.ExtractedTitle = extracted.Title

	chunks := buildFileChunks(fileChunkBuildInput{
		TenantID:      input.TenantID,
		SessionID:     input.SessionID,
		SourceChannel: input.SourceChannel,
		FileID:        fileID,
		FileName:      record.Name,
		MimeType:      mimeType,
		Checksum:      checksum,
		UploadedAt:    now,
		Text:          extracted.Text,
		ChunkSize:     s.chunkSize,
		OverlapSize:   s.overlapSize,
	})

	if err := s.files.SaveFileChunks(ctx, input.TenantID, fileID, chunks); err != nil {
		record.IngestStatus = FileIngestFailed
		record.Error = err.Error()
		record.IngestedAt = now
		_ = s.files.SaveFileRecord(ctx, input.TenantID, record)
		return nil, fmt.Errorf("ingest file: save chunks: %w", err)
	}

	if s.indexer != nil && len(chunks) > 0 {
		if err := s.indexer.IndexFileChunks(ctx, input.TenantID, fileID, chunks); err != nil {
			record.IngestStatus = FileIngestFailed
			record.Error = err.Error()
			record.IngestedAt = now
			_ = s.files.SaveFileRecord(ctx, input.TenantID, record)
			return nil, fmt.Errorf("ingest file: index chunks: %w", err)
		}
	}

	record.IngestStatus = FileIngestCompleted
	record.Error = ""
	record.IngestedAt = now
	if err := s.files.SaveFileRecord(ctx, input.TenantID, record); err != nil {
		return nil, fmt.Errorf("ingest file: finalize record: %w", err)
	}

	return &FileIngestOutput{File: record, Chunks: chunks}, nil
}

// SearchTenantFiles returns tenant-scoped file chunk recall results.
func (s *FileIngestService) SearchTenantFiles(ctx context.Context, input FileChunkSearchInput) ([]FileChunkSearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(input.TenantID) == "" {
		return nil, fmt.Errorf("search tenant files: tenantID is required")
	}
	if strings.TrimSpace(input.Query) == "" {
		return nil, fmt.Errorf("search tenant files: query is required")
	}
	if s.indexer == nil {
		return nil, fmt.Errorf("search tenant files: file indexer is not configured")
	}

	topK := input.TopK
	if topK <= 0 {
		topK = 5
	}

	chunks, err := s.indexer.SearchFileChunks(ctx, input.TenantID, input.Query, topK)
	if err != nil {
		return nil, fmt.Errorf("search tenant files: %w", err)
	}

	results := make([]FileChunkSearchResult, 0, len(chunks))
	for _, chunk := range chunks {
		record, err := s.files.LoadFileRecord(ctx, input.TenantID, chunk.FileID)
		if err != nil {
			return nil, fmt.Errorf("search tenant files: load file record: %w", err)
		}
		if record == nil {
			continue
		}
		results = append(results, FileChunkSearchResult{Chunk: chunk, File: *record})
	}
	return results, nil
}

type fileChunkBuildInput struct {
	TenantID      string
	SessionID     string
	SourceChannel MessageOrigin
	FileID        string
	FileName      string
	MimeType      string
	Checksum      string
	UploadedAt    time.Time
	Text          string
	ChunkSize     int
	OverlapSize   int
}

func buildFileChunks(input fileChunkBuildInput) []FileChunkRecord {
	text := strings.TrimSpace(input.Text)
	if text == "" {
		return nil
	}

	chunkSize := input.ChunkSize
	if chunkSize <= 0 {
		chunkSize = defaultChunkSize
	}
	overlap := input.OverlapSize
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= chunkSize {
		overlap = chunkSize / 4
	}

	step := chunkSize - overlap
	if step <= 0 {
		step = chunkSize
	}

	runes := []rune(text)
	out := make([]FileChunkRecord, 0)
	for start, idx := 0, 0; start < len(runes); start, idx = start+step, idx+1 {
		end := start + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunkText := strings.TrimSpace(string(runes[start:end]))
		if chunkText == "" {
			continue
		}
		chunkID := fmt.Sprintf("%s:%d", input.FileID, idx)
		out = append(out, FileChunkRecord{
			ChunkID:        chunkID,
			FileID:         input.FileID,
			TenantID:       input.TenantID,
			SessionID:      input.SessionID,
			SourceChannel:  input.SourceChannel,
			Index:          idx,
			Text:           chunkText,
			CharStart:      start,
			CharEnd:        end,
			TokenEstimate:  estimateTextTokens(chunkText),
			UploadedAt:     input.UploadedAt,
			FileName:       input.FileName,
			FileMimeType:   input.MimeType,
			FileSHA256:     input.Checksum,
			ExtractionMode: "text-structured-v1",
		})
		if end >= len(runes) {
			break
		}
	}
	return out
}

func estimateTextTokens(text string) int {
	fields := strings.Fields(text)
	if len(fields) > 0 {
		return len(fields)
	}
	if text == "" {
		return 0
	}
	return 1
}

type vectorFileIndexer struct {
	backend VectorBackend
}

// NewVectorFileIndexer adapts a VectorBackend for file chunk indexing/search.
func NewVectorFileIndexer(backend VectorBackend) FileMemoryIndexer {
	if backend == nil {
		return nil
	}
	return &vectorFileIndexer{backend: backend}
}

func (i *vectorFileIndexer) IndexFileChunks(ctx context.Context, tenantID, fileID string, chunks []FileChunkRecord) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(chunks) == 0 {
		return nil
	}

	docs := make([]*ai.Document, 0, len(chunks))
	for _, chunk := range chunks {
		docs = append(docs, ai.DocumentFromText(chunk.Text, map[string]any{
			"tenantID":       tenantID,
			"fileID":         fileID,
			"chunkID":        chunk.ChunkID,
			"sessionID":      chunk.SessionID,
			"origin":         string(chunk.SourceChannel),
			"uploadedAt":     chunk.UploadedAt.Format(time.RFC3339Nano),
			"mimeType":       chunk.FileMimeType,
			"recordType":     "file_chunk",
			"fileName":       chunk.FileName,
			"extractionMode": chunk.ExtractionMode,
		}))
	}

	return i.backend.Index(ctx, tenantID, fileID, docs)
}

func (i *vectorFileIndexer) SearchFileChunks(ctx context.Context, tenantID, query string, topK int) ([]FileChunkRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	docs, err := i.backend.RetrieveTenant(ctx, tenantID, query, topK)
	if err != nil {
		return nil, err
	}

	out := make([]FileChunkRecord, 0, len(docs))
	for _, doc := range docs {
		recordType, _ := doc.Metadata["recordType"].(string)
		if recordType != "file_chunk" {
			continue
		}
		chunkID, _ := doc.Metadata["chunkID"].(string)
		if chunkID == "" {
			continue
		}
		fileID, _ := doc.Metadata["fileID"].(string)
		sessionID, _ := doc.Metadata["sessionID"].(string)
		origin, _ := doc.Metadata["origin"].(string)
		mimeType, _ := doc.Metadata["mimeType"].(string)
		fileName, _ := doc.Metadata["fileName"].(string)
		extractionMode, _ := doc.Metadata["extractionMode"].(string)
		uploadedAtRaw, _ := doc.Metadata["uploadedAt"].(string)
		uploadedAt, _ := time.Parse(time.RFC3339Nano, uploadedAtRaw)

		text := documentText(doc)

		out = append(out, FileChunkRecord{
			ChunkID:        chunkID,
			FileID:         fileID,
			TenantID:       tenantID,
			SessionID:      sessionID,
			SourceChannel:  MessageOrigin(origin),
			Text:           text,
			TokenEstimate:  estimateTextTokens(text),
			UploadedAt:     uploadedAt,
			FileName:       fileName,
			FileMimeType:   mimeType,
			ExtractionMode: extractionMode,
		})
		if len(out) >= topK {
			break
		}
	}

	return out, nil
}

func documentText(doc *ai.Document) string {
	if doc == nil {
		return ""
	}
	parts := make([]string, 0, len(doc.Content))
	for _, part := range doc.Content {
		if part != nil && part.IsText() {
			parts = append(parts, part.Text)
		}
	}
	return strings.Join(parts, " ")
}
