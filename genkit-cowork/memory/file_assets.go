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
	"encoding/json"
	"fmt"
	"mime"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/media"
	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/utils"
	"github.com/firebase/genkit/go/ai"
)

type FileMediaAssetStore struct {
	RootDir   string
	registry  *media.ProcessorRegistry
	assetsMu  sync.Mutex
	loadIndex sync.Mutex
}

// NewFileMediaAssetStore creates a filesystem-backed media asset store.
func NewFileMediaAssetStore(rootDir string) *FileMediaAssetStore {
	return &FileMediaAssetStore{RootDir: rootDir, registry: media.NewProcessorRegistry()}
}

var _ MediaAssetStore = (*FileMediaAssetStore)(nil)

type chunkIndexRecord struct {
	AssetID    string `json:"assetID"`
	ChunkIndex int    `json:"chunkIndex"`
	ChunkCount int    `json:"chunkCount"`
	CharStart  int    `json:"charStart"`
	CharEnd    int    `json:"charEnd"`
	MimeType   string `json:"mimeType,omitempty"`
}

// Put stores raw media bytes under the session assets directory and returns the
// absolute file path.
func (s *FileMediaAssetStore) Put(ctx context.Context, tenantID, sessionID, assetID, mimeType string, data []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := utils.ValidatePathSegment("tenantID", tenantID); err != nil {
		return "", fmt.Errorf("write media asset: %w", err)
	}
	if err := utils.ValidatePathSegment("sessionID", sessionID); err != nil {
		return "", fmt.Errorf("write media asset: %w", err)
	}
	if err := utils.ValidatePathSegment("assetID", assetID); err != nil {
		return "", fmt.Errorf("write media asset: %w", err)
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

	sum := sha256.Sum256(data)
	now := time.Now().UTC()
	record := SessionAsset{
		AssetID:      assetID,
		Name:         filepath.Base(abs),
		MimeType:     mimeType,
		SizeBytes:    len(data),
		SHA256:       hex.EncodeToString(sum[:]),
		Path:         abs,
		UploadedAt:   now,
		IngestStatus: AssetIngestPending,
	}
	if err := s.upsertAssetRecord(tenantID, sessionID, record); err != nil {
		return "", err
	}

	return abs, nil
}

func (s *FileMediaAssetStore) Load(ctx context.Context, tenantID, sessionID, assetID string) ([]ai.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := utils.ValidatePathSegment("tenantID", tenantID); err != nil {
		return nil, fmt.Errorf("load media asset: %w", err)
	}
	if err := utils.ValidatePathSegment("sessionID", sessionID); err != nil {
		return nil, fmt.Errorf("load media asset: %w", err)
	}
	if err := utils.ValidatePathSegment("assetID", assetID); err != nil {
		return nil, fmt.Errorf("load media asset: %w", err)
	}

	assets, err := s.readAssetsIndex(tenantID, sessionID)
	if err != nil {
		return nil, err
	}

	asset, ok := findAssetByID(assets, assetID)
	if !ok {
		return nil, fmt.Errorf("load media asset: asset %q not found", assetID)
	}

	pathAbs, err := filepath.Abs(asset.Path)
	if err != nil {
		return nil, fmt.Errorf("load media asset: resolve asset path: %w", err)
	}
	if err := s.ensureAssetPathInSession(tenantID, sessionID, pathAbs); err != nil {
		return nil, err
	}

	var docsPtr []*ai.Document
	if strings.TrimSpace(asset.MimeType) != "" {
		processor := s.processorRegistry().Get(asset.MimeType)
		if processor == nil {
			return nil, fmt.Errorf("process document: file type not supported")
		}
		data, err := os.ReadFile(pathAbs)
		if err != nil {
			return nil, fmt.Errorf("read file: %w", err)
		}
		docsPtr, err = processor.Process(ctx, data)
		if err != nil {
			return nil, fmt.Errorf("process document: %w", err)
		}
	} else {
		var err error
		docsPtr, err = s.processorRegistry().ProcessDocument(ctx, pathAbs)
		if err != nil {
			return nil, err
		}
	}

	docs := make([]ai.Document, 0, len(docsPtr))
	for _, doc := range docsPtr {
		if doc == nil {
			continue
		}
		if doc.Metadata == nil {
			doc.Metadata = make(map[string]any)
		}
		doc.Metadata["tenantID"] = tenantID
		doc.Metadata["sessionID"] = sessionID
		doc.Metadata["assetID"] = asset.AssetID
		doc.Metadata["assetName"] = asset.Name
		doc.Metadata["assetPath"] = pathAbs
		doc.Metadata["assetSHA256"] = asset.SHA256
		docs = append(docs, *doc)
	}

	if err := s.writeChunkIndex(tenantID, sessionID, assetID, docs); err != nil {
		return nil, err
	}

	if len(docs) > 0 {
		asset.IngestStatus = AssetIngestCompleted
		asset.IngestedAt = time.Now().UTC()
	} else {
		asset.IngestStatus = AssetIngestPending
		asset.IngestedAt = time.Time{}
	}
	if err := s.upsertAssetRecord(tenantID, sessionID, asset); err != nil {
		return nil, err
	}

	return docs, nil
}

func (s *FileMediaAssetStore) ListAssets(ctx context.Context, tenantID, sessionID string) ([]SessionAsset, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := utils.ValidatePathSegment("tenantID", tenantID); err != nil {
		return nil, fmt.Errorf("list media assets: %w", err)
	}
	if err := utils.ValidatePathSegment("sessionID", sessionID); err != nil {
		return nil, fmt.Errorf("list media assets: %w", err)
	}

	return s.readAssetsIndex(tenantID, sessionID)
}

// DeleteSessionAssets removes all persisted assets for the provided session.
func (s *FileMediaAssetStore) DeleteSessionAssets(ctx context.Context, tenantID, sessionID string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := utils.ValidatePathSegment("tenantID", tenantID); err != nil {
		return fmt.Errorf("delete media assets: %w", err)
	}
	if err := utils.ValidatePathSegment("sessionID", sessionID); err != nil {
		return fmt.Errorf("delete media assets: %w", err)
	}
	return os.RemoveAll(filepath.Join(s.RootDir, tenantID, sessionID, "assets"))
}

func (s *FileMediaAssetStore) processorRegistry() *media.ProcessorRegistry {
	if s.registry == nil {
		s.registry = media.NewProcessorRegistry()
	}
	return s.registry
}

func (s *FileMediaAssetStore) assetsDir(tenantID, sessionID string) string {
	return filepath.Join(s.RootDir, tenantID, sessionID, "assets")
}

func (s *FileMediaAssetStore) indexPath(tenantID, sessionID string) string {
	return filepath.Join(s.assetsDir(tenantID, sessionID), "index.json")
}

func (s *FileMediaAssetStore) chunkIndexPath(tenantID, sessionID, assetID string) string {
	return filepath.Join(s.assetsDir(tenantID, sessionID), assetID+".index.json")
}

func (s *FileMediaAssetStore) readAssetsIndex(tenantID, sessionID string) ([]SessionAsset, error) {
	path := s.indexPath(tenantID, sessionID)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return []SessionAsset{}, nil
		}
		return nil, fmt.Errorf("read media assets index: %w", err)
	}

	var assets []SessionAsset
	if err := json.Unmarshal(data, &assets); err != nil {
		return nil, fmt.Errorf("unmarshal media assets index: %w", err)
	}

	sort.SliceStable(assets, func(i, j int) bool {
		if assets[i].UploadedAt.Equal(assets[j].UploadedAt) {
			return assets[i].AssetID < assets[j].AssetID
		}
		return assets[i].UploadedAt.Before(assets[j].UploadedAt)
	})
	return assets, nil
}

func (s *FileMediaAssetStore) writeAssetsIndex(tenantID, sessionID string, assets []SessionAsset) error {
	sort.SliceStable(assets, func(i, j int) bool {
		if assets[i].UploadedAt.Equal(assets[j].UploadedAt) {
			return assets[i].AssetID < assets[j].AssetID
		}
		return assets[i].UploadedAt.Before(assets[j].UploadedAt)
	})

	data, err := json.Marshal(assets)
	if err != nil {
		return fmt.Errorf("marshal media assets index: %w", err)
	}

	if err := os.MkdirAll(s.assetsDir(tenantID, sessionID), 0755); err != nil {
		return fmt.Errorf("mkdir assets dir: %w", err)
	}

	if err := atomicWriteFile(s.indexPath(tenantID, sessionID), data, 0644); err != nil {
		return fmt.Errorf("write media assets index: %w", err)
	}
	return nil
}

func (s *FileMediaAssetStore) upsertAssetRecord(tenantID, sessionID string, record SessionAsset) error {
	s.assetsMu.Lock()
	defer s.assetsMu.Unlock()

	assets, err := s.readAssetsIndex(tenantID, sessionID)
	if err != nil {
		return err
	}

	updated := upsertAssetByID(assets, record)
	if err := s.writeAssetsIndex(tenantID, sessionID, updated); err != nil {
		return err
	}
	return nil
}

func (s *FileMediaAssetStore) writeChunkIndex(tenantID, sessionID, assetID string, docs []ai.Document) error {
	s.loadIndex.Lock()
	defer s.loadIndex.Unlock()

	records := make([]chunkIndexRecord, 0, len(docs))
	for i, doc := range docs {
		record := chunkIndexRecord{AssetID: assetID, ChunkIndex: i, ChunkCount: len(docs)}
		if doc.Metadata != nil {
			record.CharStart = intFromMetadata(doc.Metadata, "charStart")
			record.CharEnd = intFromMetadata(doc.Metadata, "charEnd")
			record.MimeType = stringFromMetadata(doc.Metadata, "mimeType")
		}
		records = append(records, record)
	}

	data, err := json.Marshal(records)
	if err != nil {
		return fmt.Errorf("marshal chunk index: %w", err)
	}

	if err := atomicWriteFile(s.chunkIndexPath(tenantID, sessionID, assetID), data, 0644); err != nil {
		return fmt.Errorf("write chunk index: %w", err)
	}
	return nil
}

func (s *FileMediaAssetStore) ensureAssetPathInSession(tenantID, sessionID, pathAbs string) error {
	rootAbs, err := filepath.Abs(s.assetsDir(tenantID, sessionID))
	if err != nil {
		return fmt.Errorf("load media asset: resolve assets root: %w", err)
	}
	rel, err := filepath.Rel(rootAbs, pathAbs)
	if err != nil {
		return fmt.Errorf("load media asset: validate path: %w", err)
	}
	if rel == "." || rel == "" || strings.HasPrefix(rel, "..") {
		return fmt.Errorf("load media asset: path %q is outside session assets", pathAbs)
	}
	return nil
}

func findAssetByID(assets []SessionAsset, assetID string) (SessionAsset, bool) {
	for _, asset := range assets {
		if asset.AssetID == assetID {
			return asset, true
		}
	}
	return SessionAsset{}, false
}

func upsertAssetByID(assets []SessionAsset, asset SessionAsset) []SessionAsset {
	for i := range assets {
		if assets[i].AssetID == asset.AssetID {
			assets[i] = asset
			return assets
		}
	}
	return append(assets, asset)
}

func intFromMetadata(metadata map[string]any, key string) int {
	value, ok := metadata[key]
	if !ok {
		return 0
	}
	switch v := value.(type) {
	case int:
		return v
	case int32:
		return int(v)
	case int64:
		return int(v)
	case float32:
		return int(v)
	case float64:
		return int(v)
	default:
		return 0
	}
}

func stringFromMetadata(metadata map[string]any, key string) string {
	value, ok := metadata[key]
	if !ok {
		return ""
	}
	v, ok := value.(string)
	if !ok {
		return ""
	}
	return v
}

func extensionForMimeType(mimeType string) string {
	exts, err := mime.ExtensionsByType(mimeType)
	if err != nil || len(exts) == 0 {
		return ""
	}
	return exts[0]
}
