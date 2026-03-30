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
	"time"

	"github.com/firebase/genkit/go/ai"
)

// SessionAsset records a stored media asset associated with a message part.
type SessionAsset struct {
	AssetID      string            `json:"assetID"`
	MessageID    string            `json:"messageID"`
	Name         string            `json:"name"`
	PartIndex    int               `json:"partIndex"`
	MimeType     string            `json:"mimeType"`
	SizeBytes    int               `json:"sizeBytes"`
	SHA256       string            `json:"sha256"`
	Path         string            `json:"path"`
	UploadedAt   time.Time         `json:"uploadedAt"`
	IngestStatus AssetIngestStatus `json:"ingestStatus"`
	IngestedAt   time.Time         `json:"ingestedAt"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// MediaAssetStore persists and deletes media assets referenced by session
// messages.
type MediaAssetStore interface {
	Put(ctx context.Context, tenantID, sessionID, assetID, mimeType string, data []byte) (absolutePath string, err error)
	Load(ctx context.Context, tenantID, sessionID, assetID string) (docs []ai.Document, err error)
	ListAssets(ctx context.Context, tenantID, sessionID string) (assets []SessionAsset, err error)
	DeleteSessionAssets(ctx context.Context, tenantID, sessionID string) error
}

type AssetIngestStatus string

const (
	// AssetIngestPending indicates the asset is persisted but not yet fully ingested.
	AssetIngestPending AssetIngestStatus = "pending"
	// AssetIngestCompleted indicates parse/chunk/index completed successfully.
	AssetIngestCompleted AssetIngestStatus = "completed"
	// AssetIngestFailed indicates ingestion failed and Error contains details.
	AssetIngestFailed AssetIngestStatus = "failed"
)
