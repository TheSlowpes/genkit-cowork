package memory

import "context"

type SessionAsset struct {
	AssetID   string `json:"assetID"`
	MessageID string `json:"messageID"`
	PartIndex int    `json:"partIndex"`
	MimeType  string `json:"mimeType"`
	SizeBytes int    `json:"sizeBytes"`
	Path      string `json:"path"`
}

type MediaAssetStore interface {
	Put(ctx context.Context, sessionID, assetID, mimeType string, data []byte) (absolutePath string, err error)
	DeleteSessionAssets(ctx context.Context, sessionID string) error
}
