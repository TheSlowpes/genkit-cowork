package memory

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

type FileMediaAssetStore struct {
	RootDir string
}

func NewFileMediaAssetStore(rootDir string) *FileMediaAssetStore {
	return &FileMediaAssetStore{RootDir: rootDir}
}

func (s *FileMediaAssetStore) Put(ctx context.Context, sessionID, assetID, mimeType string, data []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	dir := filepath.Join(s.RootDir, sessionID, "assets")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("mkdir assets dir: %w", err)
	}

	path := filepath.Join(dir, assetID+filepath.Ext(mimeType))
	if err := atomicWriteFile(path, data, 0644); err != nil {
		return "", fmt.Errorf("write media asset: %w", err)
	}

	abs, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("absolute path: %w", err)
	}
	return abs, nil
}

func (s *FileMediaAssetStore) DeleteSessionAssets(ctx context.Context, sessionID string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return os.RemoveAll(filepath.Join(s.RootDir, sessionID, "assets"))
}
