package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func resolveReadPath(path string) (string, error) {
	absolutePath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("failed to resolve absolute path: %w", err)
	}
	if absolutePath == "~" {
		absolutePath, err = os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to resolve home directory: %w", err)
		}
	} else if strings.HasPrefix(absolutePath, "~/") {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to resolve home directory: %w", err)
		}
		absolutePath = filepath.Join(homeDir, absolutePath[2:])
	}
	return absolutePath, nil
}
