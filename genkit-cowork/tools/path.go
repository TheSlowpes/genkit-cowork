package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// resolveReadPath resolves a file path to an absolute path relative to cwd.
// It handles ~ expansion (Unix) and resolves relative paths against the
// provided working directory. Returns an error if the resolved path does
// not exist.
func resolveReadPath(path, cwd string) (string, error) {
	expanded := expandPath(path)

	var absolutePath string
	if filepath.IsAbs(expanded) {
		absolutePath = filepath.Clean(expanded)
	} else {
		absolutePath = filepath.Clean(filepath.Join(cwd, expanded))
	}

	if _, err := os.Stat(absolutePath); err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("file not found: %s", absolutePath)
		}
		return "", fmt.Errorf("cannot access file: %w", err)
	}

	return absolutePath, nil
}

// expandPath expands ~ prefixes to the user's home directory.
// On Windows, ~ expansion is supported but less common; the function
// handles it uniformly across platforms using os.UserHomeDir.
func expandPath(path string) string {
	if path == "~" {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return home
	}

	// Check for ~/ or ~\ (the latter for Windows).
	tildeSlash := "~" + string(filepath.Separator)
	if strings.HasPrefix(path, tildeSlash) {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return filepath.Join(home, path[len(tildeSlash):])
	}

	// Also handle ~/ on Windows where the separator is \.
	if runtime.GOOS == "windows" && strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return filepath.Join(home, path[2:])
	}

	return path
}
