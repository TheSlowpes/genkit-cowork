// Copyright 2025 Kevin Lopes
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

// resolveToCwd resolves a path relative to the given cwd.
// Handles ~ expansion and absolute paths.
func resolveToCwd(filePath, cwd string) string {
	expanded := expandPath(filePath)
	if filepath.IsAbs(expanded) {
		return filepath.Clean(expanded)
	}

	return filepath.Clean(filepath.Join(cwd, expanded))
}
