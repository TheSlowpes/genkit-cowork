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

package utils

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const (
	// GenkitHomeEnv overrides the default ~/.genkit home directory.
	GenkitHomeEnv = "GENKIT_HOME"
	// GenkitWorkspaceEnv overrides the default workspace root directory.
	GenkitWorkspaceEnv = "GENKIT_WORKSPACE"
)

// WorkspaceLayout describes the default local filesystem layout used by
// genkit-cowork for agent workspaces and persisted data.
type WorkspaceLayout struct {
	// HomeDir is the base .genkit directory.
	HomeDir string
	// WorkspaceRootDir is the shared workspace root for all tenants.
	WorkspaceRootDir string
	// WorkspaceTenantDir is the tenant-scoped workspace used as tool cwd.
	WorkspaceTenantDir string
	// TracesDir stores Genkit traces.
	TracesDir string
	// MemoryRootDir is the parent for all memory persistence directories.
	MemoryRootDir string
	// MemorySessionsDir stores file-backed session state.
	MemorySessionsDir string
	// MemoryIndexDir stores vector index persistence.
	MemoryIndexDir string
	// MemoryAssetsDir stores normalized media assets.
	MemoryAssetsDir string
	// SkillsDir stores globally available skills.
	SkillsDir string
}

// ResolveWorkspaceLayout returns the default .genkit directory layout for a
// tenant.
//
// Precedence:
//  1. GENKIT_HOME and GENKIT_WORKSPACE environment variables.
//  2. ~/.genkit fallback when variables are unset.
func ResolveWorkspaceLayout(tenantID string) (*WorkspaceLayout, error) {
	if err := ValidatePathSegment("tenantID", tenantID); err != nil {
		return nil, err
	}

	homeDir, err := resolveGenkitHomeDir()
	if err != nil {
		return nil, err
	}

	workspaceRootDir := os.Getenv(GenkitWorkspaceEnv)
	if workspaceRootDir == "" {
		workspaceRootDir = filepath.Join(homeDir, "workspace")
	}
	workspaceRootDir = filepath.Clean(workspaceRootDir)

	memoryRootDir := filepath.Join(homeDir, "memory")

	return &WorkspaceLayout{
		HomeDir:            homeDir,
		WorkspaceRootDir:   workspaceRootDir,
		WorkspaceTenantDir: filepath.Join(workspaceRootDir, tenantID),
		TracesDir:          filepath.Join(homeDir, "traces"),
		MemoryRootDir:      memoryRootDir,
		MemorySessionsDir:  filepath.Join(memoryRootDir, "sessions"),
		MemoryIndexDir:     filepath.Join(memoryRootDir, "index"),
		MemoryAssetsDir:    filepath.Join(memoryRootDir, "assets"),
		SkillsDir:          filepath.Join(homeDir, "skills"),
	}, nil
}

// EnsureWorkspaceLayout creates the standard .genkit directories on disk.
func EnsureWorkspaceLayout(ctx context.Context, layout *WorkspaceLayout) error {
	if layout == nil {
		return fmt.Errorf("workspace layout is required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	dirs := []string{
		layout.HomeDir,
		layout.WorkspaceRootDir,
		layout.WorkspaceTenantDir,
		layout.TracesDir,
		layout.MemoryRootDir,
		layout.MemorySessionsDir,
		layout.MemoryIndexDir,
		layout.MemoryAssetsDir,
		layout.SkillsDir,
	}

	for _, dir := range dirs {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create directory %q: %w", dir, err)
		}
	}

	return nil
}

// EnsureSessionWorkspaceLinks creates convenience symlinks inside the
// tenant-scoped workspace for a specific session.
//
// The links are created under:
//   - <workspace>/<tenant>/sessions/<session>/memory -> <home>/memory/sessions/<tenant>/<session>
//   - <workspace>/<tenant>/sessions/<session>/index  -> <home>/memory/index/<tenant>/<session>
func EnsureSessionWorkspaceLinks(ctx context.Context, layout *WorkspaceLayout, tenantID, sessionID string) error {
	if layout == nil {
		return fmt.Errorf("workspace layout is required")
	}
	if err := ValidatePathSegment("tenantID", tenantID); err != nil {
		return err
	}
	if err := ValidatePathSegment("sessionID", sessionID); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	workspaceSessionDir := filepath.Join(layout.WorkspaceTenantDir, "sessions", sessionID)
	targetSessionDir := filepath.Join(layout.MemorySessionsDir, tenantID, sessionID)
	targetIndexDir := filepath.Join(layout.MemoryIndexDir, tenantID, sessionID)

	if err := os.MkdirAll(workspaceSessionDir, 0o755); err != nil {
		return fmt.Errorf("create workspace session dir %q: %w", workspaceSessionDir, err)
	}
	if err := os.MkdirAll(targetSessionDir, 0o755); err != nil {
		return fmt.Errorf("create memory session dir %q: %w", targetSessionDir, err)
	}
	if err := os.MkdirAll(targetIndexDir, 0o755); err != nil {
		return fmt.Errorf("create memory index dir %q: %w", targetIndexDir, err)
	}

	memoryLinkPath := filepath.Join(workspaceSessionDir, "memory")
	if err := ensureSymlinkToTarget(memoryLinkPath, targetSessionDir); err != nil {
		return fmt.Errorf("ensure memory symlink: %w", err)
	}

	indexLinkPath := filepath.Join(workspaceSessionDir, "index")
	if err := ensureSymlinkToTarget(indexLinkPath, targetIndexDir); err != nil {
		return fmt.Errorf("ensure index symlink: %w", err)
	}

	return nil
}

func ensureSymlinkToTarget(linkPath, targetPath string) error {
	if info, err := os.Lstat(linkPath); err == nil {
		if info.Mode()&os.ModeSymlink == 0 {
			return fmt.Errorf("path exists and is not a symlink: %s", linkPath)
		}
		resolvedTarget, err := os.Readlink(linkPath)
		if err != nil {
			return fmt.Errorf("read existing symlink %q: %w", linkPath, err)
		}

		linkDir := filepath.Dir(linkPath)
		resolvedAbs := resolvedTarget
		if !filepath.IsAbs(resolvedAbs) {
			resolvedAbs = filepath.Join(linkDir, resolvedAbs)
		}
		if filepath.Clean(resolvedAbs) == filepath.Clean(targetPath) {
			return nil
		}

		if err := os.Remove(linkPath); err != nil {
			return fmt.Errorf("remove stale symlink %q: %w", linkPath, err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("stat symlink path %q: %w", linkPath, err)
	}

	relTarget, err := filepath.Rel(filepath.Dir(linkPath), targetPath)
	if err != nil {
		relTarget = targetPath
	}
	if err := os.Symlink(relTarget, linkPath); err != nil {
		return fmt.Errorf("create symlink %q -> %q: %w", linkPath, relTarget, err)
	}
	return nil
}

func resolveGenkitHomeDir() (string, error) {
	home := os.Getenv(GenkitHomeEnv)
	if home != "" {
		return filepath.Clean(home), nil
	}

	userHomeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve user home dir: %w", err)
	}

	return filepath.Join(userHomeDir, ".genkit"), nil
}
