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
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestResolveWorkspaceLayout_Default(t *testing.T) {
	tenantID := "tenant-1"

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir() error = %v", err)
	}

	t.Setenv(GenkitHomeEnv, "")
	t.Setenv(GenkitWorkspaceEnv, "")

	got, err := ResolveWorkspaceLayout(tenantID)
	if err != nil {
		t.Fatalf("ResolveWorkspaceLayout() error = %v", err)
	}

	wantHome := filepath.Join(home, ".genkit")
	if got.HomeDir != wantHome {
		t.Errorf("HomeDir = %q, want %q", got.HomeDir, wantHome)
	}
	if got.WorkspaceRootDir != filepath.Join(wantHome, "workspace") {
		t.Errorf("WorkspaceRootDir = %q, want %q", got.WorkspaceRootDir, filepath.Join(wantHome, "workspace"))
	}
	if got.WorkspaceTenantDir != filepath.Join(wantHome, "workspace", tenantID) {
		t.Errorf("WorkspaceTenantDir = %q, want %q", got.WorkspaceTenantDir, filepath.Join(wantHome, "workspace", tenantID))
	}
	if got.TracesDir != filepath.Join(wantHome, "traces") {
		t.Errorf("TracesDir = %q, want %q", got.TracesDir, filepath.Join(wantHome, "traces"))
	}
	if got.MemorySessionsDir != filepath.Join(wantHome, "memory", "sessions") {
		t.Errorf("MemorySessionsDir = %q, want %q", got.MemorySessionsDir, filepath.Join(wantHome, "memory", "sessions"))
	}
	if got.MemoryIndexDir != filepath.Join(wantHome, "memory", "index") {
		t.Errorf("MemoryIndexDir = %q, want %q", got.MemoryIndexDir, filepath.Join(wantHome, "memory", "index"))
	}
	if got.MemoryAssetsDir != filepath.Join(wantHome, "memory", "assets") {
		t.Errorf("MemoryAssetsDir = %q, want %q", got.MemoryAssetsDir, filepath.Join(wantHome, "memory", "assets"))
	}
	if got.SkillsDir != filepath.Join(wantHome, "skills") {
		t.Errorf("SkillsDir = %q, want %q", got.SkillsDir, filepath.Join(wantHome, "skills"))
	}
}

func TestResolveWorkspaceLayout_EnvOverrides(t *testing.T) {
	tenantID := "tenant-2"
	home := filepath.Join(t.TempDir(), "custom-home")
	workspace := filepath.Join(t.TempDir(), "custom-workspace")

	t.Setenv(GenkitHomeEnv, home)
	t.Setenv(GenkitWorkspaceEnv, workspace)

	got, err := ResolveWorkspaceLayout(tenantID)
	if err != nil {
		t.Fatalf("ResolveWorkspaceLayout() error = %v", err)
	}

	if got.HomeDir != home {
		t.Errorf("HomeDir = %q, want %q", got.HomeDir, home)
	}
	if got.WorkspaceRootDir != workspace {
		t.Errorf("WorkspaceRootDir = %q, want %q", got.WorkspaceRootDir, workspace)
	}
	if got.WorkspaceTenantDir != filepath.Join(workspace, tenantID) {
		t.Errorf("WorkspaceTenantDir = %q, want %q", got.WorkspaceTenantDir, filepath.Join(workspace, tenantID))
	}
	if got.TracesDir != filepath.Join(home, "traces") {
		t.Errorf("TracesDir = %q, want %q", got.TracesDir, filepath.Join(home, "traces"))
	}
}

func TestResolveWorkspaceLayout_InvalidTenantID(t *testing.T) {
	_, err := ResolveWorkspaceLayout("../tenant")
	if err == nil {
		t.Fatal("ResolveWorkspaceLayout() error = nil, want non-nil")
	}
}

func TestEnsureWorkspaceLayout(t *testing.T) {
	base := t.TempDir()
	layout := &WorkspaceLayout{
		HomeDir:            filepath.Join(base, ".genkit"),
		WorkspaceRootDir:   filepath.Join(base, ".genkit", "workspace"),
		WorkspaceTenantDir: filepath.Join(base, ".genkit", "workspace", "tenant-1"),
		TracesDir:          filepath.Join(base, ".genkit", "traces"),
		MemoryRootDir:      filepath.Join(base, ".genkit", "memory"),
		MemorySessionsDir:  filepath.Join(base, ".genkit", "memory", "sessions"),
		MemoryIndexDir:     filepath.Join(base, ".genkit", "memory", "index"),
		MemoryAssetsDir:    filepath.Join(base, ".genkit", "memory", "assets"),
		SkillsDir:          filepath.Join(base, ".genkit", "skills"),
	}

	if err := EnsureWorkspaceLayout(context.Background(), layout); err != nil {
		t.Fatalf("EnsureWorkspaceLayout() error = %v", err)
	}

	paths := []string{
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
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			t.Fatalf("Stat(%q) error = %v", p, err)
		}
		if !info.IsDir() {
			t.Fatalf("path %q is not a directory", p)
		}
	}
}

func TestEnsureSessionWorkspaceLinks(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink behavior varies on windows without elevated privileges")
	}

	base := t.TempDir()
	layout := &WorkspaceLayout{
		HomeDir:            filepath.Join(base, ".genkit"),
		WorkspaceRootDir:   filepath.Join(base, ".genkit", "workspace"),
		WorkspaceTenantDir: filepath.Join(base, ".genkit", "workspace", "tenant-1"),
		TracesDir:          filepath.Join(base, ".genkit", "traces"),
		MemoryRootDir:      filepath.Join(base, ".genkit", "memory"),
		MemorySessionsDir:  filepath.Join(base, ".genkit", "memory", "sessions"),
		MemoryIndexDir:     filepath.Join(base, ".genkit", "memory", "index"),
		MemoryAssetsDir:    filepath.Join(base, ".genkit", "memory", "assets"),
		SkillsDir:          filepath.Join(base, ".genkit", "skills"),
	}
	if err := EnsureWorkspaceLayout(context.Background(), layout); err != nil {
		t.Fatalf("EnsureWorkspaceLayout() error = %v", err)
	}

	if err := EnsureSessionWorkspaceLinks(context.Background(), layout, "tenant-1", "session-1"); err != nil {
		t.Fatalf("EnsureSessionWorkspaceLinks() error = %v", err)
	}

	workspaceSessionDir := filepath.Join(layout.WorkspaceTenantDir, "sessions", "session-1")
	memoryLink := filepath.Join(workspaceSessionDir, "memory")
	indexLink := filepath.Join(workspaceSessionDir, "index")

	if _, err := os.Lstat(memoryLink); err != nil {
		t.Fatalf("Lstat(memory link) error = %v", err)
	}
	if _, err := os.Lstat(indexLink); err != nil {
		t.Fatalf("Lstat(index link) error = %v", err)
	}

	memoryResolved, err := filepath.EvalSymlinks(memoryLink)
	if err != nil {
		t.Fatalf("EvalSymlinks(memory) error = %v", err)
	}
	wantMemoryResolved, err := filepath.EvalSymlinks(filepath.Join(layout.MemorySessionsDir, "tenant-1", "session-1"))
	if err != nil {
		t.Fatalf("EvalSymlinks(want memory dir) error = %v", err)
	}
	if memoryResolved != wantMemoryResolved {
		t.Errorf("memory symlink target = %q, want %q", memoryResolved, wantMemoryResolved)
	}

	indexResolved, err := filepath.EvalSymlinks(indexLink)
	if err != nil {
		t.Fatalf("EvalSymlinks(index) error = %v", err)
	}
	wantIndexResolved, err := filepath.EvalSymlinks(filepath.Join(layout.MemoryIndexDir, "tenant-1", "session-1"))
	if err != nil {
		t.Fatalf("EvalSymlinks(want index dir) error = %v", err)
	}
	if indexResolved != wantIndexResolved {
		t.Errorf("index symlink target = %q, want %q", indexResolved, wantIndexResolved)
	}
}

func TestEnsureSessionWorkspaceLinks_ReplaceStaleSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink behavior varies on windows without elevated privileges")
	}

	base := t.TempDir()
	layout := &WorkspaceLayout{
		HomeDir:            filepath.Join(base, ".genkit"),
		WorkspaceRootDir:   filepath.Join(base, ".genkit", "workspace"),
		WorkspaceTenantDir: filepath.Join(base, ".genkit", "workspace", "tenant-1"),
		TracesDir:          filepath.Join(base, ".genkit", "traces"),
		MemoryRootDir:      filepath.Join(base, ".genkit", "memory"),
		MemorySessionsDir:  filepath.Join(base, ".genkit", "memory", "sessions"),
		MemoryIndexDir:     filepath.Join(base, ".genkit", "memory", "index"),
		MemoryAssetsDir:    filepath.Join(base, ".genkit", "memory", "assets"),
		SkillsDir:          filepath.Join(base, ".genkit", "skills"),
	}
	if err := EnsureWorkspaceLayout(context.Background(), layout); err != nil {
		t.Fatalf("EnsureWorkspaceLayout() error = %v", err)
	}

	workspaceSessionDir := filepath.Join(layout.WorkspaceTenantDir, "sessions", "session-1")
	if err := os.MkdirAll(workspaceSessionDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) error = %v", workspaceSessionDir, err)
	}
	staleTarget := filepath.Join(base, "stale")
	if err := os.MkdirAll(staleTarget, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) error = %v", staleTarget, err)
	}
	if err := os.Symlink(staleTarget, filepath.Join(workspaceSessionDir, "memory")); err != nil {
		t.Fatalf("Symlink(stale) error = %v", err)
	}

	if err := EnsureSessionWorkspaceLinks(context.Background(), layout, "tenant-1", "session-1"); err != nil {
		t.Fatalf("EnsureSessionWorkspaceLinks() error = %v", err)
	}

	resolved, err := filepath.EvalSymlinks(filepath.Join(workspaceSessionDir, "memory"))
	if err != nil {
		t.Fatalf("EvalSymlinks(memory) error = %v", err)
	}
	want, err := filepath.EvalSymlinks(filepath.Join(layout.MemorySessionsDir, "tenant-1", "session-1"))
	if err != nil {
		t.Fatalf("EvalSymlinks(want memory dir) error = %v", err)
	}
	if resolved != want {
		t.Errorf("memory symlink target = %q, want %q", resolved, want)
	}
}
