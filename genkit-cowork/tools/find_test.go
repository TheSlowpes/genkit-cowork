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

package tools

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
)

type findOpMock struct {
	exists  bool
	matches []string
	pattern string
	path    string
	limit   int
	ignore  []string
}

func (m *findOpMock) Exists(ctx context.Context, path string) (bool, error) {
	m.path = path
	return m.exists, nil
}

func (m *findOpMock) IgnorePatterns(ctx context.Context, path string) ([]string, error) {
	return m.ignore, nil
}

func (m *findOpMock) Glob(ctx context.Context, pattern string, path string, limit int, ignore []string) ([]string, error) {
	m.pattern = pattern
	m.path = path
	m.limit = limit
	m.ignore = ignore
	return m.matches, nil
}

func TestHandleFind_UsesOperatorAndFormatsRelativePaths(t *testing.T) {
	root := t.TempDir()
	m := &findOpMock{
		exists: true,
		matches: []string{
			filepath.Join(root, "b.txt"),
			filepath.Join(root, "dir", "a.txt"),
		},
	}

	out, err := handleFind(context.Background(), FindToolInput{Pattern: "**/*.txt", Limit: 10}, root, &findToolOptions{operator: m})
	if err != nil {
		t.Fatalf("handleFind() error = %v", err)
	}

	want := []string{"b.txt", "dir/a.txt"}
	if diff := cmp.Diff(want, out.Matches); diff != "" {
		t.Fatalf("handleFind() matches mismatch (-want +got):\n%s", diff)
	}
	if m.pattern != "**/*.txt" {
		t.Fatalf("operator pattern = %q, want %q", m.pattern, "**/*.txt")
	}
	if m.limit != 11 {
		t.Fatalf("operator limit = %d, want 11", m.limit)
	}
}

func TestHandleFind_DefaultsPatternAndLimit(t *testing.T) {
	root := t.TempDir()
	m := &findOpMock{exists: true}

	out, err := handleFind(context.Background(), FindToolInput{}, root, &findToolOptions{operator: m})
	if err != nil {
		t.Fatalf("handleFind() error = %v", err)
	}
	if out.Pattern != "**/*" {
		t.Fatalf("pattern = %q, want %q", out.Pattern, "**/*")
	}
	if m.limit != DefaultMaxFiles+1 {
		t.Fatalf("operator limit = %d, want %d", m.limit, DefaultMaxFiles+1)
	}
}

func TestHandleFind_TruncatesResults(t *testing.T) {
	root := t.TempDir()
	m := &findOpMock{
		exists: true,
		matches: []string{
			filepath.Join(root, "a.txt"),
			filepath.Join(root, "b.txt"),
			filepath.Join(root, "c.txt"),
		},
	}

	out, err := handleFind(context.Background(), FindToolInput{Limit: 2}, root, &findToolOptions{operator: m})
	if err != nil {
		t.Fatalf("handleFind() error = %v", err)
	}
	if !out.Truncated || out.TruncatedBy != "results" {
		t.Fatalf("truncation = (%v, %q), want (true, results)", out.Truncated, out.TruncatedBy)
	}
	if out.Count != 2 {
		t.Fatalf("count = %d, want 2", out.Count)
	}
}

func TestGlobFiles_RecursiveMatchAndGitignore(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "main.go"), "package main")
	writeTestFile(t, filepath.Join(root, "internal", "keep.go"), "package internal")
	writeTestFile(t, filepath.Join(root, "vendor", "skip.go"), "package vendor")
	writeTestFile(t, filepath.Join(root, "notes.txt"), "notes")
	writeTestFile(t, filepath.Join(root, ".git", "config"), "ignored")

	matches, err := globFiles(context.Background(), "**/*.go", root, 0, []string{"vendor/"})
	if err != nil {
		t.Fatalf("globFiles() error = %v", err)
	}

	got := formatFindMatches(matches, root)
	want := []string{"internal/keep.go", "main.go"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("globFiles() mismatch (-want +got):\n%s", diff)
	}
}

func TestHandleFind_PassesOperatorIgnorePatterns(t *testing.T) {
	m := &findOpMock{exists: true, ignore: []string{"dist/", "*.tmp"}}

	_, err := handleFind(context.Background(), FindToolInput{}, t.TempDir(), &findToolOptions{operator: m})
	if err != nil {
		t.Fatalf("handleFind() error = %v", err)
	}
	want := []string{"dist/", "*.tmp"}
	if diff := cmp.Diff(want, m.ignore); diff != "" {
		t.Fatalf("ignore mismatch (-want +got):\n%s", diff)
	}
}

func TestDefaultFindOperator_ReadsGitignore(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, ".gitignore"), "dist/\n*.tmp\n")

	got, err := (&defaultFindOperator{}).IgnorePatterns(context.Background(), root)
	if err != nil {
		t.Fatalf("IgnorePatterns() error = %v", err)
	}
	want := []string{"dist/", "*.tmp"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("IgnorePatterns() mismatch (-want +got):\n%s", diff)
	}
}

func TestHandleFind_MissingDirectory(t *testing.T) {
	m := &findOpMock{exists: false}
	_, err := handleFind(context.Background(), FindToolInput{}, t.TempDir(), &findToolOptions{operator: m})
	if err == nil {
		t.Fatal("handleFind() error = nil, want non-nil")
	}
}

func writeTestFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
}
