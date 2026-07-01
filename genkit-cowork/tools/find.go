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
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// FindToolInput is the schema for the find tool's input parameters.
type FindToolInput struct {
	Pattern string `json:"pattern,omitempty" jsonschema_description:"Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts' (default: '**/*')"`
	Path    string `json:"path,omitempty" jsonschema_description:"Directory to search in (default: current directory)"`
	Limit   int    `json:"limit,omitempty" jsonschema_description:"Maximum number of results (default: 1000)"`
}

// FindToolOutput is the structured result returned by the find tool.
type FindToolOutput struct {
	Matches     []string `json:"matches"`
	Count       int      `json:"count"`
	SearchPath  string   `json:"searchPath"`
	Pattern     string   `json:"pattern"`
	Truncated   bool     `json:"truncated,omitempty"`
	TruncatedBy string   `json:"truncatedBy,omitempty"`
}

// FindOperator abstracts the find operation so implementations can be
// swapped out for testing or to delegate file search to remote systems (for example SSH).
type FindOperator interface {
	// Exists checks if a directory exists at the given path.
	Exists(ctx context.Context, path string) (bool, error)
	// IgnorePatterns reads ignore patterns for the search path.
	IgnorePatterns(ctx context.Context, path string) ([]string, error)
	// Glob finds files matching glob pattern. Returns relative or absolute paths.
	Glob(ctx context.Context, pattern string, path string, limit int, ignore []string) ([]string, error)
}

type findToolOptions struct {
	operator FindOperator
}

// FindToolOption configures a FindTool via functional options.
type FindToolOption func(*findToolOptions)

// WithCustomFindOperator injects a custom FindOperator implementation.
func WithCustomFindOperator(operator FindOperator) FindToolOption {
	return func(opts *findToolOptions) {
		opts.operator = operator
	}
}

// defaultFindOperator is the standard filesystem-backed implementation.
type defaultFindOperator struct{}

func (o *defaultFindOperator) Exists(ctx context.Context, path string) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	return info.IsDir(), nil
}

func (o *defaultFindOperator) Glob(ctx context.Context, pattern string, path string, limit int, ignore []string) ([]string, error) {
	return globFiles(ctx, pattern, path, limit, ignore)
}

func (o *defaultFindOperator) IgnorePatterns(ctx context.Context, path string) ([]string, error) {
	return readGitignore(ctx, path)
}

// NewFindTool creates a Genkit tool that finds files in a directory using glob patterns.
// The cwd parameter sets the working directory for resolving relative paths.
func NewFindTool(g *genkit.Genkit, cwd string, opts ...FindToolOption) ai.Tool {
	options := findToolOptions{
		operator: &defaultFindOperator{},
	}
	for _, opt := range opts {
		opt(&options)
	}

	description := fmt.Sprintf(
		"Find files by glob pattern. Supports ** for recursive matches and returns paths relative to the search directory. "+
			"Respects simple .gitignore patterns. Output is truncated to %d results or %s (whichever is hit first).",
		DefaultMaxFiles, FormatSize(DefaultMaxBytes),
	)

	return genkit.DefineTool(
		g,
		"find",
		description,
		func(ctx *ai.ToolContext, input FindToolInput) (FindToolOutput, error) {
			return handleFind(ctx, input, cwd, &options)
		},
	)
}

func handleFind(ctx context.Context, input FindToolInput, cwd string, options *findToolOptions) (FindToolOutput, error) {
	if err := ctx.Err(); err != nil {
		return FindToolOutput{}, fmt.Errorf("operation cancelled")
	}

	if options == nil || options.operator == nil {
		return FindToolOutput{}, fmt.Errorf("find operator is not configured")
	}

	pattern := strings.TrimSpace(input.Pattern)
	if pattern == "" {
		pattern = "**/*"
	}
	pattern = filepath.ToSlash(filepath.Clean(pattern))
	if pattern == "." {
		pattern = "**/*"
	}

	searchPath := input.Path
	if strings.TrimSpace(searchPath) == "" {
		searchPath = "."
	}
	absolutePath := resolveToCwd(searchPath, cwd)

	exists, err := options.operator.Exists(ctx, absolutePath)
	if err != nil {
		return FindToolOutput{}, fmt.Errorf("cannot access search directory: %w", err)
	}
	if !exists {
		return FindToolOutput{}, fmt.Errorf("search directory not found: %s", absolutePath)
	}

	limit := input.Limit
	if limit <= 0 {
		limit = DefaultMaxFiles
	}

	ignore, err := options.operator.IgnorePatterns(ctx, absolutePath)
	if err != nil {
		return FindToolOutput{}, err
	}
	matches, err := options.operator.Glob(ctx, pattern, absolutePath, limit+1, ignore)
	if err != nil {
		return FindToolOutput{}, err
	}
	if err := ctx.Err(); err != nil {
		return FindToolOutput{}, fmt.Errorf("operation cancelled")
	}

	formatted := formatFindMatches(matches, absolutePath)
	truncatedBy := ""
	truncated := false
	if len(formatted) > limit {
		formatted = formatted[:limit]
		truncated = true
		truncatedBy = "results"
	}

	content := strings.Join(formatted, "\n")
	for len(content) > DefaultMaxBytes && len(formatted) > 0 {
		formatted = formatted[:len(formatted)-1]
		content = strings.Join(formatted, "\n")
		truncated = true
		truncatedBy = "bytes"
	}

	return FindToolOutput{
		Matches:     formatted,
		Count:       len(formatted),
		SearchPath:  absolutePath,
		Pattern:     pattern,
		Truncated:   truncated,
		TruncatedBy: truncatedBy,
	}, nil
}

func globFiles(ctx context.Context, pattern, root string, limit int, ignore []string) ([]string, error) {
	matcher := newGitignoreMatcher(ignore)
	matches := make([]string, 0)

	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}

		if path == root {
			return nil
		}

		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)

		if matcher.ignored(rel, entry.IsDir()) {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if entry.IsDir() {
			return nil
		}

		ok, err := matchGlobPattern(pattern, rel)
		if err != nil {
			return err
		}
		if ok {
			matches = append(matches, path)
		}
		if limit > 0 && len(matches) >= limit {
			return errFindLimitReached
		}
		return nil
	})
	if err != nil && err != errFindLimitReached {
		return nil, err
	}

	sort.Strings(matches)
	return matches, nil
}

var errFindLimitReached = fmt.Errorf("find limit reached")

func matchGlobPattern(pattern, rel string) (bool, error) {
	pattern = filepath.ToSlash(filepath.Clean(pattern))
	if pattern == "." {
		pattern = "**/*"
	}

	base, found := strings.CutPrefix(pattern, "**/")
	if found {
		if ok, err := filepath.Match(base, filepath.Base(rel)); err != nil || ok {
			return ok, err
		}
	}
	if base, ok := strings.CutPrefix(pattern, "**/"); ok {
		if ok, err := filepath.Match(base, filepath.Base(rel)); err != nil || ok {
			return ok, err
		}
	}

	return matchGlobParts(strings.Split(pattern, "/"), strings.Split(rel, "/"))
}

func matchGlobParts(patternParts, pathParts []string) (bool, error) {
	if len(patternParts) == 0 {
		return len(pathParts) == 0, nil
	}

	part := patternParts[0]
	if part == "**" {
		if ok, err := matchGlobParts(patternParts[1:], pathParts); err != nil || ok {
			return ok, err
		}
		for i := range pathParts {
			ok, err := matchGlobParts(patternParts[1:], pathParts[i+1:])
			if err != nil || ok {
				return ok, err
			}
		}
		return false, nil
	}

	if len(pathParts) == 0 {
		return false, nil
	}
	matched, err := filepath.Match(part, pathParts[0])
	if err != nil || !matched {
		return false, err
	}
	return matchGlobParts(patternParts[1:], pathParts[1:])
}

func formatFindMatches(matches []string, root string) []string {
	formatted := make([]string, 0, len(matches))
	for _, match := range matches {
		rel, err := filepath.Rel(root, match)
		if err == nil && !strings.HasPrefix(rel, "..") {
			match = rel
		}
		formatted = append(formatted, filepath.ToSlash(match))
	}
	sort.Strings(formatted)
	return formatted
}

func readGitignore(ctx context.Context, root string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("operation cancelled")
	}
	buf, err := os.ReadFile(filepath.Join(root, ".gitignore"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read .gitignore: %w", err)
	}

	lines := splitLines(string(buf))
	patterns := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "!") {
			continue
		}
		patterns = append(patterns, line)
	}
	return patterns, nil
}

type gitignoreMatcher struct {
	patterns []string
}

func newGitignoreMatcher(patterns []string) gitignoreMatcher {
	normalized := make([]string, 0, len(patterns)+1)
	normalized = append(normalized, ".git/")
	for _, pattern := range patterns {
		pattern = filepath.ToSlash(strings.TrimSpace(pattern))
		if pattern != "" {
			normalized = append(normalized, pattern)
		}
	}
	return gitignoreMatcher{patterns: normalized}
}

func (m gitignoreMatcher) ignored(rel string, isDir bool) bool {
	rel = filepath.ToSlash(rel)
	for _, pattern := range m.patterns {
		if gitignorePatternMatches(pattern, rel, isDir) {
			return true
		}
	}
	return false
}

func gitignorePatternMatches(pattern, rel string, isDir bool) bool {
	pattern = strings.TrimPrefix(pattern, "/")
	directoryOnly := strings.HasSuffix(pattern, "/")
	pattern = strings.TrimSuffix(pattern, "/")
	if pattern == "" || (directoryOnly && !isDir && !strings.HasPrefix(rel, pattern+"/")) {
		return false
	}

	if directoryOnly && (rel == pattern || strings.HasPrefix(rel, pattern+"/")) {
		return true
	}

	if strings.Contains(pattern, "/") {
		ok, err := matchGlobPattern(pattern, rel)
		return err == nil && ok
	}

	for part := range strings.SplitSeq(rel, "/") {
		ok, err := filepath.Match(pattern, part)
		if err == nil && ok {
			return true
		}
	}
	return false
}
