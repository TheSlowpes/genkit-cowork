// Copyright 2025 Google LLC
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

import "strings"

// ChangeType describes the kind of diff change.
type ChangeType int

const (
	// Equal indicates unchanged lines between old and new content.
	Equal ChangeType = 0
	// Insert indicates lines inserted in the new content.
	Insert ChangeType = 1
	// Delete indicates lines removed from the old content.
	Delete ChangeType = -1
)

// Change represents a grouped diff segment.
type Change struct {
	// Type is the diff operation for this change.
	Type ChangeType
	// Value is the raw concatenated text for this change.
	Value string
	// Lines contains line-split data for this change when populated.
	Lines []string
	// Count is the number of merged primitive changes.
	Count int
}

// tokenize splits a string into lines, each including its trailing newline
func tokenize(value string, stripTrailingCr bool) []string {
	if stripTrailingCr {
		value = normalizeToLF(value)
	}

	parts := strings.SplitAfter(value, "\n")

	if len(parts) > 0 && parts[len(parts)-1] == "" {
		parts = parts[:len(parts)-1]
	}

	return parts
}

// diffLines computes a line-level diff using a simple LCS-based approach
func diffLines(oldContent, newContent string) []Change {
	oldLines := tokenize(oldContent, true)
	newLines := tokenize(newContent, true)

	m, n := len(oldLines), len(newLines)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if oldLines[i-1] == newLines[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else if dp[i-1][j] >= dp[i][j-1] {
				dp[i][j] = dp[i-1][j]
			} else {
				dp[i][j] = dp[i][j-1]
			}
		}
	}

	var changes []Change
	i, j := m, n
	for i > 0 || j > 0 {
		if i > 0 && j > 0 && oldLines[i-1] == newLines[j-1] {
			changes = append([]Change{{Type: Equal, Value: oldLines[i-1]}}, changes...)
			i--
			j--
		} else if j > 0 && (i == 0 || dp[i][j-1] >= dp[i-1][j]) {
			changes = append([]Change{{Type: Insert, Value: newLines[j-1]}}, changes...)
			j--
		} else {
			changes = append([]Change{{Type: Delete, Value: oldLines[i-1]}}, changes...)
			i--
		}
	}

	var merged []Change
	for _, change := range changes {
		if len(merged) > 0 && merged[len(merged)-1].Type == change.Type {
			last := &merged[len(merged)-1]
			last.Lines = append(last.Lines, change.Lines...)
			last.Value += change.Value
			last.Count++
		} else {
			merged = append(merged, change)
		}
	}

	return merged
}
