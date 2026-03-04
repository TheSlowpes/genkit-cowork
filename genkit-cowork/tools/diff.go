package tools

import "strings"

type ChangeType int

const (
	Equal  ChangeType = 0
	Insert ChangeType = 1
	Delete ChangeType = -1
)

type Change struct {
	Type  ChangeType
	Value string
	Lines []string
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
