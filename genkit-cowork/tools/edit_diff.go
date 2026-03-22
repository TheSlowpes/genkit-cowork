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

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// FuzzyMatchResult describes the outcome of fuzzy text matching.
type FuzzyMatchResult struct {
	// Found indicates whether a match was found.
	Found bool
	// Index is the byte offset where the match starts.
	Index int
	// MatchLength is the byte length of the matched text.
	MatchLength int
	// UsedFuzzyMatch indicates whether matching required fuzzy normalization.
	UsedFuzzyMatch bool
	// ContentForReplacement is the content used for replacement operations.
	// For exact matches this is the original content; for fuzzy matches this is
	// the normalized content.
	ContentForReplacement string
}

func detectLineEnding(content string) string {
	crlfIdx := strings.Index(content, "\r\n")
	lfIdx := strings.Index(content, "\n")
	if crlfIdx == -1 || lfIdx == -1 {
		return "\n"
	}
	if crlfIdx < lfIdx {
		return "\r\n"
	}
	return "\n"
}

func normalizeToLF(content string) string {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	content = strings.ReplaceAll(content, "\r", "\n")
	return content
}

func restoreLineEndings(content, lineEnding string) string {
	if lineEnding == "\r\n" {
		return strings.ReplaceAll(content, "\n", "\r\n")
	}
	return content
}

func normalizeForFuzzyMatch(content string) string {
	var b strings.Builder
	b.Grow(len(content))
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		// Trim trailing whitespace but preserve leading whitespace for indentation.
		trimmedLine := strings.TrimRightFunc(line, unicode.IsSpace)
		for _, r := range trimmedLine {
			switch r {
			// Smart single quotes
			case '\u2018', '\u2019', '\u201A', '\u201B':
				b.WriteRune('\'')
			// Smart double quotes
			case '\u201C', '\u201D', '\u201E', '\u201F':
				b.WriteRune('"')
			// Various dashes
			case '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212':
				b.WriteRune('-')
			// Special spaces
			case '\u00A0', '\u202F', '\u205F', '\u3000':
				b.WriteRune(' ')
			default:
				// Handle range U+2002 to U+200A
				if r >= '\u2002' && r <= '\u200A' {
					b.WriteRune(' ')
				} else {
					b.WriteRune(r)
				}
			}
		}

		// Restore line break after processing each line.
		if i < len(lines)-1 {
			b.WriteByte('\n')
		}
	}

	return b.String()
}

func fuzzyFindText(content, oldText string) FuzzyMatchResult {
	exactIdx := strings.Index(content, oldText)
	if exactIdx != -1 {
		return FuzzyMatchResult{
			Found:                 true,
			Index:                 exactIdx,
			MatchLength:           len(oldText),
			UsedFuzzyMatch:        false,
			ContentForReplacement: content,
		}
	}

	fuzzyContent := normalizeForFuzzyMatch(content)
	fuzzyOldText := normalizeForFuzzyMatch(oldText)
	fuzzyIdx := strings.Index(fuzzyContent, fuzzyOldText)

	if fuzzyIdx == -1 {
		return FuzzyMatchResult{
			Found:                 false,
			Index:                 -1,
			MatchLength:           0,
			UsedFuzzyMatch:        false,
			ContentForReplacement: content,
		}
	}

	return FuzzyMatchResult{
		Found:                 true,
		Index:                 fuzzyIdx,
		MatchLength:           len(fuzzyOldText),
		UsedFuzzyMatch:        true,
		ContentForReplacement: fuzzyContent,
	}
}

func stripBom(content string) (string, string) {
	if strings.HasPrefix(content, "\uFEFF") {
		return "\uFEFF", content[1:]
	}
	return "", content
}

// Generate a unified diff string with line numbers and context.
// Returns both the diff string and the first changed line number (in the new file).
func generateDiffString(oldContent, newContent string, contextLines *int) (string, *int) {
	parts := diffLines(oldContent, newContent)
	output := make([]string, 0, len(parts))

	oldLines := strings.Split(oldContent, "\n")
	newLines := strings.Split(newContent, "\n")
	maxLineNum := max(len(oldLines), len(newLines))
	lineNumWidth := len(strconv.Itoa(maxLineNum))

	oldline, newline := 1, 1
	lastWasChange := false
	var firstChangeLine *int

	for i, part := range parts {
		raw := strings.Split(part.Value, "\n")
		if raw[len(raw)-1] == "" {
			raw = raw[:len(raw)-1]
		}

		if part.Type == Insert || part.Type == Delete {
			if firstChangeLine == nil {
				line := newline
				firstChangeLine = &line
			}

			for _, line := range raw {
				if part.Type == Insert {
					lineNumStr := fmt.Sprintf("%s%s", strings.Repeat(" ", lineNumWidth-len(strconv.Itoa(newline))), strconv.Itoa(newline))
					output = append(output, fmt.Sprintf("+ %s | %s", lineNumStr, line))
					newline++
				} else {
					lineNumStr := fmt.Sprintf("%s%s", strings.Repeat(" ", lineNumWidth-len(strconv.Itoa(oldline))), strconv.Itoa(oldline))
					output = append(output, fmt.Sprintf("- %s | %s", lineNumStr, line))
					oldline++
				}
			}
			lastWasChange = true
		} else {
			nextPartIsChange := i < len(parts)-1 && (parts[i+1].Type == Insert || parts[i+1].Type == Delete)

			if lastWasChange || nextPartIsChange {
				linesToShow := raw
				skipStart := 0
				skipEnd := 0

				if !lastWasChange {
					if contextLines == nil {
						contextLines = new(int)
					}
					skipStart = max(0, len(raw)-*contextLines)
					linesToShow = raw[skipStart:]
				}

				if !nextPartIsChange && len(linesToShow) > *contextLines {
					skipEnd = len(linesToShow) - *contextLines
					linesToShow = linesToShow[:len(linesToShow)-skipEnd]
				}

				if skipStart > 0 {
					output = append(output, fmt.Sprintf("  %s | ... lines skipped ...", strings.Repeat(" ", lineNumWidth)))
					oldline += skipStart
					newline += skipStart
				}

				for _, line := range linesToShow {
					lineNumStr := fmt.Sprintf("%s%s", strings.Repeat(" ", lineNumWidth-len(strconv.Itoa(newline))), strconv.Itoa(newline))
					output = append(output, fmt.Sprintf("  %s | %s", lineNumStr, line))
					oldline++
					newline++
				}

				if skipEnd > 0 {
					output = append(output, fmt.Sprintf("  %s | ... lines skipped ...", strings.Repeat(" ", lineNumWidth)))
					oldline += skipEnd
					newline += skipEnd
				}
			} else {
				oldline += len(raw)
				newline += len(raw)
			}

			lastWasChange = false
		}
	}

	return strings.Join(output, "\n"), firstChangeLine
}
