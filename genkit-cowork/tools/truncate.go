package tools

import "fmt"

// TruncationResult holds the outcome of a truncation operation.
type TruncationResult struct {
	// Content is the truncated (or original) content.
	Content string
	// Truncated indicates whether truncation occurred.
	Truncated bool
	// TruncatedBy is "lines", "bytes", or "" if not truncated.
	TruncatedBy string
	// TotalLines is the number of lines in the original content.
	TotalLines int
	// TotalBytes is the byte length of the original content.
	TotalBytes int
	// OutputLines is the number of complete lines in the output.
	OutputLines int
	// OutputBytes is the byte length of the output.
	OutputBytes int
	// FirstLineExceedsLimit is true when the first line alone exceeds the byte limit.
	// Used by TruncateHead.
	FirstLineExceedsLimit bool
	// LastLinePartial is true when the last line was partially truncated.
	// Used by TruncateTail when the final line alone exceeds the byte limit.
	LastLinePartial bool
	// MaxLines is the line limit that was applied.
	MaxLines int
	// MaxBytes is the byte limit that was applied.
	MaxBytes int
}

// TruncationOptions configures the limits for truncation.
type TruncationOptions struct {
	// MaxLines is the maximum number of lines (0 uses DEFAULT_MAX_LINES).
	MaxLines int
	// MaxBytes is the maximum number of bytes (0 uses DEFAULT_MAX_BYTES).
	MaxBytes int
}

// splitLines splits content into lines. Unlike strings.Split, this handles
// the edge case of a trailing newline consistently: "a\nb\n" yields ["a","b",""].
func splitLines(content string) []string {
	lines := make([]string, 0, 64)
	start := 0
	for i := 0; i < len(content); i++ {
		if content[i] == '\n' {
			lines = append(lines, content[start:i])
			start = i + 1
		}
	}
	lines = append(lines, content[start:])
	return lines
}

// TruncateHead keeps the first N lines or bytes of content, whichever limit
// is hit first. It never returns partial lines. If the first line alone
// exceeds the byte limit, it returns empty content with FirstLineExceedsLimit
// set to true.
func TruncateHead(content string, opts *TruncationOptions) TruncationResult {
	maxLines := DEFAULT_MAX_LINES
	maxBytes := DEFAULT_MAX_BYTES
	if opts != nil {
		if opts.MaxLines > 0 {
			maxLines = opts.MaxLines
		}
		if opts.MaxBytes > 0 {
			maxBytes = opts.MaxBytes
		}
	}

	totalBytes := len(content)
	lines := splitLines(content)
	totalLines := len(lines)

	// No truncation needed.
	if totalLines <= maxLines && totalBytes <= maxBytes {
		return TruncationResult{
			Content:     content,
			Truncated:   false,
			TruncatedBy: "",
			TotalLines:  totalLines,
			TotalBytes:  totalBytes,
			OutputLines: totalLines,
			OutputBytes: totalBytes,
			MaxLines:    maxLines,
			MaxBytes:    maxBytes,
		}
	}

	// Check if the first line alone exceeds the byte limit.
	if len(lines[0]) > maxBytes {
		return TruncationResult{
			Content:               "",
			Truncated:             true,
			TruncatedBy:           "bytes",
			TotalLines:            totalLines,
			TotalBytes:            totalBytes,
			OutputLines:           0,
			OutputBytes:           0,
			FirstLineExceedsLimit: true,
			MaxLines:              maxLines,
			MaxBytes:              maxBytes,
		}
	}

	// Collect complete lines that fit within both limits.
	outputLines := make([]string, 0, maxLines)
	outputBytesCount := 0
	truncatedBy := "lines"

	for i := 0; i < len(lines) && i < maxLines; i++ {
		lineBytes := len(lines[i])
		if i > 0 {
			lineBytes++ // account for the newline separator
		}
		if outputBytesCount+lineBytes > maxBytes {
			truncatedBy = "bytes"
			break
		}
		outputLines = append(outputLines, lines[i])
		outputBytesCount += lineBytes
	}

	// If we collected maxLines lines without hitting the byte limit,
	// the truncation was caused by the line limit.
	if len(outputLines) >= maxLines && outputBytesCount <= maxBytes {
		truncatedBy = "lines"
	}

	outputContent := joinLines(outputLines)
	finalOutputBytes := len(outputContent)

	return TruncationResult{
		Content:     outputContent,
		Truncated:   true,
		TruncatedBy: truncatedBy,
		TotalLines:  totalLines,
		TotalBytes:  totalBytes,
		OutputLines: len(outputLines),
		OutputBytes: finalOutputBytes,
		MaxLines:    maxLines,
		MaxBytes:    maxBytes,
	}
}

// joinLines joins a slice of strings with newline separators.
func joinLines(lines []string) string {
	if len(lines) == 0 {
		return ""
	}
	// Pre-calculate total length.
	n := len(lines) - 1 // newlines between lines
	for _, l := range lines {
		n += len(l)
	}
	buf := make([]byte, 0, n)
	for i, l := range lines {
		if i > 0 {
			buf = append(buf, '\n')
		}
		buf = append(buf, l...)
	}
	return string(buf)
}

// TruncateTail keeps the last N lines or bytes of content, whichever limit
// is hit first. This is suitable for command output where the end (errors,
// final results) is most important.
//
// Unlike TruncateHead, TruncateTail may return a partial first line when the
// very last line of the original content exceeds the byte limit on its own.
// In that case LastLinePartial is set to true.
func TruncateTail(content string, opts *TruncationOptions) TruncationResult {
	maxLines := DEFAULT_MAX_LINES
	maxBytes := DEFAULT_MAX_BYTES
	if opts != nil {
		if opts.MaxLines > 0 {
			maxLines = opts.MaxLines
		}
		if opts.MaxBytes > 0 {
			maxBytes = opts.MaxBytes
		}
	}

	totalBytes := len(content)
	lines := splitLines(content)
	totalLines := len(lines)

	// No truncation needed.
	if totalLines <= maxLines && totalBytes <= maxBytes {
		return TruncationResult{
			Content:     content,
			Truncated:   false,
			TruncatedBy: "",
			TotalLines:  totalLines,
			TotalBytes:  totalBytes,
			OutputLines: totalLines,
			OutputBytes: totalBytes,
			MaxLines:    maxLines,
			MaxBytes:    maxBytes,
		}
	}

	// Work backwards from the end.
	outputLines := make([]string, 0, maxLines)
	outputBytesCount := 0
	truncatedBy := "lines"
	lastLinePartial := false

	for i := len(lines) - 1; i >= 0 && len(outputLines) < maxLines; i-- {
		lineBytes := len(lines[i])
		if len(outputLines) > 0 {
			lineBytes++ // account for the newline separator
		}

		if outputBytesCount+lineBytes > maxBytes {
			truncatedBy = "bytes"
			// Edge case: if we haven't added ANY lines yet and this last line
			// exceeds maxBytes, take the end of the line (partial).
			if len(outputLines) == 0 {
				truncatedLine := truncateStringFromEnd(lines[i], maxBytes)
				outputLines = append([]string{truncatedLine}, outputLines...)
				outputBytesCount = len(truncatedLine)
				lastLinePartial = true
			}
			break
		}

		outputLines = append([]string{lines[i]}, outputLines...)
		outputBytesCount += lineBytes
	}

	// If we collected maxLines lines without hitting the byte limit,
	// the truncation was caused by the line limit.
	if len(outputLines) >= maxLines && outputBytesCount <= maxBytes {
		truncatedBy = "lines"
	}

	outputContent := joinLines(outputLines)
	finalOutputBytes := len(outputContent)

	return TruncationResult{
		Content:         outputContent,
		Truncated:       true,
		TruncatedBy:     truncatedBy,
		TotalLines:      totalLines,
		TotalBytes:      totalBytes,
		OutputLines:     len(outputLines),
		OutputBytes:     finalOutputBytes,
		LastLinePartial: lastLinePartial,
		MaxLines:        maxLines,
		MaxBytes:        maxBytes,
	}
}

// truncateStringFromEnd returns the last maxBytes bytes of s.
// It avoids splitting multi-byte UTF-8 characters by scanning forward
// to a valid character boundary.
func truncateStringFromEnd(s string, maxBytes int) string {
	b := []byte(s)
	if len(b) <= maxBytes {
		return s
	}
	start := len(b) - maxBytes

	// Advance past any continuation bytes (10xxxxxx) to find a valid
	// UTF-8 character boundary.
	for start < len(b) && (b[start]&0xC0) == 0x80 {
		start++
	}

	return string(b[start:])
}

// FormatSize formats a byte count as a human-readable size string.
func FormatSize(bytes int) string {
	switch {
	case bytes < 1024:
		return fmt.Sprintf("%dB", bytes)
	case bytes < 1024*1024:
		return fmt.Sprintf("%.1fKB", float64(bytes)/1024)
	default:
		return fmt.Sprintf("%.1fMB", float64(bytes)/(1024*1024))
	}
}
