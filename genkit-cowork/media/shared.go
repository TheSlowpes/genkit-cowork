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

package media

import (
	"errors"
	"strings"

	"github.com/firebase/genkit/go/ai"
)

const (
	defaultChunkSize    = 1200
	defaultChunkOverlap = 200
)

var errFileTypeNotSupported = errors.New("file type not supported")

type textChunker struct {
	chunkSize int
	overlap   int
}

type chunkRange struct {
	Text  string
	Start int
	End   int
}

func newDefaultTextChunker() textChunker {
	return textChunker{chunkSize: defaultChunkSize, overlap: defaultChunkOverlap}
}

func (c textChunker) chunk(text string) []chunkRange {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil
	}

	chunkSize := c.chunkSize
	if chunkSize <= 0 {
		chunkSize = defaultChunkSize
	}
	overlap := c.overlap
	overlap = max(overlap, 0)
	if overlap >= chunkSize {
		overlap = chunkSize / 4
	}
	step := chunkSize - overlap
	if step <= 0 {
		step = chunkSize
	}

	runes := []rune(trimmed)
	chunks := make([]chunkRange, 0)
	for start := 0; start < len(runes); start += step {
		end := start + chunkSize
		end = min(end, len(runes))
		piece := strings.TrimSpace(string(runes[start:end]))
		if piece == "" {
			if end >= len(runes) {
				break
			}
			continue
		}
		chunks = append(chunks, chunkRange{Text: piece, Start: start, End: end})
		if end >= len(runes) {
			break
		}
	}

	return chunks
}

func chunksToDocuments(mimeType string, chunks []chunkRange) []*ai.Document {
	if len(chunks) == 0 {
		return nil
	}
	docs := make([]*ai.Document, 0, len(chunks))
	total := len(chunks)
	for i, chunk := range chunks {
		docs = append(docs, ai.DocumentFromText(chunk.Text, map[string]any{
			"mimeType":   mimeType,
			"chunkIndex": i,
			"chunkCount": total,
			"charStart":  chunk.Start,
			"charEnd":    chunk.End,
		}))
	}
	return docs
}

func stripHTMLTags(input string) string {
	var b strings.Builder
	b.Grow(len(input))
	inTag := false
	lastWasSpace := false

	for _, r := range input {
		switch {
		case r == '<':
			inTag = true
			if !lastWasSpace {
				b.WriteRune(' ')
				lastWasSpace = true
			}
		case r == '>':
			inTag = false
		case inTag:
			continue
		case r == '\n' || r == '\r' || r == '\t':
			if !lastWasSpace {
				b.WriteRune(' ')
				lastWasSpace = true
			}
		case r == ' ':
			if !lastWasSpace {
				b.WriteRune(' ')
				lastWasSpace = true
			}
		default:
			b.WriteRune(r)
			lastWasSpace = false
		}
	}

	return strings.TrimSpace(b.String())
}
