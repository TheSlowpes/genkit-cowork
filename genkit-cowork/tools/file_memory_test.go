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
	"testing"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
)

func TestFormatFileMemoryResults_Empty(t *testing.T) {
	if got := formatFileMemoryResults(nil); got != "No matching file memory entries found." {
		t.Fatalf("formatFileMemoryResults(nil) = %q", got)
	}
}

func TestFormatFileMemoryResults_NonEmpty(t *testing.T) {
	results := []memory.FileChunkSearchResult{{
		File: memory.FileRecord{Name: "policy.md", MimeType: "text/markdown"},
		Chunk: memory.FileChunkRecord{
			ChunkID:   "chunk-1",
			SessionID: "session-1",
			Text:      "invoice policy text",
		},
	}}

	got := formatFileMemoryResults(results)
	if got == "" || got == "No matching file memory entries found." {
		t.Fatalf("formatFileMemoryResults() = %q", got)
	}
}
