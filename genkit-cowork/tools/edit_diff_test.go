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

import "testing"

func TestFuzzyFindText_ExactAndFuzzy(t *testing.T) {
	exact := fuzzyFindText("hello world", "world")
	if !exact.Found || exact.UsedFuzzyMatch {
		t.Fatalf("exact fuzzyFindText() = %+v, want found exact match", exact)
	}

	fuzzy := fuzzyFindText("hello\u2014world", "hello-world")
	if !fuzzy.Found || !fuzzy.UsedFuzzyMatch {
		t.Fatalf("fuzzy fuzzyFindText() = %+v, want found fuzzy match", fuzzy)
	}
}

func TestLineEndingNormalizationAndRestore(t *testing.T) {
	in := "a\r\nb\r\n"
	norm := normalizeToLF(in)
	if norm != "a\nb\n" {
		t.Fatalf("normalizeToLF() = %q, want %q", norm, "a\\nb\\n")
	}
	out := restoreLineEndings(norm, "\r\n")
	if out != in {
		t.Fatalf("restoreLineEndings() = %q, want %q", out, in)
	}
}

func TestGenerateDiffString(t *testing.T) {
	diff, first := generateDiffString("a\nb\n", "a\nc\n", nil)
	if diff == "" {
		t.Fatal("generateDiffString() diff is empty")
	}
	if first == nil || *first != 2 {
		t.Fatalf("generateDiffString() first = %v, want 2", first)
	}
}
