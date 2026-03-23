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

import "testing"

func TestTokenize(t *testing.T) {
	got := tokenize("a\nb\n", true)
	if len(got) != 2 {
		t.Fatalf("len(tokenize()) = %d, want 2", len(got))
	}
	if got[0] != "a\n" || got[1] != "b\n" {
		t.Fatalf("tokenize() = %#v, want [\"a\\n\", \"b\\n\"]", got)
	}
}

func TestDiffLines_InsertDelete(t *testing.T) {
	changes := diffLines("a\nb\n", "a\nc\n")
	if len(changes) == 0 {
		t.Fatal("diffLines() returned empty changes")
	}

	hasDelete := false
	hasInsert := false
	for _, c := range changes {
		if c.Type == Delete && c.Value == "b\n" {
			hasDelete = true
		}
		if c.Type == Insert && c.Value == "c\n" {
			hasInsert = true
		}
	}
	if !hasDelete {
		t.Error("diffLines() missing delete change for b")
	}
	if !hasInsert {
		t.Error("diffLines() missing insert change for c")
	}
}
