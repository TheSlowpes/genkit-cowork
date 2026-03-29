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

package memory

import "testing"

func TestValidatePathSegment(t *testing.T) {
	valid := []string{
		"tenant-1",
		"session_abc",
		"file.123",
		"abc",
	}
	for _, v := range valid {
		t.Run("valid/"+v, func(t *testing.T) {
			if err := validatePathSegment("field", v); err != nil {
				t.Fatalf("validatePathSegment(%q) unexpected error: %v", v, err)
			}
		})
	}

	invalid := []struct {
		desc, value string
	}{
		{"empty", ""},
		{"whitespace only", "   "},
		{"dot-dot", ".."},
		{"single dot", "."},
		{"forward slash", "a/b"},
		{"backslash", `a\b`},
	}
	for _, tc := range invalid {
		t.Run("invalid/"+tc.desc, func(t *testing.T) {
			if err := validatePathSegment("field", tc.value); err == nil {
				t.Fatalf("validatePathSegment(%q) expected error, got nil", tc.value)
			}
		})
	}
}
