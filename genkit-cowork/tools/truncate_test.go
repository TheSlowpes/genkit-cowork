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

func TestTruncateHead_NoTruncation(t *testing.T) {
	content := "a\nb"
	got := TruncateHead(content, &TruncationOptions{MaxLines: 10, MaxBytes: 100})
	if got.Truncated {
		t.Fatalf("TruncateHead(%q).Truncated = true, want false", content)
	}
	if got.Content != content {
		t.Fatalf("TruncateHead(%q).Content = %q, want %q", content, got.Content, content)
	}
}

func TestTruncateHead_FirstLineExceedsLimit(t *testing.T) {
	got := TruncateHead("abcdef\nsecond", &TruncationOptions{MaxLines: 10, MaxBytes: 3})
	if !got.FirstLineExceedsLimit {
		t.Fatal("TruncateHead(...).FirstLineExceedsLimit = false, want true")
	}
	if got.Content != "" {
		t.Fatalf("TruncateHead(...).Content = %q, want empty", got.Content)
	}
}

func TestTruncateTail_LastLinePartialUTF8Safe(t *testing.T) {
	// "é" is 2 bytes in UTF-8; ensure we don't split the rune.
	got := TruncateTail("ééé", &TruncationOptions{MaxLines: 10, MaxBytes: 3})
	if !got.LastLinePartial {
		t.Fatal("TruncateTail(...).LastLinePartial = false, want true")
	}
	if got.Content != "é" {
		t.Fatalf("TruncateTail(...).Content = %q, want %q", got.Content, "é")
	}
}

func TestTruncateTail_LinesLimit(t *testing.T) {
	got := TruncateTail("1\n2\n3\n4", &TruncationOptions{MaxLines: 2, MaxBytes: 100})
	if got.TruncatedBy != "lines" {
		t.Fatalf("TruncateTail(...).TruncatedBy = %q, want %q", got.TruncatedBy, "lines")
	}
	if got.Content != "3\n4" {
		t.Fatalf("TruncateTail(...).Content = %q, want %q", got.Content, "3\n4")
	}
}

func TestFormatSize(t *testing.T) {
	tests := []struct {
		in   int
		want string
	}{
		{999, "999B"},
		{1024, "1.0KB"},
		{1024 * 1024, "1.0MB"},
	}
	for _, tt := range tests {
		if got := FormatSize(tt.in); got != tt.want {
			t.Errorf("FormatSize(%d) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
