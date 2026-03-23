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
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func TestFitDimensions(t *testing.T) {
	w, h := fitDimensions(4000, 2000, 2000, 2000)
	if w != 2000 || h != 1000 {
		t.Fatalf("fitDimensions() = %dx%d, want 2000x1000", w, h)
	}
}

func TestIsValidImageMimeType(t *testing.T) {
	if !isValidImageMimeType("image/png") {
		t.Fatal("isValidImageMimeType(image/png) = false, want true")
	}
	if isValidImageMimeType("text/plain") {
		t.Fatal("isValidImageMimeType(text/plain) = true, want false")
	}
}

func TestDetectImageMimeType(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "x.png")
	f, err := os.Create(p)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	if err := png.Encode(f, img); err != nil {
		t.Fatalf("png.Encode() error = %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if got := DetectImageMimeType(p); got != "image/png" {
		t.Fatalf("DetectImageMimeType() = %q, want %q", got, "image/png")
	}
}

func TestFormatDimensionNote(t *testing.T) {
	if got := FormatDimensionNote(&ResizeResult{WasResized: false}); got != "" {
		t.Fatalf("FormatDimensionNote() = %q, want empty", got)
	}
}
