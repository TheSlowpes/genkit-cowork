// Copyright 2025 Kevin Lopes
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
	"os"
	"path/filepath"
	"testing"
)

func TestResolveReadPath(t *testing.T) {
	tmp := t.TempDir()
	file := filepath.Join(tmp, "a.txt")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	got, err := resolveReadPath("a.txt", tmp)
	if err != nil {
		t.Fatalf("resolveReadPath() error = %v", err)
	}
	if got != file {
		t.Errorf("resolveReadPath() = %q, want %q", got, file)
	}
}

func TestResolveReadPath_NotFound(t *testing.T) {
	_, err := resolveReadPath("missing.txt", t.TempDir())
	if err == nil {
		t.Fatal("resolveReadPath() error = nil, want non-nil")
	}
}

func TestExpandPath(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir() error = %v", err)
	}
	if got := expandPath("~"); got != home {
		t.Errorf("expandPath(~) = %q, want %q", got, home)
	}
}

func TestResolveToCwd(t *testing.T) {
	cwd := t.TempDir()
	want := filepath.Join(cwd, "file.txt")
	if got := resolveToCwd("file.txt", cwd); got != want {
		t.Errorf("resolveToCwd() = %q, want %q", got, want)
	}
}
