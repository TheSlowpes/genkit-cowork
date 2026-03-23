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
	"context"
	"testing"
)

type editOpMock struct {
	content string
	written string
}

func (m *editOpMock) ReadFile(ctx context.Context, absolutePath string) ([]byte, error) {
	return []byte(m.content), nil
}

func (m *editOpMock) WriteFile(ctx context.Context, absolutePath, content string) error {
	m.written = content
	return nil
}

func (m *editOpMock) Access(ctx context.Context, absolutePath string) error { return nil }

func TestPerformEdit_ExactMatch(t *testing.T) {
	m := &editOpMock{content: "hello world"}
	out, err := performEdit(context.Background(), EditToolInput{Path: "x", OldText: "world", NewText: "friend"}, "/tmp", &editToolOptions{operator: m})
	if err != nil {
		t.Fatalf("performEdit() error = %v", err)
	}
	if out.Content == "" {
		t.Fatal("performEdit() output content is empty")
	}
	if m.written != "hello friend" {
		t.Fatalf("performEdit() wrote %q, want %q", m.written, "hello friend")
	}
}

func TestPerformEdit_MultipleMatchesRejected(t *testing.T) {
	m := &editOpMock{content: "x y x"}
	_, err := performEdit(context.Background(), EditToolInput{Path: "x", OldText: "x", NewText: "z"}, "/tmp", &editToolOptions{operator: m})
	if err == nil {
		t.Fatal("performEdit() error = nil, want non-nil")
	}
}
