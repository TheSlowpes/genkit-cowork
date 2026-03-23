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

type writeOpMock struct {
	wrotePath string
	wroteBody string
	mkdirPath string
}

func (m *writeOpMock) WriteFile(ctx context.Context, absolutePath, content string) error {
	m.wrotePath = absolutePath
	m.wroteBody = content
	return nil
}

func (m *writeOpMock) MkdirAll(ctx context.Context, absolutePath string) error {
	m.mkdirPath = absolutePath
	return nil
}

func TestPerformWrite_UsesOperator(t *testing.T) {
	m := &writeOpMock{}
	msg, err := performWrite(context.Background(), WriteToolInput{Path: "a/b.txt", Content: "hello"}, "/tmp", &writeToolOptions{operator: m})
	if err != nil {
		t.Fatalf("performWrite() error = %v", err)
	}
	if m.wroteBody != "hello" {
		t.Fatalf("performWrite() wrote %q, want %q", m.wroteBody, "hello")
	}
	if msg == "" {
		t.Fatal("performWrite() message is empty")
	}
}
