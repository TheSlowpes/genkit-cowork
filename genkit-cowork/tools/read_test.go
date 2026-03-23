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
	"errors"
	"testing"
)

type readOpMock struct {
	buf      []byte
	mimeType string
	readErr  error
	accErr   error
}

func (m *readOpMock) ReadFile(ctx context.Context, absolutePath string) ([]byte, error) {
	if m.readErr != nil {
		return nil, m.readErr
	}
	return m.buf, nil
}

func (m *readOpMock) Access(ctx context.Context, absolutePath string) error {
	return m.accErr
}

func (m *readOpMock) DetectImageMimeType(absolutePath string) string {
	return m.mimeType
}

func TestHandleTextRead_WithOffsetLimit(t *testing.T) {
	ops := &readOpMock{buf: []byte("a\nb\nc\n")}
	resp, err := handleTextRead(context.Background(), "/tmp/x.txt", ReadToolInput{Offset: 2, Limit: 2}, ops)
	if err != nil {
		t.Fatalf("handleTextRead() error = %v", err)
	}
	got, ok := resp.Output.(string)
	if !ok {
		t.Fatal("handleTextRead() output type is not string")
	}
	if got == "" {
		t.Fatal("handleTextRead() output is empty")
	}
}

func TestHandleImageRead_NoResize(t *testing.T) {
	ops := &readOpMock{buf: []byte("abc")}
	resp, err := handleImageRead(context.Background(), "/tmp/x.png", "image/png", ops, false)
	if err != nil {
		t.Fatalf("handleImageRead() error = %v", err)
	}
	if resp == nil || len(resp.Content) != 1 {
		t.Fatal("handleImageRead() missing media part")
	}
}

func TestHandleRead_AccessDenied(t *testing.T) {
	ctx := context.Background()
	op := &readOpMock{accErr: errors.New("denied")}
	_, err := handleRead(ctx, ReadToolInput{Path: "read_test.go"}, ".", &readToolOptions{operator: op, autoResizeImages: true})
	if err == nil {
		t.Fatal("handleRead() error = nil, want non-nil")
	}
}
