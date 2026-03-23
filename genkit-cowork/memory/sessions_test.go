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

package memory

import (
	"context"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/x/session"
)

func TestFilterMessages_SlidingWindow(t *testing.T) {
	msgs := []SessionMessage{{MessageID: "1"}, {MessageID: "2"}, {MessageID: "3"}}
	got := filterMessages(msgs, SlidingWindow, 2)
	if len(got) != 2 {
		t.Fatalf("len(filterMessages()) = %d, want 2", len(got))
	}
	if got[0].MessageID != "2" || got[1].MessageID != "3" {
		t.Fatalf("filterMessages() = %+v, want ids 2,3", got)
	}
}

func TestFilterMessages_TailEndsPruning(t *testing.T) {
	msgs := []SessionMessage{{MessageID: "1"}, {MessageID: "2"}, {MessageID: "3"}, {MessageID: "4"}, {MessageID: "5"}}
	got := filterMessages(msgs, TailEndsPruning, 1)
	if len(got) != 2 {
		t.Fatalf("len(filterMessages()) = %d, want 2", len(got))
	}
	if got[0].MessageID != "1" || got[1].MessageID != "5" {
		t.Fatalf("filterMessages() = %+v, want ids 1,5", got)
	}
}

func TestSessionSave_AssignsIDAndTimestamp(t *testing.T) {
	s := NewSession()
	ctx := context.Background()
	data := &session.Data[SessionState]{
		ID:    "s1",
		State: SessionState{Messages: []SessionMessage{{Content: *ai.NewUserTextMessage("hi")}}},
	}
	if err := s.Save(ctx, "s1", data); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if data.State.Messages[0].MessageID == "" {
		t.Fatal("Save() did not assign MessageID")
	}
	if data.State.Messages[0].Timestamp.Equal(time.Time{}) {
		t.Fatal("Save() did not assign Timestamp")
	}
}

func TestSessionGetAndSave(t *testing.T) {
	s := NewSession()
	ctx := context.Background()
	data := &session.Data[SessionState]{ID: "s2", State: SessionState{TenantID: "t1"}}
	if err := s.Save(ctx, "s2", data); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	got, err := s.Get(ctx, "s2")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got == nil || got.State.TenantID != "t1" {
		t.Fatalf("Get() = %+v, want tenant t1", got)
	}
}
