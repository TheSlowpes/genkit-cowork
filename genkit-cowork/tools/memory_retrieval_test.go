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

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
)

type mockSessionRetriever struct {
	messages []memory.SessionMessage
}

func (m *mockSessionRetriever) SearchSession(ctx context.Context, tenantID, sessionID, query string, topK int) ([]memory.SessionMessage, error) {
	return m.messages, nil
}

type mockTenantRetriever struct {
	messages []memory.SessionMessage
}

func (m *mockTenantRetriever) SearchTenant(ctx context.Context, tenantID, query string, topK int) ([]memory.SessionMessage, error) {
	return m.messages, nil
}

func TestFormatMemoryResults(t *testing.T) {
	msgs := []memory.SessionMessage{{
		MessageID: "m1",
		Kind:      memory.KindEpisodic,
		Origin:    memory.UIMessage,
		Content:   *ai.NewUserTextMessage("hello world"),
	}}

	got := formatMemoryResults(msgs)
	if got == "" {
		t.Fatal("formatMemoryResults returned empty output")
	}
	if got == "No matching memory entries found." {
		t.Fatal("formatMemoryResults unexpectedly returned empty-results message")
	}
}

func TestFormatMemoryResults_Empty(t *testing.T) {
	got := formatMemoryResults(nil)
	if got != "No matching memory entries found." {
		t.Fatalf("formatMemoryResults(nil) = %q, want empty-results message", got)
	}
}
