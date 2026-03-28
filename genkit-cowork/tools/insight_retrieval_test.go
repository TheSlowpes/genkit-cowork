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
	"testing"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
)

func TestFormatInsightResults_Empty(t *testing.T) {
	got := formatInsightResults(nil)
	if got != "No matching insights found." {
		t.Fatalf("formatInsightResults(nil) = %q, want empty-results message", got)
	}
}

func TestFormatInsightResults_NonEmpty(t *testing.T) {
	got := formatInsightResults([]memory.InsightRecord{{
		InsightID:  "i1",
		Kind:       memory.InsightKindFact,
		Confidence: 0.91,
		Title:      "Billing cadence",
		Summary:    "Customer expects monthly invoices",
		SessionIDs: []string{"s1"},
		FileIDs:    []string{"f1"},
	}})

	if got == "" || got == "No matching insights found." {
		t.Fatalf("formatInsightResults() = %q", got)
	}
}
