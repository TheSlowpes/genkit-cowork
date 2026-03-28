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

import (
	"context"
	"testing"
	"time"
)

func TestDefaultInsightOperator_SaveAndListInsights(t *testing.T) {
	op := NewDefaultInsightOperator()
	ctx := context.Background()

	insights := []InsightRecord{
		{InsightID: "i-b", Title: "B", Summary: "second", CreatedAt: time.Now().UTC()},
		{InsightID: "i-a", Title: "A", Summary: "first", CreatedAt: time.Now().UTC()},
	}
	if err := op.SaveInsights(ctx, "tenant-1", insights); err != nil {
		t.Fatalf("SaveInsights() error = %v", err)
	}

	got, err := op.ListInsights(ctx, "tenant-1")
	if err != nil {
		t.Fatalf("ListInsights() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(ListInsights()) = %d, want 2", len(got))
	}
	if got[0].InsightID != "i-a" || got[1].InsightID != "i-b" {
		t.Fatalf("ListInsights() IDs = [%s %s], want [i-a i-b]", got[0].InsightID, got[1].InsightID)
	}
}

func TestDefaultInsightOperator_RunLookupByIdempotencyKey(t *testing.T) {
	op := NewDefaultInsightOperator()
	ctx := context.Background()

	run := ConsolidationRunRecord{
		RunID:          "run-1",
		Status:         ConsolidationRunSucceeded,
		IdempotencyKey: "idem-1",
		CreatedAt:      time.Now().UTC(),
	}
	if err := op.SaveRun(ctx, "tenant-1", run); err != nil {
		t.Fatalf("SaveRun() error = %v", err)
	}

	got, err := op.GetRunByIdempotencyKey(ctx, "tenant-1", "idem-1")
	if err != nil {
		t.Fatalf("GetRunByIdempotencyKey() error = %v", err)
	}
	if got == nil || got.RunID != "run-1" {
		t.Fatalf("GetRunByIdempotencyKey() = %+v, want run-1", got)
	}
}

func TestFileInsightOperator_SaveAndList(t *testing.T) {
	op := NewFileInsightOperator(t.TempDir())
	ctx := context.Background()

	if err := op.SaveInsights(ctx, "tenant-1", []InsightRecord{{InsightID: "i1", Title: "title", Summary: "summary", CreatedAt: time.Now().UTC()}}); err != nil {
		t.Fatalf("SaveInsights() error = %v", err)
	}

	got, err := op.ListInsights(ctx, "tenant-1")
	if err != nil {
		t.Fatalf("ListInsights() error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("len(ListInsights()) = %d, want 1", len(got))
	}
	if got[0].InsightID != "i1" {
		t.Fatalf("InsightID = %q, want %q", got[0].InsightID, "i1")
	}
}
