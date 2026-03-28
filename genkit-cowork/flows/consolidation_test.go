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

package flows

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
)

type stubConsolidationRunner struct {
	run *memory.ConsolidationRunRecord
	err error

	calledTenant string
}

func (s *stubConsolidationRunner) RunTenant(ctx context.Context, tenantID string) (*memory.ConsolidationRunRecord, error) {
	s.calledTenant = tenantID
	if s.err != nil {
		return nil, s.err
	}
	return s.run, nil
}

func TestConsolidationFlow_RunSuccess(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	runner := &stubConsolidationRunner{run: &memory.ConsolidationRunRecord{
		RunID:     "run-1",
		TenantID:  "tenant-1",
		Status:    memory.ConsolidationRunSucceeded,
		CreatedAt: time.Now().UTC(),
	}}

	flow := NewConsolidationFlow(g, runner)
	out, err := flow.Run(ctx, &ConsolidationInput{TenantID: "tenant-1", RunAt: time.Now().UTC()})
	if err != nil {
		t.Fatalf("flow.Run() error = %v", err)
	}
	if out.Run.RunID != "run-1" {
		t.Fatalf("runID = %q, want %q", out.Run.RunID, "run-1")
	}
	if runner.calledTenant != "tenant-1" {
		t.Fatalf("runner called tenant = %q, want %q", runner.calledTenant, "tenant-1")
	}
}

func TestConsolidationFlow_MissingTenant(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	flow := NewConsolidationFlow(g, &stubConsolidationRunner{})
	_, err := flow.Run(ctx, &ConsolidationInput{})
	if err == nil {
		t.Fatal("expected error for missing tenantID")
	}
}

func TestConsolidationFlow_RunnerError(t *testing.T) {
	ctx := context.Background()
	g := newGenkitInstance(ctx)

	runner := &stubConsolidationRunner{err: errors.New("boom")}
	flow := NewConsolidationFlow(g, runner)
	_, err := flow.Run(ctx, &ConsolidationInput{TenantID: "tenant-1"})
	if err == nil {
		t.Fatal("expected runner error")
	}
}
