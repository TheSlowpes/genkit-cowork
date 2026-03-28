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
	"fmt"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

// ConsolidationInput is the input for one consolidation flow run.
type ConsolidationInput struct {
	TenantID string    `json:"tenantID"`
	RunAt    time.Time `json:"runAt"`
}

// ConsolidationOutput is the output for one consolidation flow run.
type ConsolidationOutput struct {
	Run memory.ConsolidationRunRecord `json:"run"`
}

// ConsolidationRunner is the behavior required by the consolidation flow.
type ConsolidationRunner interface {
	RunTenant(ctx context.Context, tenantID string) (*memory.ConsolidationRunRecord, error)
}

// NewConsolidationFlow creates a flow wrapper over tenant consolidation.
//
// This flow is optional and intended for scheduled/background invocations where
// traceable Genkit execution is desired.
func NewConsolidationFlow(
	g *genkit.Genkit,
	runner ConsolidationRunner,
) *core.Flow[*ConsolidationInput, *ConsolidationOutput, struct{}] {
	return genkit.DefineFlow(
		g,
		"consolidation",
		func(ctx context.Context, input *ConsolidationInput) (*ConsolidationOutput, error) {
			if input == nil {
				return nil, fmt.Errorf("consolidation flow: input is nil")
			}
			if input.TenantID == "" {
				return nil, fmt.Errorf("consolidation flow: tenantID is required")
			}
			if runner == nil {
				return nil, fmt.Errorf("consolidation flow: runner is nil")
			}

			run, err := runner.RunTenant(ctx, input.TenantID)
			if err != nil {
				return nil, fmt.Errorf("consolidation flow: run tenant: %w", err)
			}
			if run == nil {
				return nil, fmt.Errorf("consolidation flow: runner returned nil run")
			}

			return &ConsolidationOutput{Run: *run}, nil
		},
	)
}
