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
	"fmt"
	"strings"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// TenantInsightRetriever retrieves tenant-scoped derived insights.
type TenantInsightRetriever interface {
	SearchTenantInsights(ctx context.Context, tenantID, query string, topK int) ([]memory.InsightRecord, error)
}

// TenantInsightSearchInput defines tenant insight retrieval parameters.
type TenantInsightSearchInput struct {
	TenantID string `json:"tenantID" jsonschema_description:"Tenant identifier"`
	Query    string `json:"query" jsonschema_description:"Semantic search query"`
	TopK     int    `json:"topK,omitempty" jsonschema_description:"Maximum insights to return"`
}

// NewSearchTenantInsightsTool creates a model-callable tool that retrieves
// semantically relevant derived insights across a tenant.
func NewSearchTenantInsightsTool(g *genkit.Genkit, retriever TenantInsightRetriever) ai.Tool {
	return genkit.DefineTool(
		g,
		"search-tenant-insights",
		"Searches semantically relevant derived insights across all tenant memory",
		func(ctx *ai.ToolContext, input TenantInsightSearchInput) (string, error) {
			if strings.TrimSpace(input.TenantID) == "" {
				return "", fmt.Errorf("tenantID is required")
			}
			if strings.TrimSpace(input.Query) == "" {
				return "", fmt.Errorf("query is required")
			}

			topK := input.TopK
			if topK <= 0 {
				topK = 5
			}

			insights, err := retriever.SearchTenantInsights(ctx, input.TenantID, input.Query, topK)
			if err != nil {
				return "", err
			}
			return formatInsightResults(insights), nil
		},
	)
}

func formatInsightResults(insights []memory.InsightRecord) string {
	if len(insights) == 0 {
		return "No matching insights found."
	}

	var b strings.Builder
	for i, insight := range insights {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(fmt.Sprintf(
			"%d. id=%s kind=%s confidence=%.2f title=%q summary=%q sessions=%v files=%v",
			i+1,
			insight.InsightID,
			insight.Kind,
			insight.Confidence,
			insight.Title,
			insight.Summary,
			insight.SessionIDs,
			insight.FileIDs,
		))
	}
	return b.String()
}
