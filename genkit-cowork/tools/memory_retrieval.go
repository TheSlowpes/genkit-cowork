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

// SessionMemoryRetriever retrieves session-scoped semantic memory.
type SessionMemoryRetriever interface {
	SearchSession(ctx context.Context, tenantID, sessionID, query string, topK int) ([]memory.SessionMessage, error)
}

// TenantMemoryRetriever retrieves tenant-wide semantic memory.
type TenantMemoryRetriever interface {
	SearchTenant(ctx context.Context, tenantID, query string, topK int) ([]memory.SessionMessage, error)
}

// SessionMemorySearchInput defines session-scoped retrieval parameters.
type SessionMemorySearchInput struct {
	TenantID  string `json:"tenantID" jsonschema_description:"Tenant identifier"`
	SessionID string `json:"sessionID" jsonschema_description:"Session identifier"`
	Query     string `json:"query" jsonschema_description:"Semantic search query"`
	TopK      int    `json:"topK,omitempty" jsonschema_description:"Maximum number of messages to return"`
}

// TenantMemorySearchInput defines tenant-wide retrieval parameters.
type TenantMemorySearchInput struct {
	TenantID string `json:"tenantID" jsonschema_description:"Tenant identifier"`
	Query    string `json:"query" jsonschema_description:"Semantic search query"`
	TopK     int    `json:"topK,omitempty" jsonschema_description:"Maximum number of messages to return"`
}

// NewSearchSessionMemoryTool creates a model-callable tool that retrieves
// semantically similar messages within one tenant session.
func NewSearchSessionMemoryTool(g *genkit.Genkit, retriever SessionMemoryRetriever) ai.Tool {
	return genkit.DefineTool(
		g,
		"search-session-memory",
		"Searches semantically relevant memory messages within one tenant session",
		func(ctx *ai.ToolContext, input SessionMemorySearchInput) (string, error) {
			if strings.TrimSpace(input.TenantID) == "" {
				return "", fmt.Errorf("tenantID is required")
			}
			if strings.TrimSpace(input.SessionID) == "" {
				return "", fmt.Errorf("sessionID is required")
			}
			if strings.TrimSpace(input.Query) == "" {
				return "", fmt.Errorf("query is required")
			}

			topK := input.TopK
			if topK <= 0 {
				topK = 5
			}

			messages, err := retriever.SearchSession(ctx, input.TenantID, input.SessionID, input.Query, topK)
			if err != nil {
				return "", err
			}
			return formatMemoryResults(messages), nil
		},
	)
}

// NewSearchTenantMemoryTool creates a model-callable tool that retrieves
// semantically relevant messages across all tenant sessions.
func NewSearchTenantMemoryTool(g *genkit.Genkit, retriever TenantMemoryRetriever) ai.Tool {
	return genkit.DefineTool(
		g,
		"search-tenant-memory",
		"Searches semantically relevant memory messages across all tenant sessions",
		func(ctx *ai.ToolContext, input TenantMemorySearchInput) (string, error) {
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

			messages, err := retriever.SearchTenant(ctx, input.TenantID, input.Query, topK)
			if err != nil {
				return "", err
			}
			return formatMemoryResults(messages), nil
		},
	)
}

func formatMemoryResults(messages []memory.SessionMessage) string {
	if len(messages) == 0 {
		return "No matching memory entries found."
	}

	var b strings.Builder
	for i, msg := range messages {
		if i > 0 {
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "%d. id=%s kind=%s origin=%s text=%q", i+1, msg.MessageID, msg.Kind, msg.Origin, msg.Content.Text())
	}
	return b.String()
}
