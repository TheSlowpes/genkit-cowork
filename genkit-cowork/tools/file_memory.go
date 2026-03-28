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
	"os"
	"strings"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// TenantFileIngestor ingests tenant-global files for recall.
type TenantFileIngestor interface {
	Ingest(ctx context.Context, input memory.FileIngestInput) (*memory.FileIngestOutput, error)
}

// TenantFileSearcher retrieves tenant-global file memory chunks.
type TenantFileSearcher interface {
	SearchTenantFiles(ctx context.Context, input memory.FileChunkSearchInput) ([]memory.FileChunkSearchResult, error)
}

// IngestTenantFileInput defines file ingestion tool inputs.
type IngestTenantFileInput struct {
	TenantID      string               `json:"tenantID" jsonschema_description:"Tenant identifier"`
	SessionID     string               `json:"sessionID,omitempty" jsonschema_description:"Source session identifier"`
	SourceChannel memory.MessageOrigin `json:"sourceChannel,omitempty" jsonschema_description:"Source channel for provenance"`
	Path          string               `json:"path" jsonschema_description:"Path to a local file to ingest"`
	MimeType      string               `json:"mimeType,omitempty" jsonschema_description:"Optional MIME override"`
}

// SearchTenantFileMemoryInput defines file memory retrieval tool inputs.
type SearchTenantFileMemoryInput struct {
	TenantID string `json:"tenantID" jsonschema_description:"Tenant identifier"`
	Query    string `json:"query" jsonschema_description:"Semantic search query"`
	TopK     int    `json:"topK,omitempty" jsonschema_description:"Maximum chunks to return"`
}

// NewIngestTenantFileMemoryTool creates a tool that ingests tenant-global file
// memory from local filesystem paths.
func NewIngestTenantFileMemoryTool(g *genkit.Genkit, cwd string, ingestor *memory.FileIngestService) ai.Tool {
	return genkit.DefineTool(
		g,
		"ingest-tenant-file-memory",
		"Ingests a tenant-global text/structured file (txt, md, json, csv, html) for cross-session recall",
		func(ctx *ai.ToolContext, input IngestTenantFileInput) (string, error) {
			if strings.TrimSpace(input.TenantID) == "" {
				return "", fmt.Errorf("tenantID is required")
			}
			if strings.TrimSpace(input.Path) == "" {
				return "", fmt.Errorf("path is required")
			}

			resolved, err := resolveReadPath(input.Path, cwd)
			if err != nil {
				return "", err
			}

			data, err := os.ReadFile(resolved)
			if err != nil {
				return "", fmt.Errorf("read file: %w", err)
			}

			out, err := ingestor.Ingest(ctx, memory.FileIngestInput{
				TenantID:      input.TenantID,
				SessionID:     input.SessionID,
				SourceChannel: input.SourceChannel,
				FileName:      resolved,
				MimeType:      input.MimeType,
				Data:          data,
			})
			if err != nil {
				return "", err
			}

			return fmt.Sprintf("Ingested file id=%s name=%s mime=%s chunks=%d status=%s", out.File.FileID, out.File.Name, out.File.MimeType, len(out.Chunks), out.File.IngestStatus), nil
		},
	)
}

// NewSearchTenantFileMemoryTool creates a tool for tenant-scoped file memory
// retrieval.
func NewSearchTenantFileMemoryTool(g *genkit.Genkit, service *memory.FileIngestService) ai.Tool {
	return genkit.DefineTool(
		g,
		"search-tenant-file-memory",
		"Searches semantically relevant tenant-global file chunks across sessions",
		func(ctx *ai.ToolContext, input SearchTenantFileMemoryInput) (string, error) {
			if strings.TrimSpace(input.TenantID) == "" {
				return "", fmt.Errorf("tenantID is required")
			}
			if strings.TrimSpace(input.Query) == "" {
				return "", fmt.Errorf("query is required")
			}

			results, err := service.SearchTenantFiles(ctx, memory.FileChunkSearchInput{
				TenantID: input.TenantID,
				Query:    input.Query,
				TopK:     input.TopK,
			})
			if err != nil {
				return "", err
			}
			return formatFileMemoryResults(results), nil
		},
	)
}

func formatFileMemoryResults(results []memory.FileChunkSearchResult) string {
	if len(results) == 0 {
		return "No matching file memory entries found."
	}

	var b strings.Builder
	for i, item := range results {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(fmt.Sprintf(
			"%d. file=%s chunk=%s session=%s mime=%s text=%q",
			i+1,
			item.File.Name,
			item.Chunk.ChunkID,
			item.Chunk.SessionID,
			item.File.MimeType,
			item.Chunk.Text,
		))
	}
	return b.String()
}
