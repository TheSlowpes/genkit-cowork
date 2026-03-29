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

package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/postgresql"
	"github.com/jackc/pgx/v5/pgxpool"
)

type PGVectorBackend struct {
	pool      *pgxpool.Pool
	docStore  *postgresql.DocStore
	retriever ai.Retriever
	schema    string
	table     string
}

func NewPGVectorBackend(pool *pgxpool.Pool, docStore *postgresql.DocStore, retriever ai.Retriever, schema, table string) *PGVectorBackend {
	return &PGVectorBackend{
		pool:      pool,
		docStore:  docStore,
		retriever: retriever,
		schema:    schema,
		table:     table,
	}
}

func (b *PGVectorBackend) Index(ctx context.Context, tenantID, _ string, docs []*ai.Document) error {
	for _, doc := range docs {
		if doc.Metadata == nil {
			doc.Metadata = make(map[string]any)
		}
		doc.Metadata["tenantID"] = tenantID
		if _, ok := doc.Metadata["recordType"]; !ok {
			doc.Metadata["recordType"] = recordTypeSessionMessage
		}

		copyMetadata(doc.Metadata, "sessionID", "session_id")
		copyMetadata(doc.Metadata, "tenantID", "tenant_id")
		copyMetadata(doc.Metadata, "messageID", "message_id")
		copyMetadata(doc.Metadata, "recordType", "record_type")
		copyMetadata(doc.Metadata, "fileID", "file_id")
		copyMetadata(doc.Metadata, "chunkID", "chunk_id")
		copyMetadata(doc.Metadata, "mimeType", "mime_type")
		copyMetadata(doc.Metadata, "fileName", "file_name")
		copyMetadata(doc.Metadata, "uploadedAt", "uploaded_at")
		copyMetadata(doc.Metadata, "extractionMode", "extraction_mode")
	}

	return b.docStore.Index(ctx, docs)
}

func (b *PGVectorBackend) RetrieveTenant(ctx context.Context, tenantID, query string, topK int) ([]*ai.Document, error) {
	resp, err := b.retriever.Retrieve(ctx, &ai.RetrieverRequest{
		Query: ai.DocumentFromText(query, nil),
		Options: &postgresql.RetrieverOptions{
			K:      topK,
			Filter: fmt.Sprintf("tenant_id = '%s'", escapeLiteral(tenantID)),
		},
	})
	if err != nil {
		return nil, err
	}

	for _, doc := range resp.Documents {
		copyMetadata(doc.Metadata, "record_type", "recordType")
		copyMetadata(doc.Metadata, "message_id", "messageID")
		copyMetadata(doc.Metadata, "session_id", "sessionID")
		copyMetadata(doc.Metadata, "tenant_id", "tenantID")
		copyMetadata(doc.Metadata, "file_id", "fileID")
		copyMetadata(doc.Metadata, "chunk_id", "chunkID")
		copyMetadata(doc.Metadata, "mime_type", "mimeType")
		copyMetadata(doc.Metadata, "file_name", "fileName")
		copyMetadata(doc.Metadata, "uploaded_at", "uploadedAt")
		copyMetadata(doc.Metadata, "extraction_mode", "extractionMode")
	}

	return resp.Documents, nil
}

func (b *PGVectorBackend) RetrieveSession(ctx context.Context, tenantID, sessionID, query string, topK int) ([]*ai.Document, error) {
	resp, err := b.retriever.Retrieve(ctx, &ai.RetrieverRequest{
		Query: ai.DocumentFromText(query, nil),
		Options: &postgresql.RetrieverOptions{
			K:      topK,
			Filter: fmt.Sprintf("tenant_id = '%s' AND session_id = '%s'", escapeLiteral(tenantID), escapeLiteral(sessionID)),
		},
	})
	if err != nil {
		return nil, err
	}

	for _, doc := range resp.Documents {
		copyMetadata(doc.Metadata, "record_type", "recordType")
		copyMetadata(doc.Metadata, "message_id", "messageID")
		copyMetadata(doc.Metadata, "session_id", "sessionID")
		copyMetadata(doc.Metadata, "tenant_id", "tenantID")
		copyMetadata(doc.Metadata, "file_id", "fileID")
		copyMetadata(doc.Metadata, "chunk_id", "chunkID")
		copyMetadata(doc.Metadata, "mime_type", "mimeType")
		copyMetadata(doc.Metadata, "file_name", "fileName")
		copyMetadata(doc.Metadata, "uploaded_at", "uploadedAt")
		copyMetadata(doc.Metadata, "extraction_mode", "extractionMode")
	}

	return resp.Documents, nil
}

func (b *PGVectorBackend) Delete(ctx context.Context, tenantID, sessionID string) error {
	query := fmt.Sprintf(`DELETE FROM "%s"."%s" WHERE tenant_id = $1 AND session_id = $2`, b.schema, b.table)
	_, err := b.pool.Exec(ctx, query, tenantID, sessionID)
	return err
}

func (b *PGVectorBackend) RetrieveTenantByRecordType(ctx context.Context, tenantID, query, recordType string, topK int) ([]*ai.Document, error) {
	resp, err := b.retriever.Retrieve(ctx, &ai.RetrieverRequest{
		Query: ai.DocumentFromText(query, nil),
		Options: &postgresql.RetrieverOptions{
			K: topK,
			Filter: fmt.Sprintf(
				"tenant_id = '%s' AND record_type = '%s'",
				escapeLiteral(tenantID),
				escapeLiteral(recordType),
			),
		},
	})
	if err != nil {
		return nil, err
	}

	for _, doc := range resp.Documents {
		copyMetadata(doc.Metadata, "record_type", "recordType")
		copyMetadata(doc.Metadata, "message_id", "messageID")
		copyMetadata(doc.Metadata, "session_id", "sessionID")
		copyMetadata(doc.Metadata, "tenant_id", "tenantID")
		copyMetadata(doc.Metadata, "file_id", "fileID")
		copyMetadata(doc.Metadata, "chunk_id", "chunkID")
		copyMetadata(doc.Metadata, "mime_type", "mimeType")
		copyMetadata(doc.Metadata, "file_name", "fileName")
		copyMetadata(doc.Metadata, "uploaded_at", "uploadedAt")
		copyMetadata(doc.Metadata, "extraction_mode", "extractionMode")
	}

	return resp.Documents, nil
}

func simpleEmbed(dim int) ai.EmbedderFunc {
	return func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("embed request cancelled: %w", err)
		}

		res := &ai.EmbedResponse{Embeddings: make([]*ai.Embedding, 0, len(req.Input))}

		for _, doc := range req.Input {
			vec := make([]float32, dim)
			text := messageText(ai.Message{Content: doc.Content})

			for i, r := range text {
				vec[i%dim] += float32(r%97) / 100
			}

			res.Embeddings = append(res.Embeddings, &ai.Embedding{Embedding: vec})
		}

		return res, nil
	}
}

func messageText(msg ai.Message) string {
	parts := make([]string, 0, len(msg.Content))
	for _, part := range msg.Content {
		if part.IsText() {
			parts = append(parts, part.Text)
		}
	}
	return strings.Join(parts, " ")
}

func copyMetadata(metadata map[string]any, sourceKey, targetKey string) {
	if value, ok := metadata[sourceKey]; ok {
		metadata[targetKey] = value
	}
}

func escapeLiteral(input string) string {
	return strings.ReplaceAll(input, "'", "''")
}
