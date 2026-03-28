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
	"log"
	"os"
	"strings"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/postgresql"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	embeddingDim = 8
	schemaName   = "public"
	tableName    = "session_memory"

	recordTypeSessionMessage = "session_message"
	recordTypeFileChunk      = "file_chunk"
)

func main() {
	ctx := context.Background()

	dsn := os.Getenv("PGVECTOR_DSN")
	if dsn == "" {
		log.Fatal("PGVECTOR_DSN is required")
	}

	database := os.Getenv("PGVECTOR_DATABASE")
	if database == "" {
		database = "postgres"
	}

	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		log.Fatalf("failed to create postgres pool: %v", err)
	}
	defer pool.Close()

	engine, err := postgresql.NewPostgresEngine(
		ctx,
		postgresql.WithPool(pool),
		postgresql.WithDatabase(database),
	)
	if err != nil {
		log.Fatalf("failed to create postgres engine: %v", err)
	}

	pgPlugin := &postgresql.Postgres{Engine: engine}
	g := genkit.Init(ctx, genkit.WithPlugins(pgPlugin))

	embedder := genkit.DefineEmbedder(g, "demo/simple-embedder", nil, simpleEmbed(embeddingDim))

	err = engine.InitVectorstoreTable(ctx, postgresql.VectorstoreTableOptions{
		SchemaName:        schemaName,
		TableName:         tableName,
		VectorSize:        embeddingDim,
		OverwriteExisting: true,
		MetadataColumns: []postgresql.Column{
			{Name: "tenant_id", DataType: "TEXT", Nullable: false},
			{Name: "record_type", DataType: "TEXT", Nullable: false},
			{Name: "session_id", DataType: "TEXT", Nullable: true},
			{Name: "message_id", DataType: "TEXT", Nullable: true},
			{Name: "kind", DataType: "TEXT", Nullable: true},
			{Name: "origin", DataType: "TEXT", Nullable: true},
			{Name: "file_id", DataType: "TEXT", Nullable: true},
			{Name: "chunk_id", DataType: "TEXT", Nullable: true},
			{Name: "mime_type", DataType: "TEXT", Nullable: true},
			{Name: "file_name", DataType: "TEXT", Nullable: true},
			{Name: "uploaded_at", DataType: "TEXT", Nullable: true},
			{Name: "extraction_mode", DataType: "TEXT", Nullable: true},
		},
	})
	if err != nil {
		log.Fatalf("failed to initialize pgvector table: %v", err)
	}

	ds, retriever, err := postgresql.DefineRetriever(ctx, g, pgPlugin, &postgresql.Config{
		SchemaName:      schemaName,
		TableName:       tableName,
		ContentColumn:   "content",
		EmbeddingColumn: "embedding",
		IDColumn:        "id",
		MetadataColumns: []string{
			"tenant_id",
			"record_type",
			"session_id",
			"message_id",
			"kind",
			"origin",
			"file_id",
			"chunk_id",
			"mime_type",
			"file_name",
			"uploaded_at",
			"extraction_mode",
		},
		Embedder: embedder,
	})
	if err != nil {
		log.Fatalf("failed to define postgres retriever: %v", err)
	}

	vectorBackend := NewPGVectorBackend(pool, ds, retriever, schemaName, tableName)
	tenantID := "tenant-1"
	fileBackend := memory.NewFileSessionOperator("./data/sessions", tenantID)
	vectorOperator := memory.NewVectorOperator(fileBackend, vectorBackend, "./data/sessions", tenantID)
	sessionStore := memory.NewSession(
		memory.WithCustomSessionOperator(vectorOperator),
		memory.WithTenantID(tenantID),
	)

	sessionID := "session-1"
	state := memory.SessionState{
		TenantID: tenantID,
		Messages: []memory.SessionMessage{
			{
				Origin: memory.UIMessage,
				Content: ai.Message{
					Role:    ai.RoleUser,
					Content: []*ai.Part{ai.NewTextPart("I need my March invoice")},
				},
			},
			{
				Origin: memory.ModelMessage,
				Content: ai.Message{
					Role:    ai.RoleModel,
					Content: []*ai.Part{ai.NewTextPart("Sure, I can help with invoice details")},
				},
			},
		},
	}

	err = sessionStore.Save(ctx, sessionID, &session.Data[memory.SessionState]{
		ID:    sessionID,
		State: state,
	})
	if err != nil {
		log.Fatalf("failed to save session: %v", err)
	}

	matches, err := vectorOperator.Search(ctx, tenantID, sessionID, "invoice", 3)
	if err != nil {
		log.Fatalf("failed to search memory: %v", err)
	}

	for i, msg := range matches {
		log.Printf("match %d id=%s kind=%s text=%q", i+1, msg.MessageID, msg.Kind, messageText(msg.Content))
	}

	fileOperator := memory.NewFileRecordOperator("./data/files")
	blobStore := memory.NewFileBlobDiskStore("./data/files")
	fileIndexer := memory.NewVectorFileIndexer(vectorBackend)
	fileIngest := memory.NewFileIngestService(fileOperator, blobStore, nil, fileIndexer)

	fileIngested, err := fileIngest.Ingest(ctx, memory.FileIngestInput{
		TenantID:      tenantID,
		SessionID:     "session-2",
		SourceChannel: memory.UIMessage,
		FileName:      "billing-policy.md",
		Data: []byte("# Billing Policy\n" +
			"Invoices are generated monthly.\n" +
			"Disputes must be opened within 15 days."),
	})
	if err != nil {
		log.Fatalf("failed to ingest tenant file memory: %v", err)
	}
	log.Printf("file ingested id=%s chunks=%d", fileIngested.File.FileID, len(fileIngested.Chunks))

	fileChunkDocs, err := vectorBackend.RetrieveTenantByRecordType(
		ctx,
		tenantID,
		"invoice dispute policy",
		recordTypeFileChunk,
		5,
	)
	if err != nil {
		log.Fatalf("failed to retrieve file chunks with record_type filter: %v", err)
	}

	for i, doc := range fileChunkDocs {
		fileID, _ := doc.Metadata["fileID"].(string)
		chunkID, _ := doc.Metadata["chunkID"].(string)
		fileName, _ := doc.Metadata["fileName"].(string)
		log.Printf("file chunk match %d fileID=%s chunkID=%s fileName=%s text=%q", i+1, fileID, chunkID, fileName, messageText(ai.Message{Content: doc.Content}))
	}
}

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
