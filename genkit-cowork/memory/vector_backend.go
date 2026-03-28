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
	"encoding/json"
	"fmt"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/localvec"
)

// VectorBackend defines indexing and retrieval operations used by
// VectorOperator.
type VectorBackend interface {
	Index(ctx context.Context, tenantID, sessionID string, docs []*ai.Document) error
	RetrieveTenant(ctx context.Context, tenantID, query string, topK int) ([]*ai.Document, error)
	RetrieveSession(ctx context.Context, tenantID, sessionID, query string, topK int) ([]*ai.Document, error)
	Delete(ctx context.Context, tenantID, sessionID string) error
}

// LocalVecConfig configures the localvec-backed VectorBackend implementation.
type LocalVecConfig struct {
	Embedder        ai.Embedder
	TenantID        string
	OverFetchFactor int
}

type localVecBackend struct {
	docStore        *localvec.DocStore
	retriever       ai.Retriever
	overFetchFactor int
}

// NewLocalVecBackend creates a VectorBackend implemented with the localvec
// plugin and retriever.
func NewLocalVecBackend(g *genkit.Genkit, name string, cfg LocalVecConfig) (VectorBackend, error) {
	if err := localvec.Init(); err != nil {
		return nil, fmt.Errorf("localvec init: %w", err)
	}

	ds, retriever, err := localvec.DefineRetriever(
		g,
		name,
		localvec.Config{
			Dir:      cfg.TenantID,
			Embedder: cfg.Embedder,
		},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("localvec define retriever: %w", err)
	}

	overFetch := cfg.OverFetchFactor
	if overFetch <= 0 {
		overFetch = 10
	}

	return &localVecBackend{
		docStore:        ds,
		retriever:       retriever,
		overFetchFactor: overFetch,
	}, nil
}

func (b *localVecBackend) Index(ctx context.Context, tenantID, sessionID string, docs []*ai.Document) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("localvec index: context cancelled: %w", err)
	}

	for _, doc := range docs {
		if doc.Metadata == nil {
			doc.Metadata = make(map[string]any)
		}
		doc.Metadata["tenantID"] = tenantID
		doc.Metadata["sessionID"] = sessionID
	}

	if err := localvec.Index(ctx, docs, b.docStore); err != nil {
		return fmt.Errorf("localvec index: %w", err)
	}

	return nil
}

func (b *localVecBackend) Delete(ctx context.Context, tenantID, sessionID string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("localvec delete: context cancelled: %w", err)
	}

	var toDelete []string
	for key, val := range b.docStore.Data {
		tid, ok := val.Doc.Metadata["tenantID"].(string)
		if !ok || tid != tenantID {
			continue
		}
		sid, ok := val.Doc.Metadata["sessionID"].(string)
		if ok && sid == sessionID {
			toDelete = append(toDelete, key)
		}
	}

	for _, key := range toDelete {
		delete(b.docStore.Data, key)
	}

	if len(toDelete) > 0 {
		if err := b.persistDocStore(); err != nil {
			return fmt.Errorf("localvec delete: persist: %w", err)
		}
	}

	return nil
}

func (b *localVecBackend) persistDocStore() error {
	data, err := json.Marshal(b.docStore.Data)
	if err != nil {
		return fmt.Errorf("marshal doc store: %w", err)
	}

	tmpFile := b.docStore.Filename + ".tmp"
	if err := os.WriteFile(tmpFile, data, 0644); err != nil {
		return fmt.Errorf("write temp file: %w", err)
	}

	if err := os.Rename(tmpFile, b.docStore.Filename); err != nil {
		return fmt.Errorf("rename temp file: %w", err)
	}

	return nil
}

func (b *localVecBackend) RetrieveTenant(ctx context.Context, tenantID, query string, topK int) ([]*ai.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("localvec retrieve: context cancelled: %w", err)
	}

	fetchK := topK * b.overFetchFactor

	req := ai.RetrieverRequest{
		Query:   ai.DocumentFromText(query, nil),
		Options: localvec.RetrieverOptions{K: fetchK},
	}

	resp, err := b.retriever.Retrieve(ctx, &req)
	if err != nil {
		return nil, fmt.Errorf("localvec retrieve: %w", err)
	}

	var results []*ai.Document
	for _, doc := range resp.Documents {
		tid, ok := doc.Metadata["tenantID"].(string)
		if !ok || tid != tenantID {
			continue
		}
		results = append(results, doc)
		if len(results) >= topK {
			break
		}
	}
	return results, nil
}

func (b *localVecBackend) RetrieveSession(ctx context.Context, tenantID, sessionID, query string, topK int) ([]*ai.Document, error) {
	docs, err := b.RetrieveTenant(ctx, tenantID, query, topK*b.overFetchFactor)
	if err != nil {
		return nil, err
	}

	var results []*ai.Document
	for _, doc := range docs {
		sid, ok := doc.Metadata["sessionID"].(string)
		if !ok || sid != sessionID {
			continue
		}
		results = append(results, doc)
		if len(results) >= topK {
			break
		}
	}

	return results, nil
}
