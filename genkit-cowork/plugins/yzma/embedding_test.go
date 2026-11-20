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

package yzma

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/hybridgroup/yzma/pkg/llama"
)

type testEmbeddingOperator struct {
	*testModelOperator
	embed func(context.Context, llama.Model, EmbeddingInput) ([][]float32, error)
}

func (o *testEmbeddingOperator) EmbedText(ctx context.Context, model llama.Model, input EmbeddingInput) ([][]float32, error) {
	return o.embed(ctx, model, input)
}

func writeEmbeddingFixture(t *testing.T, dir, name string, pooling uint64) string {
	t.Helper()
	return writeGGUF(t, dir, name, map[string]ggufValue{
		"general.architecture":  {kind: ggufString, text: "bert"},
		"bert.attention.causal": {kind: 7},
		"bert.pooling_type":     {kind: 4, num: pooling},
		"bert.embedding_length": {kind: 4, num: 2},
		"tokenizer.ggml.model":  {kind: ggufString, text: "bert"},
		"tokenizer.ggml.tokens": {kind: ggufArray},
	}, "token_embd.weight", "blk.0.attn_q.weight", "blk.0.layer_output_norm.weight")
}

func TestDiscoveredEmbedderThroughGenkit(t *testing.T) {
	dir := t.TempDir()
	path := writeEmbeddingFixture(t, dir, "Encoder.Q4.gguf", uint64(llama.PoolingTypeCLS))
	writeEmbeddingFixture(t, dir, "ranker.gguf", uint64(llama.PoolingTypeRank))
	writeGGUFFixture(t, dir, "text.gguf", nil)
	var loaded []string
	var freed []llama.Model
	raw := [][]float32{{3, 4}, {0, -2}}
	op := &testEmbeddingOperator{
		testModelOperator: &testModelOperator{
			load: func(_ context.Context, opts ModelOptions) (llama.Model, error) {
				loaded = append(loaded, opts.GGUFPath)
				return 42, nil
			},
			free: func(_ context.Context, model llama.Model) error {
				freed = append(freed, model)
				return nil
			},
		},
		embed: func(_ context.Context, model llama.Model, input EmbeddingInput) ([][]float32, error) {
			if model != 42 || input.Pooling != llama.PoolingTypeCLS || input.Dimensions != 2 {
				t.Errorf("model handle %d, embedding input %+v; want handle 42, CLS pooling and 2 dimensions", model, input)
			}
			if diff := cmp.Diff([]string{"first document", "second"}, input.Texts); diff != "" {
				t.Errorf("texts (-want +got):\n%s", diff)
			}
			return raw, nil
		},
	}
	plugin, g := initDiscoveryTest(t, dir, op)
	ctx := t.Context()
	const name = "yzma/encoder.q4"
	if Model(g, "encoder.q4") != nil || genkit.LookupEmbedder(g, "yzma/text") != nil || genkit.LookupEmbedder(g, "yzma/ranker") != nil {
		t.Fatal("a model resolved under the wrong role")
	}
	embedder := genkit.LookupEmbedder(g, name)
	if embedder == nil {
		t.Fatal("discovered embedder did not resolve before listing")
	}
	actions := plugin.ListActions(ctx)
	if len(actions) != 2 || actions[0].Name != name || actions[0].Key != "/embedder/"+name ||
		actions[0].Type != api.ActionTypeEmbedder || actions[0].InputSchema == nil ||
		actions[0].Metadata["info"].(map[string]any)["dimensions"] != 2 || actions[1].Type != api.ActionTypeModel {
		t.Fatalf("invalid catalog: %+v", actions)
	}
	if len(loaded) != 0 {
		t.Fatalf("resolution or listing loaded models: %v", loaded)
	}
	for range 2 {
		got, err := genkit.Embed(ctx, g, ai.WithEmbedderName(name), ai.WithDocs(
			&ai.Document{Content: []*ai.Part{ai.NewTextPart("first "), ai.NewTextPart("document")}},
			ai.DocumentFromText("second", nil),
		))
		if err != nil {
			t.Fatal(err)
		}
		want := []*ai.Embedding{{Embedding: []float32{0.6, 0.8}}, {Embedding: []float32{0, -1}}}
		if diff := cmp.Diff(want, got.Embeddings); diff != "" {
			t.Errorf("embeddings (-want +got):\n%s", diff)
		}
	}
	if diff := cmp.Diff([]string{path}, loaded); diff != "" {
		t.Errorf("loaded paths (-want +got):\n%s", diff)
	}
	if err := plugin.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff([]llama.Model{42}, freed); diff != "" {
		t.Errorf("freed weights (-want +got):\n%s", diff)
	}
	if _, err := embedder.Embed(ctx, &ai.EmbedRequest{Input: []*ai.Document{ai.DocumentFromText("test", nil)}}); !errors.Is(err, ErrClosed) {
		t.Errorf("cached embedder after Close: got %v, want ErrClosed", err)
	}
}

func TestDefineEmbedderOverridesCatalogAndValidatesVectors(t *testing.T) {
	dir := t.TempDir()
	writeEmbeddingFixture(t, dir, "local.gguf", uint64(llama.PoolingTypeCLS))
	unknown := writeEmbeddingFixture(t, dir, "unknown-role.gguf", uint64(llama.PoolingTypeNone))
	var vectors [][]float32
	loads := 0
	op := &testEmbeddingOperator{
		testModelOperator: &testModelOperator{
			load: func(_ context.Context, opts ModelOptions) (llama.Model, error) {
				loads++
				if opts.GGUFPath != unknown {
					t.Errorf("loaded %q, want explicitly configured %q", opts.GGUFPath, unknown)
				}
				return 1, nil
			},
		},
		embed: func(_ context.Context, _ llama.Model, input EmbeddingInput) ([][]float32, error) {
			if input.Pooling != llama.PoolingTypeMean {
				t.Errorf("embedder pooling = %v, want mean", input.Pooling)
			}
			return vectors, nil
		},
	}
	ctx := t.Context()
	plugin, g := initDiscoveryTest(t, dir, op)
	if len(plugin.ListActions(ctx)) != 1 || genkit.LookupEmbedder(g, "yzma/unknown-role") != nil {
		t.Fatal("GGUF with an unknown embedding role did not stay out of the catalog")
	}
	embedder := plugin.DefineEmbedder(g, EmbeddingDefinition{
		Name: "local", Loading: ModelOptions{GGUFPath: unknown},
		Pooling: llama.PoolingTypeMean, Dimensions: 2,
	})
	if len(plugin.ListActions(ctx)) != 0 || loads != 0 {
		t.Fatal("explicit definition should suppress the catalog entry without loading weights")
	}
	request := &ai.EmbedRequest{Input: []*ai.Document{ai.DocumentFromText("text", nil)}}
	for _, request := range []*ai.EmbedRequest{
		{},
		{Input: []*ai.Document{nil}},
		{Input: []*ai.Document{ai.DocumentFromText("", nil)}},
		{Input: []*ai.Document{{Content: []*ai.Part{ai.NewTextPart("text"), ai.NewMediaPart("image/png", "data:image/png;base64,AA==")}}}},
		{Input: request.Input, Options: map[string]any{"dimensions": 1}},
	} {
		if resp, err := embedder.Embed(ctx, request); err == nil || resp != nil {
			t.Errorf("invalid input returned response %v, err %v", resp, err)
		}
	}
	if loads != 0 {
		t.Fatalf("invalid requests loaded weights %d times, want 0", loads)
	}
	for _, tc := range []struct {
		vectors [][]float32
		wantErr string
	}{
		{nil, "got 0 embedding vectors"},
		{[][]float32{{1}}, "vector has 1 dimensions"},
		{[][]float32{{0, 0}}, "finite, nonzero L2 norm"},
		{[][]float32{{1, float32(math.Inf(1))}}, "finite, nonzero L2 norm"},
		{[][]float32{{float32(math.NaN()), 1}}, "finite, nonzero L2 norm"},
	} {
		vectors = tc.vectors
		if resp, err := embedder.Embed(ctx, request); err == nil || !strings.Contains(err.Error(), tc.wantErr) || resp != nil {
			t.Errorf("raw vectors %v: response %v, err %v; want %q", vectors, resp, err, tc.wantErr)
		}
	}
	vectors = [][]float32{{math.MaxFloat32, 0}}
	got, err := genkit.Embed(ctx, g, ai.WithEmbedderName("yzma/local"), ai.WithDocs(request.Input...))
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff([]*ai.Embedding{{Embedding: []float32{1, 0}}}, got.Embeddings); diff != "" {
		t.Errorf("embeddings (-want +got):\n%s", diff)
	}
	if loads != 1 {
		t.Errorf("loaded %d times, want 1", loads)
	}
}

func TestNativeDiscoveredEmbedding(t *testing.T) {
	path, lib := os.Getenv("YZMA_TEST_MODEL"), os.Getenv("YZMA_LIB")
	if path == "" || lib == "" || os.Getenv("YZMA_TEST_TYPE") != "embedding" {
		t.Skip("set YZMA_TEST_MODEL, YZMA_LIB, and YZMA_TEST_TYPE=embedding to run native embeddings")
	}
	path, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	name := strings.TrimSuffix(strings.ToLower(filepath.Base(path)), ".gguf")
	plugin := &Yzma{ModelsDir: filepath.Dir(path), LibPath: lib}
	ctx := t.Context()
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	t.Cleanup(func() {
		if err := plugin.Close(context.WithoutCancel(ctx)); err != nil {
			t.Error(err)
		}
	})
	const dims = 384 // BGE-small-en-v1.5.
	qualified := api.NewName(provider, name)
	listed := false
	for _, action := range plugin.ListActions(ctx) {
		if action.Name == qualified && action.Type == api.ActionTypeEmbedder &&
			action.Metadata["info"].(map[string]any)["dimensions"] == dims {
			listed = true
		}
	}
	if !listed {
		t.Fatalf("catalog did not list %q as a %d-dimensional embedder", qualified, dims)
	}
	first := ai.DocumentFromText("Rivers flow toward the ocean.", nil)
	second := ai.DocumentFromText("A needle and thread mend a torn shirt.", nil)
	embed := func(name string, docs ...*ai.Document) (*ai.EmbedResponse, error) {
		return genkit.Embed(ctx, g, ai.WithEmbedderName(name), ai.WithDocs(docs...))
	}
	resp, err := embed(qualified, first, second)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Embeddings) != 2 {
		t.Fatalf("got %d embeddings, want 2", len(resp.Embeddings))
	}
	for i, doc := range []*ai.Document{first, second} {
		vector := resp.Embeddings[i].Embedding
		checkNativeVector(t, vector, dims)
		single, err := embed(qualified, doc)
		if err != nil {
			t.Fatal(err)
		}
		if len(single.Embeddings) != 1 {
			t.Fatalf("got %d embeddings for a single document, want 1", len(single.Embeddings))
		}
		if diff := cmp.Diff(single.Embeddings[0].Embedding, vector, cmpopts.EquateApprox(0, 1e-6)); diff != "" {
			t.Errorf("document %d single versus batch (-want +got):\n%s", i, diff)
		}
	}
	if cmp.Equal(resp.Embeddings[0].Embedding, resp.Embeddings[1].Embedding, cmpopts.EquateApprox(0, 1e-6)) {
		t.Error("distinct documents returned the same vector")
	}
	oversize := ai.DocumentFromText(strings.Repeat("token ", 600), nil)
	if got, err := embed(qualified, first, oversize); err == nil || !strings.Contains(err.Error(), "document 1") ||
		!strings.Contains(err.Error(), "embedding context/batch limit") || got != nil {
		t.Errorf("oversize document: response %v, err %v; want context/batch limit error", got, err)
	}

	// Verify the explicit pooling path too, without implying that BGE uses mean pooling.
	plugin.DefineEmbedder(g, EmbeddingDefinition{
		Name: "bge-mean", Loading: ModelOptions{GGUFPath: path}, Pooling: llama.PoolingTypeMean, Dimensions: dims,
	})
	mean, err := embed("yzma/bge-mean", first)
	if err != nil {
		t.Fatal(err)
	}
	if len(mean.Embeddings) != 1 {
		t.Fatalf("got %d explicit mean pooled embeddings, want 1", len(mean.Embeddings))
	}
	checkNativeVector(t, mean.Embeddings[0].Embedding, dims)
	if cmp.Equal(resp.Embeddings[0].Embedding, mean.Embeddings[0].Embedding, cmpopts.EquateApprox(0, 1e-6)) {
		t.Error("CLS and mean pooling returned the same vector")
	}
}

func checkNativeVector(t *testing.T, vector []float32, dims int) {
	t.Helper()
	if len(vector) != dims {
		t.Fatalf("got %d dimensions, want %d", len(vector), dims)
	}
	var sum float64
	for _, v := range vector {
		sum += float64(v) * float64(v)
	}
	if norm := math.Sqrt(sum); math.IsNaN(norm) || math.Abs(norm-1) > 1e-6 {
		t.Fatalf("embedding L2 norm = %v, want 1", norm)
	}
}
