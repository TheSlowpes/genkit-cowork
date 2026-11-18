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
	"fmt"
	"math"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/hybridgroup/yzma/pkg/llama"
)

// EmbeddingDefinition registers a local text embedder with explicit pooling and
// output dimensions. Use it to override a catalog entry or to select an
// BERT GGUF whose metadata does not identify its role.
type EmbeddingDefinition struct {
	Name    string         `json:"name"`
	Loading ModelOptions   `json:"loading"`
	Context ContextOptions `json:"context,omitempty"`
	// Pooling must be llama.PoolingTypeCLS or llama.PoolingTypeMean.
	Pooling llama.PoolingType `json:"pooling"`
	// Dimensions must match the model's native output size.
	Dimensions int `json:"dimensions"`
}

type embeddingSpec struct {
	pooling    llama.PoolingType
	dimensions int
}

func (m ModelDefinition) actionType() api.ActionType {
	if m.embedding.dimensions > 0 {
		return api.ActionTypeEmbedder
	}
	return api.ActionTypeModel
}

func (s embeddingSpec) options(name string) *ai.EmbedderOptions {
	return &ai.EmbedderOptions{
		Label:      provider + " - " + name,
		Dimensions: s.dimensions,
		Supports:   &ai.EmbedderSupports{Input: []string{"text"}},
	}
}

func isEmbeddingPooling(pooling llama.PoolingType) bool {
	return pooling == llama.PoolingTypeCLS || pooling == llama.PoolingTypeMean
}

func discoverEmbedding(values map[string]ggufValue, tensors map[string]bool) (embeddingSpec, bool) {
	// Older BERT converters, including the tested BGE file, omit general.type.
	// Require encoder, tokenizer, and pooling evidence rather than guessing
	// from the filename or treating an arbitrary hidden state as an embedding.
	typ, hasType := values["general.type"]
	_, hasProjector := values["clip.projector_type"]
	pooling, dims := values["bert.pooling_type"], values["bert.embedding_length"]
	tokens := values["tokenizer.ggml.tokens"]
	if (hasType && typ != (ggufValue{kind: ggufString, text: "model"})) || hasProjector ||
		values["general.architecture"].text != "bert" ||
		values["bert.attention.causal"] != (ggufValue{kind: 7}) ||
		values["tokenizer.ggml.model"].text != "bert" ||
		tokens.kind != ggufArray || tokens.element != ggufString || tokens.num == 0 ||
		!tensors["token_embd.weight"] || !tensors["blk.0.attn_q.weight"] ||
		!tensors["blk.0.layer_output_norm.weight"] || tensors["cls.weight"] || tensors["cls.output.weight"] ||
		pooling.kind != 4 || !isEmbeddingPooling(llama.PoolingType(pooling.num)) ||
		dims.kind != 4 || dims.num == 0 || dims.num > math.MaxInt32 {
		return embeddingSpec{}, false
	}
	return embeddingSpec{pooling: llama.PoolingType(pooling.num), dimensions: int(dims.num)}, true
}

// DefineEmbedder validates and snapshots an explicit registration. It panics
// for invalid configuration, duplicate names, or an unavailable plugin. Define
// an override before any lookup resolves the discovered name. Weights load on
// Preload or the first embedding request, not on registration.
func (y *Yzma) DefineEmbedder(g *genkit.Genkit, def EmbeddingDefinition) ai.Embedder {
	y.mu.Lock()
	defer y.mu.Unlock()
	if !y.initted || y.closed {
		panic("yzma.DefineEmbedder: plugin is not initialized or is closed")
	}
	if strings.TrimSpace(def.Name) == "" {
		panic("yzma.DefineEmbedder: name must not be empty")
	}
	if y.models[def.Name] != nil {
		panic("yzma.DefineEmbedder: model already defined: " + def.Name)
	}
	if !isEmbeddingPooling(def.Pooling) || def.Dimensions <= 0 || def.Dimensions > math.MaxInt32 {
		panic("yzma.DefineEmbedder: CLS or mean pooling and positive output dimensions are required")
	}
	if _, err := applyModelOptions(def.Loading, llama.ModelParams{}); err != nil {
		panic("yzma.DefineEmbedder: " + err.Error())
	}
	if _, err := applyContextOptions(def.Context, llama.ContextParams{}); err != nil {
		panic("yzma.DefineEmbedder: " + err.Error())
	}
	spec := embeddingSpec{pooling: def.Pooling, dimensions: def.Dimensions}
	gen := y.newGenerator(cloneModelDefinition(ModelDefinition{
		Name: def.Name, Loading: def.Loading, Context: def.Context, embedding: spec,
	}))
	gen.explicit = true
	embedder := genkit.DefineEmbedder(g, api.NewName(provider, def.Name), spec.options(def.Name), gen.embed)
	y.models[def.Name] = gen
	return embedder
}

// EmbeddingInput contains ordered text documents and the fixed settings for an
// embedder registration.
type EmbeddingInput struct {
	Texts      []string
	Context    ContextOptions
	Pooling    llama.PoolingType
	Dimensions int
}

// TextEmbeddingOperator is the optional embedding extension for a ModelOperator.
// EmbedText returns one raw vector per text, in order. Vectors must remain valid
// after EmbedText returns. The plugin checks their shape and normalizes them.
type TextEmbeddingOperator interface {
	EmbedText(ctx context.Context, model llama.Model, input EmbeddingInput) ([][]float32, error)
}

func (g *generator) embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	texts, err := embeddingTexts(req)
	if err != nil {
		return nil, err
	}
	op, ok := g.operator.(TextEmbeddingOperator)
	if !ok {
		return nil, fmt.Errorf("yzma: the model operator does not support text embeddings")
	}
	if err := g.acquire(ctx); err != nil {
		return nil, err
	}
	defer g.release()
	spec := g.model.embedding
	vectors, err := op.EmbedText(ctx, g.loaded, EmbeddingInput{
		Texts: texts, Context: g.model.Context, Pooling: spec.pooling, Dimensions: spec.dimensions,
	})
	if err != nil {
		return nil, fmt.Errorf("embed with model %q: %w", g.model.Name, err)
	}
	if len(vectors) != len(texts) {
		return nil, fmt.Errorf("yzma: got %d embedding vectors for %d documents", len(vectors), len(texts))
	}
	result := &ai.EmbedResponse{Embeddings: make([]*ai.Embedding, len(vectors))}
	for i, vector := range vectors {
		normalized, err := normalizeEmbedding(vector, spec.dimensions)
		if err != nil {
			return nil, fmt.Errorf("yzma: embedding document %d: %w", i, err)
		}
		result.Embeddings[i] = &ai.Embedding{Embedding: normalized}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

func embeddingTexts(req *ai.EmbedRequest) ([]string, error) {
	if req == nil || len(req.Input) == 0 {
		return nil, fmt.Errorf("yzma: at least one embedding document is required")
	}
	if opts, ok := req.Options.(map[string]any); req.Options != nil && (!ok || len(opts) != 0) {
		return nil, fmt.Errorf("yzma: embedding request options are not supported")
	}
	texts := make([]string, len(req.Input))
	for i, doc := range req.Input {
		if doc == nil {
			return nil, fmt.Errorf("yzma: embedding document %d is nil", i)
		}
		var text strings.Builder
		for _, part := range doc.Content {
			if !part.IsText() {
				return nil, fmt.Errorf("yzma: embedding document %d contains a non-text part", i)
			}
			text.WriteString(part.Text)
		}
		if text.Len() == 0 {
			return nil, fmt.Errorf("yzma: embedding document %d must contain non-empty text", i)
		}
		texts[i] = text.String()
	}
	return texts, nil
}

func normalizeEmbedding(vector []float32, dims int) ([]float32, error) {
	if len(vector) != dims {
		return nil, fmt.Errorf("vector has %d dimensions, want %d", len(vector), dims)
	}
	var sum float64
	for _, value := range vector {
		v := float64(value)
		sum += v * v
	}
	if sum == 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		return nil, fmt.Errorf("vector must have a finite, nonzero L2 norm")
	}
	norm := math.Sqrt(sum)
	result := make([]float32, dims)
	for i, value := range vector {
		result[i] = float32(float64(value) / norm)
	}
	return result, nil
}
