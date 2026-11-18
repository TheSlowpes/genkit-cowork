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
	"fmt"
	"math"
	"runtime"
	"slices"

	"github.com/hybridgroup/yzma/pkg/llama"
)

func (defaultModelOperator) EmbedText(ctx context.Context, model llama.Model, input EmbeddingInput) (_ [][]float32, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if arch, _ := llama.ModelMetaValStr(model, "general.architecture"); arch != "bert" {
		return nil, fmt.Errorf("only BERT text embedders are supported, got architecture %q", arch)
	}
	dims := llama.ModelNEmbdOut(model)
	if int(dims) != input.Dimensions {
		return nil, fmt.Errorf("model reports %d output dimensions, registered as %d", dims, input.Dimensions)
	}
	params, err := resolveContextParams(input.Context)
	if err != nil {
		return nil, err
	}
	// By default the embedding context fits the model's trained sequence size.
	if input.Context.ContextSize == nil {
		params.NCtx = 0
	}
	params.PoolingType = input.Pooling
	params.AttentionType = llama.AttentionTypeNonCausal
	params.Embeddings = 1
	lctx, err := llama.InitFromModel(model, params)
	if err != nil {
		return nil, fmt.Errorf("allocate embedding context: %w", err)
	}
	llama.SetAbortCallback(lctx, func() bool { return ctx.Err() != nil })
	defer func() {
		llama.SetAbortCallback(lctx, nil)
		err = errors.Join(err, llama.Free(lctx))
	}()
	if pooling := llama.GetPoolingType(lctx); pooling != input.Pooling {
		return nil, fmt.Errorf("native pooling type %d does not match configured type %d", pooling, input.Pooling)
	}
	// The encoder must see the entire sequence in one microbatch so attention
	// and pooling include all the tokens. We cannot split or truncate it.
	limit := min(int64(llama.NCtxSeq(lctx)), int64(llama.NBatch(lctx)), int64(llama.NUBatch(lctx)),
		int64(llama.ModelNCtxTrain(model)))
	vocab := llama.ModelGetVocab(model)
	memory, err := llama.GetMemory(lctx)
	if err != nil {
		return nil, err
	}
	vectors := make([][]float32, len(input.Texts))
	for i, text := range input.Texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(text) > math.MaxInt32 {
			return nil, fmt.Errorf("document %d exceeds the tokenizer byte limit", i)
		}
		tokens := llama.Tokenize(vocab, text, true, false)
		if len(tokens) == 0 || int64(len(tokens)) > limit {
			return nil, fmt.Errorf("document %d has %d tokens; embedding context/batch limit is %d", i, len(tokens), limit)
		}
		if i > 0 && memory != 0 {
			if err := llama.MemoryClear(memory, true); err != nil {
				return nil, fmt.Errorf("reset context before document %d: %w", i, err)
			}
		}
		// llama.cpp routes BERT through Decode, even though the model
		// architecture is a bidirectional encoder.
		status, err := llama.Decode(lctx, llama.BatchGetOne(tokens))
		runtime.KeepAlive(tokens)
		if err := errors.Join(err, ctx.Err()); err != nil {
			return nil, fmt.Errorf("encode document %d: %w", i, err)
		}
		if status != 0 {
			return nil, fmt.Errorf("encode document %d: llama.cpp returned status %d", i, status)
		}
		vector, err := llama.GetEmbeddingsSeq(lctx, 0, dims)
		if err != nil {
			return nil, fmt.Errorf("read embedding for document %d: %w", i, err)
		}
		// Yzma returns native memory that the next Encode or Free can invalidate.
		vectors[i] = slices.Clone(vector)
	}
	return vectors, nil
}
