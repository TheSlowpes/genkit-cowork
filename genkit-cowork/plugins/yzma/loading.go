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
	"path/filepath"
	"slices"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
)

var (
	// ErrClosed indicates that plugin shutdown has begun.
	ErrClosed = errors.New("yzma: plugin is closed")
	// ErrGenerationNotImplemented indicates that a custom ModelOperator does
	// not implement TextGenerationOperator.
	ErrGenerationNotImplemented = errors.New("yzma: configured operator does not implement text generation")
)

// ModelOperator abstracts initialization, weight loading, and cleanup.
// Implementations must support calls for different models concurrently.
// LoadModel must return a non-zero handle on success; on error it must free any
// partial allocation and return zero. FreeModel must leave the handle valid on
// error so cleanup can be retried. Methods must honor context cancellation where
// possible; a native call may only observe it at a progress callback.
type ModelOperator interface {
	Init(ctx context.Context, libPath string) error
	LoadModel(ctx context.Context, opts ModelOptions) (llama.Model, error)
	FreeModel(ctx context.Context, model llama.Model) error
}

// nativeRuntime protects Yzma's process-global FFI bindings from reinitialization.
// It intentionally outlives individual plugins and their model weights.
var nativeRuntime struct {
	sync.Mutex
	path        string
	initialized bool
}

type defaultModelOperator struct{}

func (defaultModelOperator) Init(ctx context.Context, libPath string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := filepath.Abs(libPath)
	if err != nil {
		return err
	}
	nativeRuntime.Lock()
	defer nativeRuntime.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if nativeRuntime.initialized {
		if nativeRuntime.path != path {
			return fmt.Errorf("llama.cpp is already initialized from %q, cannot use %q", nativeRuntime.path, path)
		}
		return nil
	}
	if err := llama.Load(path); err != nil {
		return fmt.Errorf("load llama.cpp libraries: %w", err)
	}
	if err := llama.Init(); err != nil {
		llama.Close()
		return fmt.Errorf("initialize llama.cpp backends: %w", err)
	}
	nativeRuntime.path, nativeRuntime.initialized = path, true
	return nil
}

func (defaultModelOperator) LoadModel(ctx context.Context, opts ModelOptions) (llama.Model, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	var model llama.Model
	err := withModelParams(opts, func(params llama.ModelParams) error {
		params.SetProgressCallback(func(float32, uintptr) uint8 {
			if ctx.Err() != nil {
				return 0
			}
			return 1
		})
		defer params.SetProgressCallback(nil)
		if err := ctx.Err(); err != nil {
			return err
		}
		var err error
		model, err = llama.ModelLoadFromFile(opts.GGUFPath, params)
		return err
	})
	if err = errors.Join(err, ctx.Err()); err != nil {
		if model != 0 {
			err = errors.Join(err, llama.ModelFree(model))
		}
		return 0, err
	}
	return model, nil
}

func (defaultModelOperator) FreeModel(ctx context.Context, model llama.Model) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return llama.ModelFree(model)
}

// generator holds one registration's weights. gate protects the handle and
// serializes loading, use, and cleanup. It remains held throughout generation
// so an inference context cannot outlive the weights it references.
type generator struct {
	model    ModelDefinition
	operator ModelOperator
	gate     chan struct{}
	closed   <-chan struct{}
	loaded   llama.Model
}

// acquire loads weights if needed and reserves their use until release.
func (g *generator) acquire(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-g.closed:
		return ErrClosed
	case g.gate <- struct{}{}:
	}
	if err := g.available(ctx); err != nil {
		g.release()
		return err
	}
	if g.loaded == 0 {
		loaded, err := g.operator.LoadModel(ctx, g.model.Loading)
		if err != nil {
			g.release()
			return fmt.Errorf("load model %q: %w", g.model.Name, err)
		}
		if loaded == 0 {
			g.release()
			return fmt.Errorf("load model %q: operator returned a zero handle", g.model.Name)
		}
		g.loaded = loaded
	}
	if err := g.available(ctx); err != nil {
		g.release()
		return err
	}
	return nil
}

func (g *generator) available(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case <-g.closed:
		return ErrClosed
	default:
		return nil
	}
}

func (g *generator) release() {
	<-g.gate
}

// close waits for active use. Failed cleanup retains the handle for a retry.
func (g *generator) close(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case g.gate <- struct{}{}:
	}
	defer g.release()
	if err := ctx.Err(); err != nil {
		return err
	}
	if g.loaded == 0 {
		return nil
	}
	if err := g.operator.FreeModel(ctx, g.loaded); err != nil {
		return err
	}
	g.loaded = 0
	return nil
}

// cloneModelDefinition snapshots caller-owned configuration at registration.
func cloneModelDefinition(model ModelDefinition) ModelDefinition {
	model.Loading.NGPULayers = cloneValue(model.Loading.NGPULayers)
	model.Loading.MainGPU = cloneValue(model.Loading.MainGPU)
	model.Loading.DeviceSelection = slices.Clone(model.Loading.DeviceSelection)
	model.Loading.TensorSplit = slices.Clone(model.Loading.TensorSplit)
	model.Context.ContextSize = cloneValue(model.Context.ContextSize)
	model.Context.LogicalMaximumBatchSize = cloneValue(model.Context.LogicalMaximumBatchSize)
	model.Context.PhysicalMaximumBatchSize = cloneValue(model.Context.PhysicalMaximumBatchSize)
	model.Context.Threads = cloneValue(model.Context.Threads)
	model.Context.BatchThreads = cloneValue(model.Context.BatchThreads)
	if cfg := model.GenerationDefaults; cfg != nil {
		copied := *cfg
		copied.MaxOutputTokens = cloneValue(cfg.MaxOutputTokens)
		copied.Temperature = cloneValue(cfg.Temperature)
		copied.TopK = cloneValue(cfg.TopK)
		copied.TopP = cloneValue(cfg.TopP)
		copied.MinP = cloneValue(cfg.MinP)
		copied.Seed = cloneValue(cfg.Seed)
		copied.RepeatPenalty = cloneValue(cfg.RepeatPenalty)
		copied.FrequencyPenalty = cloneValue(cfg.FrequencyPenalty)
		copied.PresencePenalty = cloneValue(cfg.PresencePenalty)
		copied.PenaltyLastN = cloneValue(cfg.PenaltyLastN)
		if cfg.StopSequences != nil {
			stops := slices.Clone(*cfg.StopSequences)
			copied.StopSequences = &stops
		}
		model.GenerationDefaults = &copied
	}
	return model
}

// cloneValue copies an optional value. The unrestricted constraint permits all
// numeric pointer types used by the public configuration structs.
func cloneValue[T any](value *T) *T {
	if value == nil {
		return nil
	}
	copied := *value
	return &copied
}
