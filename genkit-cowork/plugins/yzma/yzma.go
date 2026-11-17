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
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
)

const (
	provider = "yzma"
	libEnv   = "YZMA_LIB"
)

var (
	mediaSupportedModels = []string{}
	toolsSupportedModels = []string{}
	roleMapping          = map[ai.Role]string{
		ai.RoleUser:   "user",
		ai.RoleModel:  "assistant",
		ai.RoleSystem: "system",
		ai.RoleTool:   "tool",
	}
	defaultYzmaSupports = ai.ModelSupports{
		Multiturn:  true,
		Media:      false,
		Tools:      false,
		SystemRole: true,
	}
)

var _ api.Plugin = (*Yzma)(nil)

// Yzma registers local models backed by Yzma's in-process llama.cpp bindings.
// Init initializes the runtime; DefineModel registers models without loading
// weights. Call Close to free loaded models before discarding the plugin.
// Configuration fields must not be mutated after Init.
type Yzma struct {
	LibPath   string // Path to the llama.cpp library. If empty, the value of the environment variable YZMA_LIB will be used.
	ModelsDir string // Reserved for future GGUF discovery; models must be defined explicitly.
	Verbose   bool   // Reserved for future native logging configuration.
	// Operator overrides native initialization and model I/O. Nil uses llama.cpp.
	Operator ModelOperator

	mu       sync.Mutex // Guards the plugin's own state below.
	initted  bool
	closed   bool
	done     chan struct{} // Closed once to reject new model work during shutdown.
	models   map[string]*generator
	operator ModelOperator

	// The capabilities cache has its own lock: the discovery goroutines write
	// to it whil DefineModel reads it holding mu, so sharing one lock would
	// mean dropping and retaking mu mid-function.
	capMu           sync.Mutex
	capabilitesCach map[string]modelCapabilitiesCacheEntry
}

// IsDefinedModel reports whether a model is registered with Genkit.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, api.NewName(provider, name)) != nil
}

// Model returns the [ai.Model] with the given name.
// It returns nil if the model was not configured.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}

// modelCapabilitiesCacheEntry is one model's detected capabilities.
type modelCapabilitiesCacheEntry struct {
}

// Name returns the Genkit provider name.
func (y *Yzma) Name() string {
	return provider
}

// Init initializes the shared native runtime without loading any model weights.
// It panics on initialization failure, as required by the Genkit plugin API.
func (y *Yzma) Init(ctx context.Context) []api.Action {
	y.mu.Lock()
	defer y.mu.Unlock()
	if y.initted {
		panic("yzma.Init already called")
	}
	libPath := y.LibPath
	if libPath == "" {
		lib := os.Getenv(libEnv)
		if lib == "" {
			panic("yzma.Init: Yzma requires setting the path to the llama.cpp library")
		}
		libPath = lib
	}
	operator := y.Operator
	if operator == nil {
		operator = defaultModelOperator{}
	}
	if err := operator.Init(ctx, libPath); err != nil {
		panic("yzma.Init: " + err.Error())
	}
	y.operator = operator
	y.models = make(map[string]*generator)
	y.done = make(chan struct{})
	y.initted = true
	return []api.Action{}
}

// DefineModel validates and snapshots configuration, then registers a lazy model.
// It panics for invalid configuration, duplicate names, or an unavailable plugin.
// Weights are loaded by Preload or the first generation call, not registration.
// Generation currently loads weights and returns ErrGenerationNotImplemented.
func (y *Yzma) DefineModel(g *genkit.Genkit, model ModelDefinition, opts *ai.ModelOptions) ai.Model {
	y.mu.Lock()
	defer y.mu.Unlock()
	if !y.initted {
		panic("yzma.Init not called")
	}
	if y.closed {
		panic("yzma.DefineModel: plugin is closed")
	}
	if strings.TrimSpace(model.Name) == "" {
		panic("yzma.DefineModel: name must not be empty")
	}
	if y.models[model.Name] != nil || IsDefinedModel(g, model.Name) {
		panic("yzma.DefineModel: model already defined: " + model.Name)
	}
	if _, err := applyModelOptions(model.Loading, llama.ModelParams{}); err != nil {
		panic("yzma.DefineModel: " + err.Error())
	}
	if _, err := applyContextOptions(model.Context, llama.ContextParams{}); err != nil {
		panic("yzma.DefineModel: " + err.Error())
	}
	if _, err := resolveGenerationConfig(nil, model.GenerationDefaults, nil); err != nil {
		panic("yzma.DefineModel: " + err.Error())
	}

	var modelOpts ai.ModelOptions
	if opts != nil {
		modelOpts = *opts
	} else {
		supports := modelSupportsFromStaticLists(model.Name)
		supports.Tools = model.Type == "chat" && supports.Tools
		modelOpts = ai.ModelOptions{
			Label:    model.Name,
			Supports: supports,
			Versions: []string{},
		}
	}

	meta := &ai.ModelOptions{
		Label:    provider + " - " + model.Name,
		Supports: modelOpts.Supports,
		Versions: []string{},
	}
	gen := &generator{
		model: cloneModelDefinition(model), operator: y.operator,
		gate: make(chan struct{}, 1), closed: y.done,
	}
	registered := genkit.DefineModel(g, api.NewName(provider, model.Name), meta, gen.generate)
	y.models[model.Name] = gen
	return registered
}

func (g *generator) generate(ctx context.Context, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	if err := g.acquire(ctx); err != nil {
		return nil, err
	}
	defer g.release()
	return nil, ErrGenerationNotImplemented
}

// Preload loads a registered model's weights without running generation.
// Concurrent calls share a successful load. A failed load can be retried.
// name is the unprefixed name supplied to DefineModel.
func (y *Yzma) Preload(ctx context.Context, name string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	y.mu.Lock()
	if !y.initted {
		y.mu.Unlock()
		return fmt.Errorf("yzma.Preload: plugin is not initialized")
	}
	if y.closed {
		y.mu.Unlock()
		return ErrClosed
	}
	gen := y.models[name]
	y.mu.Unlock()
	if gen == nil {
		return fmt.Errorf("yzma.Preload: model %q is not defined", name)
	}
	if err := gen.acquire(ctx); err != nil {
		return err
	}
	gen.release()
	return nil
}

// Close rejects new work, waits for active model use, and frees loaded weights.
// If ctx expires or cleanup fails, call Close again to finish releasing resources.
// The process-wide native runtime stays initialized so other plugin instances
// can continue using it. Close does not unload shared libraries or backends.
func (y *Yzma) Close(ctx context.Context) error {
	y.mu.Lock()
	if !y.initted {
		y.mu.Unlock()
		return nil
	}
	if !y.closed {
		y.closed = true
		close(y.done)
	}
	models := make([]*generator, 0, len(y.models))
	for _, gen := range y.models {
		models = append(models, gen)
	}
	y.mu.Unlock()
	var errs []error
	for _, gen := range models {
		if err := gen.close(ctx); err != nil {
			errs = append(errs, fmt.Errorf("close model %q: %w", gen.model.Name, err))
		}
	}
	return errors.Join(errs...)
}

func modelSupportsFromStaticLists(modelName string) *ai.ModelSupports {
	return &ai.ModelSupports{
		Multiturn:  true,
		SystemRole: true,
		Tools:      slices.Contains(toolsSupportedModels, modelName),
		Media:      slices.Contains(mediaSupportedModels, modelName),
	}
}

func convertTools(tools []*ai.ToolDefinition) ([]message.ToolDefinition, error) {
	yzmaTools := make([]message.ToolDefinition, len(tools))
	for _, tool := range tools {
		yzmaTools = append(yzmaTools, message.ToolDefinition{
			Type: "function",
			Function: message.ToolFunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		})
	}
	return yzmaTools, nil
}
