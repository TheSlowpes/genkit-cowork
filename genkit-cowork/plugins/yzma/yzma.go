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
	"github.com/firebase/genkit/go/core"
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
	roleMapping = map[ai.Role]string{
		ai.RoleUser:   "user",
		ai.RoleModel:  "assistant",
		ai.RoleSystem: "system",
		ai.RoleTool:   "tool",
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
// Type is required and determines prompt preparation and supported capabilities.
// ModelOptions may customize Label, Stage, and Versions, but not Supports.
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
	if model.Type != ModelTypeChat && model.Type != ModelTypeCompletion {
		panic(fmt.Sprintf("yzma.DefineModel: unsupported model type %q; use chat or completion", model.Type))
	}
	if model.Type == ModelTypeCompletion && (model.ChatTemplate != "" || model.ChatTemplateName != "" || model.ToolCalling) {
		panic("yzma.DefineModel: completion models cannot use chat templates or tools")
	}
	if model.ChatTemplate != "" && model.ChatTemplateName != "" {
		panic("yzma.DefineModel: choose ChatTemplate or ChatTemplateName, not both")
	}
	if opts != nil && opts.Supports != nil {
		panic("yzma.DefineModel: capabilities are determined by model type; leave ModelOptions.Supports unset")
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

	meta := &ai.ModelOptions{
		ConfigSchema: core.InferSchemaMap(GenerateContentConfig{}),
		Label:        provider + " - " + model.Name,
		Supports:     modelSupports(model.Type, model.ToolCalling),
	}
	if opts != nil {
		if opts.Label != "" {
			meta.Label = opts.Label
		}
		meta.Stage = opts.Stage
		meta.Versions = slices.Clone(opts.Versions)
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
	messages, err := generationMessages(input, g.model.ToolCalling)
	if err != nil {
		return nil, err
	}
	if g.model.Type == ModelTypeCompletion && (len(messages) != 1 || messages[0].GetRole() != "user") {
		return nil, fmt.Errorf("yzma: completion models require exactly one user text message; system messages and history are not supported")
	}
	requestConfig, err := decodeGenerationConfig(input.Config)
	if err != nil {
		return nil, err
	}
	cfg, err := resolveGenerationConfig(nil, g.model.GenerationDefaults, requestConfig)
	if err != nil {
		return nil, err
	}
	tools, err := convertTools(input.Tools)
	if err != nil {
		return nil, err
	}
	if input.ToolChoice == ai.ToolChoiceNone {
		tools = nil
	}
	if err := g.acquire(ctx); err != nil {
		return nil, err
	}
	defer g.release()
	operator, ok := g.operator.(TextGenerationOperator)
	if !ok {
		return nil, ErrGenerationNotImplemented
	}
	var onText func(context.Context, string) error
	if cb != nil && !g.model.ToolCalling {
		onText = func(ctx context.Context, text string) error {
			return cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart(text)}})
		}
	}
	result, err := operator.GenerateText(ctx, g.loaded, TextGenerationInput{
		Type: g.model.Type, Messages: messages, ChatTemplate: g.model.ChatTemplate,
		ChatTemplateName: g.model.ChatTemplateName, ToolCalling: g.model.ToolCalling, Tools: tools,
		Context: g.model.Context, Sampler: cfg.Sampler,
		MaxOutputTokens: cfg.MaxOutputTokens, StopSequences: cfg.StopSequences,
	}, onText)
	if err != nil {
		return nil, fmt.Errorf("generate with model %q: %w", g.model.Name, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	parts := []*ai.Part{ai.NewTextPart(result.Text)}
	if g.model.ToolCalling {
		parts, err = parsedToolParts(result, input.Tools, input.ToolChoice)
		if err != nil {
			return nil, err
		}
		if cb != nil {
			if err := cb(ctx, &ai.ModelResponseChunk{Content: parts}); err != nil {
				return nil, err
			}
		}
	}
	reason := ai.FinishReasonStop
	if result.Limited {
		reason = ai.FinishReasonLength
	}
	return &ai.ModelResponse{
		Message: ai.NewModelMessage(parts...), Request: input,
		FinishReason: reason,
		Usage: &ai.GenerationUsage{
			InputTokens: result.InputTokens, OutputTokens: result.OutputTokens,
			TotalTokens: result.InputTokens + result.OutputTokens,
		},
	}, nil
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

func modelSupports(modelType ModelType, toolCalling bool) *ai.ModelSupports {
	return &ai.ModelSupports{
		Multiturn:   modelType == ModelTypeChat,
		SystemRole:  modelType == ModelTypeChat,
		Tools:       modelType == ModelTypeChat && toolCalling,
		ToolChoice:  modelType == ModelTypeChat && toolCalling,
		Output:      []string{"text"},
		ContentType: []string{"text/plain"},
	}
}

func convertTools(tools []*ai.ToolDefinition) ([]message.ToolDefinition, error) {
	yzmaTools := make([]message.ToolDefinition, len(tools))
	seen := make(map[string]bool, len(tools))
	for i, tool := range tools {
		if tool == nil || strings.TrimSpace(tool.Name) == "" || strings.ContainsRune(tool.Name, '\x00') {
			return nil, fmt.Errorf("yzma: tools require non-empty names without NUL bytes")
		}
		if seen[tool.Name] {
			return nil, fmt.Errorf("yzma: duplicate tool %q", tool.Name)
		}
		seen[tool.Name] = true
		schema := tool.InputSchema
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false}
		}
		parameters, err := toolTemplateSchema(schema, schema, 0)
		if err != nil {
			return nil, fmt.Errorf("tool %q: %w", tool.Name, err)
		}
		if parameters["type"] != "object" {
			return nil, fmt.Errorf("tool %q: tool inputs must use an object schema", tool.Name)
		}
		yzmaTools[i] = message.ToolDefinition{
			Type: "function",
			Function: message.ToolFunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  parameters,
			},
		}
	}
	return yzmaTools, nil
}
