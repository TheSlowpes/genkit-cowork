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

var _ api.DynamicPlugin = (*Yzma)(nil)

// Yzma registers local models backed by Yzma's in-process llama.cpp bindings.
// Init initializes the runtime and catalogs installed models. Neither discovery
// nor DefineModel loads weights. Call Close to free models and media companions.
// Configuration fields must not be mutated after Init.
type Yzma struct {
	LibPath   string // Path to the llama.cpp library. If empty, the value of the environment variable YZMA_LIB will be used.
	ModelsDir string // Directory of installed GGUFs. Empty uses Yzma's default.
	Verbose   bool   // Reserved for future native logging configuration.
	// Operator overrides native initialization and model I/O. Nil uses llama.cpp.
	Operator ModelOperator

	mu       sync.Mutex // Guards the plugin's own state below.
	initted  bool
	closed   bool
	done     chan struct{} // Closed once to reject new model work during shutdown.
	models   map[string]*generator
	operator ModelOperator
	catalog  map[string]ModelDefinition
	loadGate chan struct{} // Shared by manual and discovered models.
}

// IsDefinedModel reports whether Genkit can resolve a model. It does not load weights.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, api.NewName(provider, name)) != nil
}

// Model returns the [ai.Model] with the given name.
// It resolves catalog entries on demand and returns nil for unknown names.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
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
	y.loadGate = make(chan struct{}, 1)
	y.catalog = discoverModels(ctx, y.ModelsDir)
	y.initted = true
	return []api.Action{}
}

// DefineModel validates and snapshots configuration, then registers a lazy model.
// It panics for invalid configuration, duplicate names, or an unavailable plugin.
// Weights are loaded by Preload or the first generation call, not registration.
// Define overrides before any lookup or generation resolves the discovered name.
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
	switch model.Type {
	case ModelTypeChat, ModelTypeCompletion, ModelTypeSpeech, ModelTypeDecision:
	default:
		panic(fmt.Sprintf("yzma.DefineModel: unsupported model type %q", model.Type))
	}
	if err := validateSpeechDefinition(model, y.operator); err != nil {
		panic("yzma.DefineModel: " + err.Error())
	}
	if err := validateDecisionDefinition(model); err != nil {
		panic("yzma.DefineModel: " + err.Error())
	}
	if model.Type == ModelTypeCompletion && (model.ChatTemplate != "" || model.ChatTemplateName != "" || model.ToolCalling) {
		panic("yzma.DefineModel: completion models cannot use chat templates or tools")
	}
	if model.ChatTemplate != "" && model.ChatTemplateName != "" {
		panic("yzma.DefineModel: choose ChatTemplate or ChatTemplateName, not both")
	}
	if model.Projector != nil {
		if err := validateProjectorOptions(model); err != nil {
			panic("yzma.DefineModel: " + err.Error())
		}
	}
	if opts != nil && opts.Supports != nil {
		panic("yzma.DefineModel: capabilities are determined by model type; leave ModelOptions.Supports unset")
	}
	if y.models[model.Name] != nil {
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

	gen := y.newGenerator(cloneModelDefinition(model))
	gen.explicit = true
	// Genkit checks the registry for duplicates here without dynamic resolution.
	registered := genkit.DefineModel(g, api.NewName(provider, model.Name), y.modelMetadata(model, opts), gen.generate)
	y.models[model.Name] = gen
	return registered
}

func (y *Yzma) modelMetadata(model ModelDefinition, opts *ai.ModelOptions) *ai.ModelOptions {
	meta := &ai.ModelOptions{
		Label:    provider + " - " + model.Name,
		Supports: modelSupports(model.Type, model.ToolCalling, y.operator),
	}
	if model.Type != ModelTypeDecision {
		meta.ConfigSchema = core.InferSchemaMap(GenerateContentConfig{})
	}
	switch {
	case model.Type == ModelTypeSpeech:
		meta.ConfigSchema = core.InferSchemaMap(SpeechConfig{})
		meta.Supports = &ai.ModelSupports{
			Output: []string{ai.OutputFormatMedia}, ContentType: []string{speechMIME},
		}
	case model.Projector != nil:
		_, meta.Supports.Media = y.operator.(MediaGenerationOperator)
		meta.Supports.Constrained = ai.ConstrainedSupportNone
		meta.Supports.Output = []string{"text"}
		meta.Supports.ContentType = []string{"text/plain"}
		if _, ok := y.operator.(ConstrainedMediaGenerationOperator); ok && model.Projector.Images {
			meta.Supports.Constrained = ai.ConstrainedSupportNoTools
			meta.Supports.Output = append(meta.Supports.Output, ai.OutputFormatJSON)
			meta.Supports.ContentType = append(meta.Supports.ContentType, "application/json")
		}
	}
	if opts != nil {
		if opts.Label != "" {
			meta.Label = opts.Label
		}
		meta.Stage = opts.Stage
		meta.Versions = slices.Clone(opts.Versions)
	}
	return meta
}

func (y *Yzma) newGenerator(model ModelDefinition) *generator {
	return &generator{
		model: model, operator: y.operator, closed: y.done,
		gate: make(chan struct{}, 1), loadGate: y.loadGate,
	}
}

func (g *generator) generate(ctx context.Context, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	if g.model.Type == ModelTypeSpeech {
		return g.generateSpeech(ctx, input, cb)
	}
	if g.model.Type == ModelTypeDecision {
		return g.generateDecision(ctx, input, cb)
	}
	if g.model.Projector != nil {
		return g.generateMedia(ctx, input, cb)
	}
	constraint, err := prepareConstraint(input)
	if err != nil {
		return nil, err
	}
	messageInput, toolCalling := input, g.model.ToolCalling
	var constrainedOperator ConstrainedTextGenerationOperator
	if constraint != nil {
		if cb != nil {
			return nil, fmt.Errorf("yzma: constrained output does not support streaming")
		}
		var ok bool
		constrainedOperator, ok = g.operator.(ConstrainedTextGenerationOperator)
		if !ok {
			return nil, fmt.Errorf("yzma: the model operator does not support constrained output")
		}
		// A constrained turn uses plain text messages even on a tool-capable
		// registration. This rejects tool history and skips tool formatting.
		textInput := *input
		textInput.Output = nil
		messageInput, toolCalling = &textInput, false
	}
	messages, err := generationMessages(messageInput, toolCalling)
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
	if constraint != nil && len(cfg.StopSequences) > 0 {
		return nil, fmt.Errorf("yzma: constrained output does not support stop sequences")
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
	if cb != nil && !toolCalling {
		onText = func(ctx context.Context, text string) error {
			return cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart(text)}})
		}
	}
	generation := TextGenerationInput{
		Type: g.model.Type, Messages: messages, ChatTemplate: g.model.ChatTemplate,
		ChatTemplateName: g.model.ChatTemplateName, ToolCalling: toolCalling, Tools: tools,
		Context: g.model.Context, Sampler: cfg.Sampler,
		MaxOutputTokens: cfg.MaxOutputTokens, StopSequences: cfg.StopSequences,
	}
	var result TextGenerationResult
	if constraint != nil {
		result, err = constrainedOperator.GenerateConstrainedText(ctx, g.loaded, generation, constraint.grammar)
	} else {
		result, err = operator.GenerateText(ctx, g.loaded, generation, onText)
	}
	if err != nil {
		return nil, fmt.Errorf("generate with model %q: %w", g.model.Name, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if constraint != nil {
		result.Text, err = constraint.outputText(result)
		if err != nil {
			return nil, err
		}
	}
	parts := []*ai.Part{ai.NewTextPart(result.Text)}
	if toolCalling {
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
	return generationResponse(input, result, parts), nil
}

func generationResponse(input *ai.ModelRequest, result TextGenerationResult, parts []*ai.Part) *ai.ModelResponse {
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
	}
}

// Preload loads a registered model's weights and any media companion without inference.
// Concurrent calls share a successful load. A failed load can be retried.
// name is the unprefixed name supplied to DefineModel or resolved through Model.
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

// Close rejects new work, waits for active use, and frees media companions and weights.
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

func modelSupports(modelType ModelType, toolCalling bool, operator ModelOperator) *ai.ModelSupports {
	if modelType == ModelTypeDecision {
		return &ai.ModelSupports{
			// Genkit counts the optional system message as a turn.
			Multiturn: true, SystemRole: true, Constrained: ai.ConstrainedSupportNoTools,
			Output: []string{ai.OutputFormatJSON}, ContentType: []string{"application/json"},
		}
	}
	supports := &ai.ModelSupports{
		Multiturn:   modelType == ModelTypeChat,
		SystemRole:  modelType == ModelTypeChat,
		Tools:       modelType == ModelTypeChat && toolCalling,
		ToolChoice:  modelType == ModelTypeChat && toolCalling,
		Output:      []string{"text"},
		ContentType: []string{"text/plain"},
	}
	if _, ok := operator.(ConstrainedTextGenerationOperator); ok {
		supports.Constrained = ai.ConstrainedSupportNoTools
		supports.Output = append(supports.Output, ai.OutputFormatJSON)
		supports.ContentType = append(supports.ContentType, "application/json")
	}
	return supports
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
