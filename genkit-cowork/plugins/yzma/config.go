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

// ModelType selects prompt preparation and the model's Genkit capabilities.
// Manual definitions select it explicitly. Discovery selects it from GGUF metadata.
type ModelType string

const (
	// ModelTypeChat formats conversation messages with a chat template.
	ModelTypeChat ModelType = "chat"
	// ModelTypeCompletion tokenizes one user text prompt without chat formatting.
	ModelTypeCompletion ModelType = "completion"
	// ModelTypeSpeech turns one user text prompt into a complete WAV clip.
	ModelTypeSpeech ModelType = "speech"
	// ModelTypeDecision scores typed System One questions about a state.
	ModelTypeDecision ModelType = "decision"
)

// ModelDefinition describes a registered local model and its resource settings.
// Loading and context options are fixed at registration; generation defaults may
// be overridden by individual requests. Registration validates and snapshots the
// configuration. Text inference allocates a context and sampler per request.
// Decision inference keeps its context with the lazily loaded weights.
type ModelDefinition struct {
	Name string `json:"name"`
	// Type is required.
	Type ModelType `json:"type"`

	Loading ModelOptions   `json:"loading"`
	Context ContextOptions `json:"context,omitempty"`
	// Projector enables inline media on a chat model. Set it explicitly
	// when discovery cannot identify a unique compatible projector.
	Projector *ProjectorOptions `json:"projector,omitempty"`

	// SpeechDecoderPath is the companion mmproj GGUF for ModelTypeSpeech.
	// Leave it empty for other model types.
	SpeechDecoderPath string `json:"speechDecoderPath,omitempty"`

	// Decision selects the readout for ModelTypeDecision.
	Decision DecisionOptions `json:"decision,omitempty"`

	// ChatTemplate overrides the embedded template for chat models.
	// Completion registrations must leave this empty.
	ChatTemplate string `json:"chatTemplate,omitempty"`
	// ChatTemplateName selects a named GGUF template, such as "tool_use".
	// Leave empty to use the default template. It cannot accompany ChatTemplate.
	ChatTemplateName string `json:"chatTemplateName,omitempty"`
	// ToolCalling enables the tool adapter for a chat model trained to use tools.
	// The plugin does not infer this capability from filenames or model families.
	ToolCalling bool `json:"toolCalling,omitempty"`
	// GenerationDefaults supplies sampling and output defaults for text models.
	GenerationDefaults *GenerateContentConfig `json:"generationDefaults,omitempty"`

	// Set only for discovered or explicitly defined embedders.
	embedding embeddingSpec
}

// ProjectorOptions identifies a paired GGUF and its enabled input capabilities.
type ProjectorOptions struct {
	GGUFPath string `json:"ggufPath"`
	Images   bool   `json:"images,omitempty"`
	// Audio enables inline mono 16 kHz PCM WAV input.
	Audio bool `json:"audio,omitempty"`
	// Video enables sampled inline MP4 on a compatible vision pair. It does not
	// enable standalone images or audio.
	Video bool `json:"video,omitempty"`
}

// ModelOptions controls weight loading. Changing these options requires reloading
// the model. Nil pointers and empty mode strings leave native defaults unchanged.
type ModelOptions struct {
	// GGUFPath is the path to the model's GGUF file.
	GGUFPath string `json:"ggufPath"`
	// NGPULayers maps to llama.ModelParams.NGpuLayers. Zero keeps layers on the
	// CPU; -1 requests offloading all layers.
	NGPULayers *int32 `json:"nGPULayers,omitempty"`
	// DeviceSelection names backend devices to resolve before loading.
	// An empty list leaves device selection to llama.cpp.
	DeviceSelection []string  `json:"deviceSelection,omitempty"`
	GPUSplitMode    SplitMode `json:"gpuSplitMode,omitempty"`
	// MainGPU is the main GPU index, interpreted according to GPUSplitMode.
	MainGPU *int32 `json:"mainGPU,omitempty"`
	// TensorSplit gives relative weight allocations across selected GPUs.
	TensorSplit []float32 `json:"tensorSplit,omitempty"`
	LoadMode    LoadMode  `json:"loadMode,omitempty"`
	// LazyMode controls native tensor reads, not when the plugin loads a model.
	LazyMode LazyMode `json:"lazyMode,omitempty"`
}

// ContextOptions controls inference-context allocation and execution.
// These are registration settings even if a context is created per request.
// Nil pointers preserve native defaults.
type ContextOptions struct {
	// ContextSize maps to NCtx and includes input and generated tokens.
	// An explicit zero asks llama.cpp to use the model's context size.
	ContextSize *uint32 `json:"contextSize,omitempty"`
	// LogicalMaximumBatchSize maps to NBatch, the logical token batch limit.
	LogicalMaximumBatchSize *uint32 `json:"logicalMaximumBatchSize,omitempty"`
	// PhysicalMaximumBatchSize maps to NUbatch, the physical token microbatch limit.
	PhysicalMaximumBatchSize *uint32 `json:"physicalMaximumBatchSize,omitempty"`
	// Threads maps to NThreads for token generation.
	Threads *int32 `json:"threads,omitempty"`
	// BatchThreads maps to NThreadsBatch for prompt processing.
	BatchThreads *int32 `json:"batchThreads,omitempty"`
}

// GenerateContentConfig is the per-request configuration a Yzma model accepts
// through Genkit's ai.WithConfig. Unset fields inherit model defaults, then
// Yzma sampler defaults. MaxOutputTokens defaults to 1024.
// Pointers distinguish explicit zero values from omitted settings.
type GenerateContentConfig struct {
	// MaxOutputTokens limits generated tokens, not the combined context size.
	MaxOutputTokens *int `json:"maxOutputTokens,omitempty"`
	// StopSequences stops generation at a matching sequence. Nil inherits;
	// a non-nil empty slice explicitly clears inherited stop sequences.
	StopSequences *[]string `json:"stopSequences,omitempty"`
	// Temperature maps to sampler Temp; zero requests greedy sampling.
	Temperature *float64 `json:"temperature,omitempty"`
	// TopK maps to sampler TopK; non-positive values disable this filter.
	TopK *int32 `json:"topK,omitempty"`
	// TopP maps to sampler TopP; 1 disables this filter.
	TopP *float64 `json:"topP,omitempty"`
	// MinP maps to sampler MinP; zero disables this filter.
	MinP *float64 `json:"minP,omitempty"`
	// Seed controls sampling, without guaranteeing reproducibility across backends.
	Seed *uint32 `json:"seed,omitempty"`
	// RepeatPenalty maps to PenaltyRepeat; 1 disables the penalty.
	RepeatPenalty *float64 `json:"repeatPenalty,omitempty"`
	// FrequencyPenalty maps to PenaltyFreq; zero disables the penalty.
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty"`
	// PresencePenalty maps to PenaltyPresent; zero disables the penalty.
	PresencePenalty *float64 `json:"presencePenalty,omitempty"`
	// PenaltyLastN is the token history window; non-positive values disable penalties.
	PenaltyLastN *int32 `json:"penaltyLastN,omitempty"`
}
