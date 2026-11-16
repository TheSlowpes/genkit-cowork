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

// ModelDefinition describes a registered local model and its resource settings.
// Loading and context options are fixed at registration; generation defaults may
// be overridden by individual requests. These types describe configuration only;
// applying defaults and executing inference are not implemented yet.
type ModelDefinition struct {
	Name string `json:"name"`
	Type string `json:"type"`

	Loading ModelOptions   `json:"loading"`
	Context ContextOptions `json:"context,omitempty"`

	// ChatTemplate overrides the model's embedded chat template when non-empty.
	ChatTemplate string `json:"chatTemplate,omitempty"`
	// GenerationDefaults supplies sampling and output defaults for this model.
	GenerationDefaults *GenerateContentConfig `json:"generationDefaults,omitempty"`
}

// ModelOptions controls weight loading. Changing these options requires reloading
// the model. Nil pointers and empty mode strings leave native defaults unchanged.
type ModelOptions struct {
	// GGUFPath is the path to the model's GGUF file.
	GGUFPath string `json:"ggufPath"`
	// NGPULayers maps to llama.ModelParams.NGpuLayers. Zero selects CPU-only layers.
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

// GenerateContentConfig controls one generation request.
// Nil fields inherit model generation defaults, then provider defaults.
// Pointers preserve explicit zero values, including greedy temperature and seed 0.
// Common Genkit field names are repeated rather than embedding
// ai.GenerationCommonConfig, whose scalar fields cannot distinguish zero from unset.
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
