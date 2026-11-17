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
	"fmt"
	"math"
	"runtime"
	"slices"
	"strings"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// withModelParams resolves native defaults after llama.cpp has been initialized.
// use must consume the params synchronously and must not retain their pointers.
// Device and tensor-split buffers remain alive until use returns.
func withModelParams(opts ModelOptions, use func(llama.ModelParams) error) error {
	params, err := applyModelOptions(opts, llama.ModelDefaultParams())
	if err != nil {
		return err
	}
	var devices []llama.GGMLBackendDevice
	if len(opts.DeviceSelection) > 0 {
		devices = make([]llama.GGMLBackendDevice, 0, len(opts.DeviceSelection)+1)
		for _, name := range opts.DeviceSelection {
			device := llama.GGMLBackendDeviceByName(name)
			if device == 0 {
				return fmt.Errorf("unknown device %q", name)
			}
			if llama.GGMLBackendDevType(device) == llama.GGMLBackendDeviceTypeCPU {
				return fmt.Errorf("device %q is a CPU; use nGPULayers: 0 instead", name)
			}
			devices = append(devices, device)
		}
		devices = append(devices, 0)
		if err := params.SetDevices(devices); err != nil {
			return err
		}
	}
	// llama.cpp reads a full MaxDevices-sized array, even when only a few
	// proportions were supplied. Pad it instead of exposing the short input.
	var tensorSplit []float32
	if len(opts.TensorSplit) > 0 {
		maxDevices := llama.MaxDevices()
		if uint64(len(opts.TensorSplit)) > maxDevices {
			return fmt.Errorf("tensorSplit has %d entries, maximum is %d", len(opts.TensorSplit), maxDevices)
		}
		tensorSplit = make([]float32, maxDevices)
		copy(tensorSplit, opts.TensorSplit)
		params.TensorSplit = &tensorSplit[0]
	}
	defer func() {
		runtime.KeepAlive(devices)
		runtime.KeepAlive(tensorSplit)
	}()
	return use(params)
}

// applyModelOptions overlays scalar options without calling the native runtime.
// Pointer-backed arrays are bound only by withModelParams.
func applyModelOptions(opts ModelOptions, params llama.ModelParams) (llama.ModelParams, error) {
	if strings.TrimSpace(opts.GGUFPath) == "" || strings.ContainsRune(opts.GGUFPath, '\x00') {
		return params, fmt.Errorf("ggufPath must be a non-empty path without NUL bytes")
	}
	var err error
	if params.SplitMode, err = resolveSplitMode(opts.GPUSplitMode, params.SplitMode); err != nil {
		return params, err
	}
	if params.LoadMode, err = resolveLoadMode(opts.LoadMode, params.LoadMode); err != nil {
		return params, err
	}
	if params.LazyMode, err = resolveLazyMode(opts.LazyMode, params.LazyMode); err != nil {
		return params, err
	}
	if opts.NGPULayers != nil {
		if *opts.NGPULayers < -1 {
			return params, fmt.Errorf("nGPULayers must be -1 or non-negative")
		}
		params.NGpuLayers = *opts.NGPULayers
	}
	if opts.MainGPU != nil {
		if *opts.MainGPU < 0 {
			return params, fmt.Errorf("mainGPU must be non-negative")
		}
		params.MainGpu = *opts.MainGPU
	}
	seen := make(map[string]bool, len(opts.DeviceSelection))
	for _, name := range opts.DeviceSelection {
		if strings.TrimSpace(name) == "" || strings.ContainsRune(name, '\x00') || seen[name] {
			return params, fmt.Errorf("deviceSelection contains an empty, duplicate, or invalid name %q", name)
		}
		seen[name] = true
	}
	hasWeight := false
	for _, weight := range opts.TensorSplit {
		if math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) || weight < 0 {
			return params, fmt.Errorf("tensorSplit weights must be finite and non-negative")
		}
		hasWeight = hasWeight || weight > 0
	}
	if len(opts.TensorSplit) > 0 && !hasWeight {
		return params, fmt.Errorf("tensorSplit must contain a positive weight")
	}
	return params, nil
}

// resolveContextParams requires the native runtime to be initialized.
func resolveContextParams(opts ContextOptions) (llama.ContextParams, error) {
	return applyContextOptions(opts, llama.ContextDefaultParams())
}

// applyContextOptions overlays context options without allocating a context.
func applyContextOptions(opts ContextOptions, params llama.ContextParams) (llama.ContextParams, error) {
	if opts.ContextSize != nil {
		params.NCtx = *opts.ContextSize
	}
	if opts.LogicalMaximumBatchSize != nil {
		if *opts.LogicalMaximumBatchSize == 0 {
			return params, fmt.Errorf("logicalMaximumBatchSize must be positive")
		}
		params.NBatch = *opts.LogicalMaximumBatchSize
	}
	if opts.PhysicalMaximumBatchSize != nil {
		if *opts.PhysicalMaximumBatchSize == 0 {
			return params, fmt.Errorf("physicalMaximumBatchSize must be positive")
		}
		params.NUbatch = *opts.PhysicalMaximumBatchSize
	}
	if opts.Threads != nil {
		if *opts.Threads < 0 {
			return params, fmt.Errorf("threads must be non-negative")
		}
		params.NThreads = *opts.Threads
	}
	if opts.BatchThreads != nil {
		if *opts.BatchThreads < 0 {
			return params, fmt.Errorf("batchThreads must be non-negative")
		}
		params.NThreadsBatch = *opts.BatchThreads
	}
	return params, nil
}

// resolvedGenerationConfig owns per-request sampler and stopping settings.
// The generation loop must use greedy sampling when Sampler.Temp is zero.
type resolvedGenerationConfig struct {
	Sampler         llama.SamplerParams
	MaxOutputTokens int
	StopSequences   []string
}

// resolveGenerationConfig applies provider, model, then request settings to fresh
// Yzma sampler defaults. Output defaults to 1024 tokens until overridden.
// This does not construct a sampler chain or invoke inference.
func resolveGenerationConfig(provider, model, request *GenerateContentConfig) (resolvedGenerationConfig, error) {
	result := resolvedGenerationConfig{
		Sampler:         *llama.DefaultSamplerParams(),
		MaxOutputTokens: 1024,
	}
	for _, cfg := range []*GenerateContentConfig{provider, model, request} {
		if cfg == nil {
			continue
		}
		if cfg.MaxOutputTokens != nil {
			if *cfg.MaxOutputTokens <= 0 {
				return resolvedGenerationConfig{}, fmt.Errorf("maxOutputTokens must be positive")
			}
			result.MaxOutputTokens = *cfg.MaxOutputTokens
		}
		if cfg.StopSequences != nil {
			for _, stop := range *cfg.StopSequences {
				if stop == "" {
					return resolvedGenerationConfig{}, fmt.Errorf("stopSequences must not contain an empty string")
				}
			}
			result.StopSequences = slices.Clone(*cfg.StopSequences)
		}
		if cfg.TopK != nil {
			result.Sampler.TopK = *cfg.TopK
		}
		if cfg.Seed != nil {
			result.Sampler.Seed = *cfg.Seed
		}
		if cfg.PenaltyLastN != nil {
			result.Sampler.PenaltyLastN = *cfg.PenaltyLastN
		}
		for _, field := range []struct {
			name  string
			value *float64
			dst   *float32
		}{
			{"temperature", cfg.Temperature, &result.Sampler.Temp},
			{"topP", cfg.TopP, &result.Sampler.TopP},
			{"minP", cfg.MinP, &result.Sampler.MinP},
			{"repeatPenalty", cfg.RepeatPenalty, &result.Sampler.PenaltyRepeat},
			{"frequencyPenalty", cfg.FrequencyPenalty, &result.Sampler.PenaltyFreq},
			{"presencePenalty", cfg.PresencePenalty, &result.Sampler.PenaltyPresent},
		} {
			if field.value == nil {
				continue
			}
			value := *field.value
			if math.IsNaN(value) || math.IsInf(value, 0) || math.Abs(value) > math.MaxFloat32 {
				return resolvedGenerationConfig{}, fmt.Errorf("%s must be a finite float32 value", field.name)
			}
			if (field.name == "topP" || field.name == "minP") && (value < 0 || value > 1) {
				return resolvedGenerationConfig{}, fmt.Errorf("%s must be between 0 and 1", field.name)
			}
			if (field.name == "temperature" || field.name == "repeatPenalty") && value < 0 {
				return resolvedGenerationConfig{}, fmt.Errorf("%s must be non-negative", field.name)
			}
			*field.dst = float32(value)
		}
	}
	return result, nil
}
