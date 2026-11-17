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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/hybridgroup/yzma/pkg/llama"
)

func ptr[T any](v T) *T { return &v }

func TestResolveModes(t *testing.T) {
	testModes(t, resolveSplitMode, llama.SplitModeLayer, map[SplitMode]llama.SplitMode{
		SplitModeNone: llama.SplitModeNone, SplitModeLayer: llama.SplitModeLayer,
		SplitModeRow: llama.SplitModeRow, SplitModeTensor: llama.SplitModeTensor,
	})
	testModes(t, resolveLoadMode, llama.LoadModeAuto, map[LoadMode]llama.LoadMode{
		LoadModeNone: llama.LoadModeNone, LoadModeAuto: llama.LoadModeAuto,
		LoadModeMmap: llama.LoadModeMmap, LoadModeMlock: llama.LoadModeMlock,
		LoadModeMmapmLock: llama.LoadModeMmapMlock, LoadModeDirectIO: llama.LoadModeDirectIO,
	})
	testModes(t, resolveLazyMode, llama.LazyModeAuto, map[LazyMode]llama.LazyMode{
		LazyModeOff: llama.LazyModeOff, LazyModeAuto: llama.LazyModeAuto, LazyModeOn: llama.LazyModeOn,
	})
}

func testModes[M ~string, N ~int32](t *testing.T, resolve func(M, N) (N, error), fallback N, modes map[M]N) {
	t.Helper()
	modes[""] = fallback
	for mode, want := range modes {
		got, err := resolve(mode, fallback)
		if err != nil || got != want {
			t.Errorf("resolve(%q) = %v, %v; want %v, nil", mode, got, err, want)
		}
	}
	if _, err := resolve("invalid", fallback); err == nil {
		t.Error("unknown mode must return an error")
	}
}

func TestApplyModelOptions(t *testing.T) {
	defaults := llama.ModelParams{
		NGpuLayers: 20, MainGpu: 1, SplitMode: llama.SplitModeLayer,
		LoadMode: llama.LoadModeAuto, LazyMode: llama.LazyModeAuto, CheckTensors: 1,
	}
	got, err := applyModelOptions(ModelOptions{GGUFPath: "model.gguf"}, defaults)
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(defaults, got); diff != "" {
		t.Fatalf("defaults changed (-want +got):\n%s", diff)
	}
	got, err = applyModelOptions(ModelOptions{
		GGUFPath: "model.gguf", NGPULayers: ptr(int32(0)), MainGPU: ptr(int32(0)),
		GPUSplitMode: SplitModeNone, LoadMode: LoadModeNone, LazyMode: LazyModeOff,
	}, defaults)
	if err != nil {
		t.Fatal(err)
	}
	want := defaults
	want.NGpuLayers, want.MainGpu = 0, 0
	want.SplitMode, want.LoadMode, want.LazyMode = llama.SplitModeNone, llama.LoadModeNone, llama.LazyModeOff
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("explicit options (-want +got):\n%s", diff)
	}
}

func TestApplyModelOptionsValidation(t *testing.T) {
	for name, opts := range map[string]ModelOptions{
		"path":        {},
		"NUL path":    {GGUFPath: "a\x00b"},
		"GPU layers":  {GGUFPath: "a", NGPULayers: ptr(int32(-2))},
		"main GPU":    {GGUFPath: "a", MainGPU: ptr(int32(-1))},
		"split mode":  {GGUFPath: "a", GPUSplitMode: "bad"},
		"load mode":   {GGUFPath: "a", LoadMode: "bad"},
		"lazy mode":   {GGUFPath: "a", LazyMode: "bad"},
		"device name": {GGUFPath: "a", DeviceSelection: []string{""}},
		"duplicates":  {GGUFPath: "a", DeviceSelection: []string{"GPU0", "GPU0"}},
		"NUL device":  {GGUFPath: "a", DeviceSelection: []string{"GPU\x00"}},
		"negative":    {GGUFPath: "a", TensorSplit: []float32{-1}},
		"zero split":  {GGUFPath: "a", TensorSplit: []float32{0, 0}},
		"NaN":         {GGUFPath: "a", TensorSplit: []float32{float32(math.NaN())}},
		"infinity":    {GGUFPath: "a", TensorSplit: []float32{float32(math.Inf(1))}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := applyModelOptions(opts, llama.ModelParams{}); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestApplyContextOptions(t *testing.T) {
	defaults := llama.ContextParams{
		NCtx: 4096, NBatch: 512, NUbatch: 128, NThreads: 8, NThreadsBatch: 16,
		Offload_kqv: 1, NSeqMax: 1,
	}
	got, err := applyContextOptions(ContextOptions{}, defaults)
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(defaults, got); diff != "" {
		t.Fatalf("defaults changed (-want +got):\n%s", diff)
	}
	got, err = applyContextOptions(ContextOptions{
		ContextSize: ptr(uint32(0)), LogicalMaximumBatchSize: ptr(uint32(256)),
		PhysicalMaximumBatchSize: ptr(uint32(64)), Threads: ptr(int32(0)), BatchThreads: ptr(int32(2)),
	}, defaults)
	if err != nil {
		t.Fatal(err)
	}
	want := defaults
	want.NCtx, want.NBatch, want.NUbatch, want.NThreads, want.NThreadsBatch = 0, 256, 64, 0, 2
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("explicit options (-want +got):\n%s", diff)
	}
	for name, opts := range map[string]ContextOptions{
		"batch":        {LogicalMaximumBatchSize: ptr(uint32(0))},
		"microbatch":   {PhysicalMaximumBatchSize: ptr(uint32(0))},
		"threads":      {Threads: ptr(int32(-1))},
		"batchThreads": {BatchThreads: ptr(int32(-1))},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := applyContextOptions(opts, defaults); err == nil {
				t.Error("expected validation error")
			}
		})
	}
}

func TestResolveGenerationConfig(t *testing.T) {
	got, err := resolveGenerationConfig(nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(*llama.DefaultSamplerParams(), got.Sampler); diff != "" {
		t.Fatalf("native defaults changed (-want +got):\n%s", diff)
	}
	if got.MaxOutputTokens != 1024 {
		t.Errorf("max tokens = %d, want 1024", got.MaxOutputTokens)
	}
	provider := &GenerateContentConfig{
		Temperature: ptr(0.8), TopK: ptr(int32(20)), TopP: ptr(0.9),
		MaxOutputTokens: ptr(100), StopSequences: ptr([]string{"STOP"}),
	}
	model := &GenerateContentConfig{
		Temperature: ptr(0.4), MaxOutputTokens: ptr(200), MinP: ptr(0.1),
		RepeatPenalty: ptr(1.2), PenaltyLastN: ptr(int32(32)),
		FrequencyPenalty: ptr(-0.5), PresencePenalty: ptr(0.2),
	}
	request := &GenerateContentConfig{
		Temperature: ptr(0.0), Seed: ptr(uint32(0)), MinP: ptr(0.0),
		StopSequences: ptr([]string{}),
	}
	got, err = resolveGenerationConfig(provider, model, request)
	if err != nil {
		t.Fatal(err)
	}
	wantSampler := *llama.DefaultSamplerParams()
	wantSampler.Temp, wantSampler.Seed, wantSampler.TopK = 0, 0, 20
	wantSampler.TopP, wantSampler.MinP = 0.9, 0
	wantSampler.PenaltyRepeat, wantSampler.PenaltyLastN = 1.2, 32
	wantSampler.PenaltyFreq, wantSampler.PenaltyPresent = -0.5, 0.2
	if diff := cmp.Diff(wantSampler, got.Sampler); diff != "" {
		t.Errorf("merged sampler (-want +got):\n%s", diff)
	}
	if got.MaxOutputTokens != 200 || len(got.StopSequences) != 0 {
		t.Errorf("loop settings = %d, %v; want 200, empty", got.MaxOutputTokens, got.StopSequences)
	}
	// Resolving again must inherit the original model and provider values.
	next, err := resolveGenerationConfig(provider, model, nil)
	if err != nil {
		t.Fatal(err)
	}
	next.StopSequences[0] = "changed"
	next.Sampler.DrySequenceBreakers[0] = "changed"
	again, err := resolveGenerationConfig(provider, model, nil)
	if err != nil {
		t.Fatal(err)
	}
	if again.StopSequences[0] != "STOP" || *model.Temperature != 0.4 ||
		again.Sampler.DrySequenceBreakers[0] == "changed" {
		t.Error("request state leaked into defaults or another request")
	}
}

func TestResolveGenerationConfigValidation(t *testing.T) {
	for name, cfg := range map[string]GenerateContentConfig{
		"temperature":    {Temperature: ptr(-0.1)},
		"topP":           {TopP: ptr(1.1)},
		"minP":           {MinP: ptr(-0.1)},
		"repeat":         {RepeatPenalty: ptr(-1.0)},
		"NaN":            {Temperature: ptr(math.NaN())},
		"infinity":       {FrequencyPenalty: ptr(math.Inf(1))},
		"overflow":       {PresencePenalty: ptr(math.MaxFloat64)},
		"output limit":   {MaxOutputTokens: ptr(0)},
		"negative limit": {MaxOutputTokens: ptr(-1)},
		"empty stop":     {StopSequences: ptr([]string{""})},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := resolveGenerationConfig(nil, nil, &cfg); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
	_, err := resolveGenerationConfig(nil, nil, &GenerateContentConfig{Temperature: ptr(math.NaN())})
	if err == nil || !strings.Contains(err.Error(), "temperature") {
		t.Errorf("error = %v, want temperature field name", err)
	}
}

func Example_resolveGenerationConfig() {
	model := &GenerateContentConfig{Temperature: ptr(0.7), MaxOutputTokens: ptr(128)}
	request := &GenerateContentConfig{Temperature: ptr(0.0)}
	cfg, err := resolveGenerationConfig(nil, model, request)
	if err != nil {
		panic(err)
	}
	fmt.Println(cfg.Sampler.Temp, cfg.MaxOutputTokens)
	// Output: 0 128
}
