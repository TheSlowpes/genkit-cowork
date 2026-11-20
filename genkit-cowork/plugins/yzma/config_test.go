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
	"encoding/json"
	"fmt"
	"testing"
)

func TestGenerateContentConfigOptionalValues(t *testing.T) {
	var cfg GenerateContentConfig
	if err := json.Unmarshal([]byte(`{}`), &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.Temperature != nil || cfg.Seed != nil || cfg.StopSequences != nil {
		t.Fatal("omitted fields must remain unset")
	}

	const input = `{"temperature":0,"seed":0,"stopSequences":[]}`
	if err := json.Unmarshal([]byte(input), &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.Temperature == nil || *cfg.Temperature != 0 {
		t.Fatal("explicit greedy temperature was lost")
	}
	if cfg.Seed == nil || *cfg.Seed != 0 {
		t.Fatal("explicit seed 0 was lost")
	}
	if cfg.StopSequences == nil || len(*cfg.StopSequences) != 0 {
		t.Fatal("explicit empty stop sequences were lost")
	}

	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]string{
		"temperature": "0", "seed": "0", "stopSequences": "[]",
	} {
		if got := string(fields[key]); got != want {
			t.Errorf("%s = %s, want %s", key, got, want)
		}
	}
	if _, ok := fields["topP"]; ok {
		t.Error("unset topP must be omitted")
	}
}

func TestModelOptionsOptionalGPULayers(t *testing.T) {
	var opts ModelOptions
	if opts.NGPULayers != nil {
		t.Fatal("omitted GPU layers must preserve native defaults")
	}
	if err := json.Unmarshal([]byte(`{"ggufPath":"model.gguf","nGPULayers":0}`), &opts); err != nil {
		t.Fatal(err)
	}
	if opts.NGPULayers == nil || *opts.NGPULayers != 0 {
		t.Fatal("explicit CPU-only layer count was lost")
	}
}

// ExampleModelDefinition demonstrates configuration without loading native libraries.
func ExampleModelDefinition() {
	gpuLayers := int32(0)
	contextSize := uint32(4096)
	temperature := 0.0
	model := ModelDefinition{
		Name: "local-chat",
		Type: "chat",
		Loading: ModelOptions{
			GGUFPath:   "/models/chat.gguf",
			NGPULayers: &gpuLayers,
		},
		Context: ContextOptions{ContextSize: &contextSize},
		GenerationDefaults: &GenerateContentConfig{
			Temperature: &temperature,
		},
	}
	fmt.Println(model.Name, *model.Loading.NGPULayers, *model.Context.ContextSize)
	// Output: local-chat 0 4096
}
