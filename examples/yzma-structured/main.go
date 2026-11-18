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

// The yzma-structured program shows how GenerateData decodes a Go struct from a
// local model. The output schema requires a JSON object with a single allowed
// code, even though the prompt asks for plain text.
//
// Set YZMA_LIB to the directory containing the native llama.cpp libraries.
// From the repository root, run:
//
//	task dev:yzma:structured -- -model "$HOME/models/SmolLM2-135M.Q4_K_M.gguf"
//
// The program prints "Decoded Go value: {Code:YZMA}".
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/yzma"
)

type status struct {
	Code string `json:"code" jsonschema:"enum=YZMA"`
}

func main() {
	path := flag.String("model", "", "Path to a local base GGUF, such as SmolLM2-135M.Q4_K_M.gguf")
	flag.Parse()
	if *path == "" {
		flag.Usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, *path); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, path string) (err error) {
	local := &yzma.Yzma{}
	g := genkit.Init(ctx, genkit.WithPlugins(local))
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		err = errors.Join(err, local.Close(closeCtx))
	}()
	cpuLayers, contextSize := int32(0), uint32(512)
	model := local.DefineModel(g, yzma.ModelDefinition{
		Name:    "structured",
		Type:    yzma.ModelTypeCompletion,
		Loading: yzma.ModelOptions{GGUFPath: path, NGPULayers: &cpuLayers},
		Context: yzma.ContextOptions{ContextSize: &contextSize},
	}, nil)
	maxTokens, temperature := 32, 0.0
	value, _, err := genkit.GenerateData[status](ctx, g,
		ai.WithModel(model),
		ai.WithPrompt("The output is a single line of plain text: goodbye."),
		ai.WithConfig(&yzma.GenerateContentConfig{MaxOutputTokens: &maxTokens, Temperature: &temperature}),
	)
	if err != nil {
		return err
	}
	if value == nil {
		return fmt.Errorf("generation did not return a Go value")
	}
	fmt.Printf("Decoded Go value: %+v\n", *value)
	return nil
}
