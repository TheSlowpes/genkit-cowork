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

// Command yzma-loading demonstrates Genkit registration, explicit preloading,
// text generation, and cleanup for a local GGUF model. It uses CPU-only weight
// placement to avoid requiring a GPU.
//
// Install the native libraries compatible with the pinned Yzma release, then run:
//
//	YZMA_LIB=/path/to/libs go run ./examples/yzma-loading -model /path/to/model.gguf
//
// To chat with the model in the Genkit Dev UI, keep the example alive with -serve:
//
//	genkit start -- go run ./examples/yzma-loading -model /path/to/model.gguf -serve
//
// Set YZMA_LIB before starting Genkit. Select yzma/local-chat in the Models view.
// Alternatively, pass -prompt "Hello" to generate a single response and exit.
// For a base model, pass -type completion. It registers yzma/local-completion
// and treats one user message as a raw prompt without requiring a chat template.
// Successful execution prints that the model loaded and frees its weights before
// exiting. Missing files and load failures produce a non-zero exit status.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/yzma"
)

type options struct {
	lib, path, template, prompt string
	modelType                   yzma.ModelType
	serve                       bool
}

func main() {
	var opts options
	flag.StringVar(&opts.path, "model", "", "Path to a local GGUF model")
	flag.StringVar(&opts.lib, "lib", os.Getenv("YZMA_LIB"), "Directory containing llama.cpp shared libraries")
	flag.StringVar(&opts.prompt, "prompt", "", "Generate one text response before exiting")
	flag.BoolVar(&opts.serve, "serve", false, "Keep running for the Genkit Dev UI")
	flag.StringVar(&opts.template, "template", "", "Override the chat model's embedded template")
	modelType := flag.String("type", string(yzma.ModelTypeChat), "Model type: chat or completion")
	flag.Parse()
	opts.modelType = yzma.ModelType(*modelType)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, opts); err != nil {
		slog.Error("local model failed", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, opts options) (err error) {
	if opts.lib == "" || opts.path == "" {
		return fmt.Errorf("set YZMA_LIB or -lib, and provide -model")
	}
	if opts.modelType != yzma.ModelTypeChat && opts.modelType != yzma.ModelTypeCompletion {
		return fmt.Errorf("invalid -type %q; use chat or completion", opts.modelType)
	}
	if opts.modelType == yzma.ModelTypeCompletion && opts.template != "" {
		return fmt.Errorf("-template is only valid for -type chat")
	}
	plugin := &yzma.Yzma{LibPath: opts.lib}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	defer func() {
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer cancel()
		err = errors.Join(err, plugin.Close(cleanup))
	}()
	cpuLayers := int32(0)
	contextSize := uint32(4096)
	name := "local-" + string(opts.modelType)
	modelName := "yzma/" + name
	plugin.DefineModel(g, yzma.ModelDefinition{
		Name:         name,
		Type:         opts.modelType,
		ChatTemplate: opts.template,
		Context:      yzma.ContextOptions{ContextSize: &contextSize},
		Loading: yzma.ModelOptions{
			GGUFPath: opts.path, NGPULayers: &cpuLayers,
		},
	}, nil)
	if err := plugin.Preload(ctx, name); err != nil {
		return err
	}
	fmt.Printf("Local model loaded as %s.\n", modelName)
	if opts.prompt != "" {
		maxTokens, temperature := 64, 0.0
		text, err := genkit.GenerateText(ctx, g,
			ai.WithModelName(modelName), ai.WithPrompt(opts.prompt),
			ai.WithConfig(&yzma.GenerateContentConfig{MaxOutputTokens: &maxTokens, Temperature: &temperature}),
		)
		if err != nil {
			return err
		}
		fmt.Println(text)
	}
	if opts.serve {
		fmt.Printf("Ready for the Genkit Dev UI. Select %s under Models.\n", modelName)
		<-ctx.Done()
	}
	return nil
}
