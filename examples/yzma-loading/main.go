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
// For a tool-capable chat model, -tools registers an add tool and the localChat
// flow. Select that flow in the Dev UI to exercise the complete tool loop.
// Embedded Jinja templates are loaded automatically; -template-name selects a
// named embedded template when required.
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
	lib, path, template, templateName, prompt string
	modelType                                 yzma.ModelType
	serve                                     bool
	tools                                     bool
	maxTokens                                 int
}

func main() {
	var opts options
	flag.StringVar(&opts.path, "model", "", "Path to a local GGUF model")
	flag.StringVar(&opts.lib, "lib", os.Getenv("YZMA_LIB"), "Directory containing llama.cpp shared libraries")
	flag.StringVar(&opts.prompt, "prompt", "", "Generate one text response before exiting")
	flag.BoolVar(&opts.serve, "serve", false, "Keep running for the Genkit Dev UI")
	flag.StringVar(&opts.template, "template", "", "Override the chat model's embedded template")
	flag.StringVar(&opts.templateName, "template-name", "", "Select a named template embedded in the GGUF")
	flag.BoolVar(&opts.tools, "tools", false, "Enable tool calling and register the add demo tool")
	flag.IntVar(&opts.maxTokens, "max-tokens", 256, "Maximum output tokens per model call")
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
	if opts.modelType == yzma.ModelTypeCompletion && (opts.template != "" || opts.templateName != "" || opts.tools) {
		return fmt.Errorf("templates and tools are only valid for -type chat")
	}
	if opts.template != "" && opts.templateName != "" {
		return fmt.Errorf("choose -template or -template-name, not both")
	}
	if opts.maxTokens <= 0 {
		return fmt.Errorf("-max-tokens must be positive")
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
		Name:             name,
		Type:             opts.modelType,
		ChatTemplate:     opts.template,
		ChatTemplateName: opts.templateName,
		ToolCalling:      opts.tools,
		Context:          yzma.ContextOptions{ContextSize: &contextSize},
		Loading: yzma.ModelOptions{
			GGUFPath: opts.path, NGPULayers: &cpuLayers,
		},
	}, nil)
	var tools []ai.ToolRef
	if opts.tools {
		type addInput struct {
			A int `json:"a"`
			B int `json:"b"`
		}
		tools = append(tools, genkit.DefineTool(g, "add", "Add two integers",
			func(ctx *ai.ToolContext, input addInput) (int, error) {
				if err := ctx.Err(); err != nil {
					return 0, err
				}
				return input.A + input.B, nil
			},
		))
	}
	generate := func(ctx context.Context, prompt string) (string, error) {
		temperature := 0.0
		options := []ai.GenerateOption{
			ai.WithModelName(modelName), ai.WithPrompt(prompt),
			ai.WithConfig(&yzma.GenerateContentConfig{MaxOutputTokens: &opts.maxTokens, Temperature: &temperature}),
		}
		if len(tools) > 0 {
			options = append(options, ai.WithTools(tools...))
		}
		return genkit.GenerateText(ctx, g, options...)
	}
	type chatInput struct {
		Prompt string `json:"prompt"`
	}
	genkit.DefineFlow(g, "localChat", func(ctx context.Context, input chatInput) (string, error) {
		return generate(ctx, input.Prompt)
	})
	if err := plugin.Preload(ctx, name); err != nil {
		return err
	}
	fmt.Printf("Local model loaded as %s.\n", modelName)
	if opts.prompt != "" {
		text, err := generate(ctx, opts.prompt)
		if err != nil {
			return err
		}
		fmt.Println(text)
	}
	if opts.serve {
		fmt.Printf("Ready for the Genkit Dev UI. Select %s under Models.\n", modelName)
		if opts.tools {
			fmt.Println("Use the localChat flow to test Genkit executing the add tool.")
		}
		<-ctx.Done()
	}
	return nil
}
