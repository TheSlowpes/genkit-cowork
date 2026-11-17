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
// and cleanup for a local GGUF model. It uses CPU-only weight placement to avoid
// requiring a GPU. It does not run inference because token generation is not yet
// implemented.
//
// Install the native libraries compatible with the pinned Yzma release, then run:
//
//	YZMA_LIB=/path/to/libs go run ./examples/yzma-loading -model /path/to/model.gguf
//
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

	"github.com/firebase/genkit/go/genkit"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/yzma"
)

func main() {
	path := flag.String("model", "", "Path to a local GGUF model")
	lib := flag.String("lib", os.Getenv("YZMA_LIB"), "Directory containing llama.cpp shared libraries")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, *lib, *path); err != nil {
		slog.Error("model loading failed", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, lib, path string) (err error) {
	if lib == "" || path == "" {
		return fmt.Errorf("set YZMA_LIB or -lib, and provide -model")
	}
	plugin := &yzma.Yzma{LibPath: lib}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	defer func() {
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer cancel()
		err = errors.Join(err, plugin.Close(cleanup))
	}()
	cpuLayers := int32(0)
	plugin.DefineModel(g, yzma.ModelDefinition{
		Name: "local-chat",
		Type: "chat",
		Loading: yzma.ModelOptions{
			GGUFPath: path, NGPULayers: &cpuLayers,
		},
	}, nil)
	if err := plugin.Preload(ctx, "local-chat"); err != nil {
		return err
	}
	fmt.Println("Local model loaded. Token generation is not implemented yet.")
	return nil
}
