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

// Command yzma-embeddings turns two strings into vectors for local retrieval.
// It resolves the GGUF by name through Genkit, so no manual registration
// or separate inference server is needed. See the plugin README for the exact
// command and the tested 384-dimensional BGE GGUF.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/yzma"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func main() {
	modelsDir := flag.String("models-dir", "", "directory containing local GGUFs; defaults to Yzma's models directory")
	name := flag.String("embedder", "yzma/bge-small-en-v1.5-q4_k_m", "qualified Genkit embedder name")
	flag.Parse()
	if err := run(context.Background(), *modelsDir, *name); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, modelsDir, name string) (err error) {
	plugin := &yzma.Yzma{ModelsDir: modelsDir}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		err = errors.Join(err, plugin.Close(cleanupCtx))
	}()

	resp, err := genkit.Embed(ctx, g, ai.WithEmbedderName(name), ai.WithDocs(
		ai.DocumentFromText("Rivers flow toward the ocean.", nil),
		ai.DocumentFromText("A needle and thread mend a torn shirt.", nil),
	))
	if err != nil {
		return err
	}
	fmt.Printf("Embedding count: %d\n", len(resp.Embeddings))
	for i, result := range resp.Embeddings {
		fmt.Printf("Vector %d: %d dimensions\n", i+1, len(result.Embedding))
	}
	return nil
}
