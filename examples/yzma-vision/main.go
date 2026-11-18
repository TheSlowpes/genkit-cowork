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

// The yzma-vision program reads a JPEG or PNG and sends it between text parts
// in a Genkit request. It runs locally, without a model server or media fetches.
// See the Yzma plugin README for the model/projector files, exact command,
// and an example response.
package main

import (
	"context"
	"encoding/base64"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/yzma"
)

func main() {
	model := flag.String("model", "", "Path to the decoder GGUF")
	projector := flag.String("projector", "", "Path to the matching projector GGUF")
	image := flag.String("image", "", "Path to a JPEG or PNG")
	question := flag.String("question", "What animal is in this image?", "Question about the image")
	flag.Parse()
	if *model == "" || *projector == "" || *image == "" {
		flag.Usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, *model, *projector, *image, *question); err != nil {
		log.Fatal(err)
	}
}

func readImage(path string) (*ai.Part, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	const limit = 4 << 20
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if len(data) > limit {
		return nil, fmt.Errorf("image exceeds %d bytes", limit)
	}
	mime := http.DetectContentType(data)
	if mime != "image/jpeg" && mime != "image/png" {
		return nil, fmt.Errorf("expected JPEG or PNG, got %s", mime)
	}
	return ai.NewMediaPart(mime, "data:"+mime+";base64,"+base64.StdEncoding.EncodeToString(data)), nil
}

func run(ctx context.Context, modelPath, projectorPath, imagePath, question string) (err error) {
	image, err := readImage(imagePath)
	if err != nil {
		return err
	}
	local := &yzma.Yzma{}
	g := genkit.Init(ctx, genkit.WithPlugins(local))
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		err = errors.Join(err, local.Close(closeCtx))
	}()
	gpu, threads, contextSize := int32(0), int32(4), uint32(4096)
	model := local.DefineModel(g, yzma.ModelDefinition{
		Name: "vision", Type: yzma.ModelTypeChat,
		Loading:   yzma.ModelOptions{GGUFPath: modelPath, NGPULayers: &gpu},
		Projector: &yzma.ProjectorOptions{GGUFPath: projectorPath, Images: true},
		Context:   yzma.ContextOptions{ContextSize: &contextSize, Threads: &threads, BatchThreads: &threads},
	}, nil)
	maxTokens, temperature := 48, 0.0
	result, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Look at this image: "), image, ai.NewTextPart(" "+question))),
		ai.WithConfig(&yzma.GenerateContentConfig{MaxOutputTokens: &maxTokens, Temperature: &temperature}),
	)
	if err != nil {
		return err
	}
	fmt.Printf("Answer: %s\n", result.Text())
	return nil
}
