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
// in a Genkit request. With -box it prints a grammar-constrained bounding box
// instead of a text answer. It runs locally, without a model server or media
// fetches, and never acts on a prediction.
// Use -check-grounding with the documented Pelican pair to test two generated
// target positions against known coordinates. A whole-frame box fails.
// See the Yzma plugin README for the model/projector files, exact command,
// and an example response.
package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
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
	"github.com/TheSlowpes/genkit-cowork/internal/groundingfixture"
)

func main() {
	model := flag.String("model", "", "Path to the decoder GGUF")
	projector := flag.String("projector", "", "Path to the matching projector GGUF")
	image := flag.String("image", "", "Path to a JPEG or PNG")
	question := flag.String("question", "", "Question or bounding box instruction")
	box := flag.Bool("box", false, "Return a bounding box as JSON")
	grounding := flag.Bool("check-grounding", false, "Localize a red square in two generated PNGs")
	flag.Parse()
	if *model == "" || *projector == "" ||
		(!*grounding && *image == "") || (*grounding && (*box || *image != "" || *question != "")) {
		flag.Usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if *question == "" {
		if *box {
			*question = "Locate the main subject in the image."
		} else {
			*question = "What animal is in this image?"
		}
	}
	if err := run(ctx, *model, *projector, *image, *question, *box, *grounding); err != nil {
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

func run(ctx context.Context, modelPath, projectorPath, imagePath, question string, predictBox, checkGrounding bool) (err error) {
	var image *ai.Part
	if !checkGrounding {
		image, err = readImage(imagePath)
		if err != nil {
			return err
		}
	}
	local := &yzma.Yzma{}
	g := genkit.Init(ctx, genkit.WithPlugins(local))
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		err = errors.Join(err, local.Close(closeCtx))
	}()
	gpu, threads, contextSize := int32(0), int32(2), uint32(4096)
	maxTokens, temperature := 48, 0.0
	projector := &yzma.ProjectorOptions{GGUFPath: projectorPath, Images: true}
	if predictBox || checkGrounding {
		maxTokens = 128
	}
	if checkGrounding {
		projector.ImageMinTokens = 1024
	}
	model := local.DefineModel(g, yzma.ModelDefinition{
		Name: "vision", Type: yzma.ModelTypeChat,
		Loading:   yzma.ModelOptions{GGUFPath: modelPath, NGPULayers: &gpu},
		Projector: projector,
		Context:   yzma.ContextOptions{ContextSize: &contextSize, Threads: &threads, BatchThreads: &threads},
		GenerationDefaults: &yzma.GenerateContentConfig{
			MaxOutputTokens: &maxTokens, Temperature: &temperature,
		},
	}, nil)
	if checkGrounding {
		return checkTargets(ctx, g, model)
	}
	if predictBox {
		box, err := yzma.GenerateBoundingBox(ctx, g, model, image, question)
		if err != nil {
			return err
		}
		return json.NewEncoder(os.Stdout).Encode(box)
	}
	result, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Look at this image: "), image, ai.NewTextPart(" "+question))),
	)
	if err != nil {
		return err
	}
	fmt.Printf("Answer: %s\n", result.Text())
	return nil
}

func checkTargets(ctx context.Context, g *genkit.Genkit, model ai.Model) error {
	var failures []error
	for _, target := range groundingfixture.Targets() {
		if err := checkTarget(ctx, g, model, target); err != nil {
			failures = append(failures, fmt.Errorf("%s target: %w", target.Name, err))
		}
	}
	return errors.Join(failures...)
}

func checkTarget(ctx context.Context, g *genkit.Genkit, model ai.Model, target groundingfixture.Target) error {
	url, err := target.PNG()
	if err != nil {
		return err
	}
	box, err := yzma.GenerateBoundingBox(ctx, g, model, ai.NewMediaPart("image/png", url), groundingfixture.Instruction)
	if err != nil {
		return err
	}
	if err := target.Check(box.Left, box.Top, box.Right, box.Bottom); err != nil {
		return err
	}
	fmt.Printf("%s red square: %+v\n", target.Name, *box)
	return nil
}
