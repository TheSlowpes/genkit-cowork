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

// The yzma-video program sends a local MP4 inline to a Genkit model. Yzma
// samples the frames at the clip's position in the prompt and ignores the
// soundtrack. See the Yzma plugin README for the model pair, a command to
// generate the test clip, and the expected answer.
package main

import (
	"context"
	"encoding/base64"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/yzma"
)

func main() {
	model := flag.String("model", "", "Path to the SmolVLM 256M Instruct GGUF")
	projector := flag.String("projector", "", "Path to the matching projector GGUF")
	video := flag.String("video", "", "Path to a short H.264 fast-start MP4")
	question := flag.String("question", "What color is the first frame?", "Question about the clip")
	flag.Parse()
	if *model == "" || *projector == "" || *video == "" {
		flag.Usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, *model, *projector, *video, *question); err != nil {
		log.Fatal(err)
	}
}

func readVideo(path string) (*ai.Part, error) {
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
		return nil, fmt.Errorf("video exceeds %d bytes", limit)
	}
	return ai.NewMediaPart("video/mp4", "data:video/mp4;base64,"+base64.StdEncoding.EncodeToString(data)), nil
}

func run(ctx context.Context, modelPath, projectorPath, videoPath, question string) (err error) {
	video, err := readVideo(videoPath)
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
	gpu, threads, contextSize := int32(0), int32(2), uint32(4096)
	model := local.DefineModel(g, yzma.ModelDefinition{
		Name: "video", Type: yzma.ModelTypeChat,
		Loading:   yzma.ModelOptions{GGUFPath: modelPath, NGPULayers: &gpu},
		Projector: &yzma.ProjectorOptions{GGUFPath: projectorPath, Video: true},
		Context:   yzma.ContextOptions{ContextSize: &contextSize, Threads: &threads, BatchThreads: &threads},
	}, nil)
	maxTokens, temperature := 32, 0.0
	result, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Watch this clip: "), video, ai.NewTextPart(" "+question))),
		ai.WithConfig(&yzma.GenerateContentConfig{MaxOutputTokens: &maxTokens, Temperature: &temperature}),
	)
	if err != nil {
		return err
	}
	fmt.Printf("Answer: %s\n", result.Text())
	return nil
}
