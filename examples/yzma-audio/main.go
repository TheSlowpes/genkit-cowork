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

// The yzma-audio program sends a WAV clip to a local speech recognition model
// through Genkit. The program reads the file and passes its bytes inline; the
// model request does not reference the filesystem.
// See the Yzma plugin README for the exact command and a sample clip.
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
	model := flag.String("model", "", "Path to the Qwen3-ASR decoder GGUF")
	projector := flag.String("projector", "", "Path to its matching qwen3a projector GGUF")
	audio := flag.String("audio", "", "Path to a mono 16 kHz PCM16 WAV")
	flag.Parse()
	if *model == "" || *projector == "" || *audio == "" {
		flag.Usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, *model, *projector, *audio); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, modelPath, projectorPath, audioPath string) (err error) {
	f, err := os.Open(audioPath)
	if err != nil {
		return err
	}
	defer f.Close()
	const limit = 1 << 20
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return err
	}
	if len(data) > limit {
		return fmt.Errorf("WAV exceeds %d bytes", limit)
	}
	audio := ai.NewMediaPart("audio/wav", "data:audio/wav;base64,"+base64.StdEncoding.EncodeToString(data))

	local := &yzma.Yzma{}
	g := genkit.Init(ctx, genkit.WithPlugins(local))
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		err = errors.Join(err, local.Close(closeCtx))
	}()
	gpu, threads, contextSize := int32(0), int32(2), uint32(2048)
	model := local.DefineModel(g, yzma.ModelDefinition{
		Name: "transcribe", Type: yzma.ModelTypeChat,
		Loading:   yzma.ModelOptions{GGUFPath: modelPath, NGPULayers: &gpu},
		Projector: &yzma.ProjectorOptions{GGUFPath: projectorPath, Audio: true},
		Context:   yzma.ContextOptions{ContextSize: &contextSize, Threads: &threads, BatchThreads: &threads},
		// Qwen3-ASR uses an empty system turn and an assistant language prefix.
		ChatTemplate: `<|im_start|>system
<|im_end|>
<|im_start|>user
{{ messages[0].content }}<|im_end|>
<|im_start|>assistant
language English<asr_text>`,
	}, nil)
	maxTokens, temperature := 96, 0.0
	result, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithMessages(ai.NewUserMessage(audio)),
		ai.WithConfig(&yzma.GenerateContentConfig{MaxOutputTokens: &maxTokens, Temperature: &temperature}),
	)
	if err != nil {
		return err
	}
	fmt.Printf("Transcript: %s\n", result.Text())
	return nil
}
