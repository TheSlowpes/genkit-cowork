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

// The yzma-speech example asks a local Soprano model to say a line of text.
// Genkit returns inline WAV audio, which the example decodes to a file for
// playback. See the Yzma plugin README for the exact command, artifact links,
// and the native end-to-end check.
package main

import (
	"context"
	"encoding/base64"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/yzma"
)

func main() {
	backbone := flag.String("model", "", "path to the Soprano backbone GGUF")
	decoder := flag.String("decoder", "", "path to the matching Soprano mmproj GGUF")
	text := flag.String("text", "Hello from Genkit.", "English text to speak")
	voice := flag.String("voice", "default", "built-in voice (default)")
	output := flag.String("out", "speech.wav", "output WAV path")
	flag.Parse()
	if *backbone == "" || *decoder == "" || os.Getenv("YZMA_LIB") == "" {
		log.Fatal("set YZMA_LIB and provide -model and -decoder")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, *backbone, *decoder, *text, *voice, *output); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, backbone, decoder, text, voice, output string) (err error) {
	plugin := &yzma.Yzma{}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	defer func() {
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		err = errors.Join(err, plugin.Close(cleanup))
	}()

	cpuLayers, threads := int32(0), int32(4)
	plugin.DefineModel(g, yzma.ModelDefinition{
		Name:              "soprano",
		Type:              yzma.ModelTypeSpeech,
		Loading:           yzma.ModelOptions{GGUFPath: backbone, NGPULayers: &cpuLayers},
		Context:           yzma.ContextOptions{Threads: &threads, BatchThreads: &threads},
		SpeechDecoderPath: decoder,
	}, nil)
	res, err := genkit.Generate(ctx, g,
		ai.WithModelName("yzma/soprano"),
		ai.WithPrompt(text),
		ai.WithConfig(yzma.SpeechConfig{Voice: voice}),
	)
	if err != nil {
		return err
	}
	encoded, ok := strings.CutPrefix(res.Media(), "data:audio/wav;base64,")
	if !ok {
		return fmt.Errorf("expected inline WAV audio in the Genkit response")
	}
	wav, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return err
	}
	if err := os.WriteFile(output, wav, 0o644); err != nil {
		return err
	}
	fmt.Printf("Wrote %s (%d bytes)\n", output, len(wav))
	return nil
}
