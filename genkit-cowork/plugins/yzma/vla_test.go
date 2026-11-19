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

package yzma

import (
	"bytes"
	"context"
	"encoding/base64"
	"image/color"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/hybridgroup/yzma/pkg/llama"
)

type testConstrainedMediaOperator struct {
	*testVisionOperator
	generateConstrained func(MediaGenerationInput, string) TextGenerationResult
}

func (o *testConstrainedMediaOperator) GenerateConstrainedMedia(_ context.Context, _ llama.Model, _ ProjectorHandle, input MediaGenerationInput, grammar string) (TextGenerationResult, error) {
	return o.generateConstrained(input, grammar), nil
}

func TestGenerateBoundingBox(t *testing.T) {
	data := solidPNG(t, 4, 2, color.White)
	const instruction = "Locate the white rectangle."
	for _, tc := range []struct {
		name   string
		result TextGenerationResult
		want   string
	}{
		{"valid", TextGenerationResult{Text: `{"bottom":1000,"left":0,"right":1000,"top":0}`}, ""},
		{"out of range", TextGenerationResult{Text: `{"bottom":1001,"left":0,"right":1000,"top":0}`}, "invalid bounding box"},
		{"reversed", TextGenerationResult{Text: `{"bottom":1000,"left":650,"right":350,"top":0}`}, "invalid bounding box"},
		{"empty", TextGenerationResult{Text: `{"bottom":0,"left":0,"right":1000,"top":0}`}, "invalid bounding box"},
		{"missing edge", TextGenerationResult{Text: `{"bottom":1000,"right":1000,"top":0}`}, "does not match the schema"},
		{"extra action", TextGenerationResult{Text: `{"bottom":1000,"left":0,"right":1000,"top":0,"action":"click"}`}, "does not match the schema"},
		{"limited", TextGenerationResult{Text: `{"bottom":1000,"left":0,"right":1000,"top":0}`, Limited: true}, "incomplete constrained output"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			op := &testConstrainedMediaOperator{
				testVisionOperator: &testVisionOperator{
					testModelOperator: &testModelOperator{
						load: func(context.Context, ModelOptions) (llama.Model, error) { return 42, nil },
					},
					loadProjector: func(context.Context, llama.Model, ModelDefinition) (ProjectorHandle, error) { return 17, nil },
					freeProjector: func(context.Context, ProjectorHandle) error { return nil },
				},
				generateConstrained: func(input MediaGenerationInput, grammar string) TextGenerationResult {
					if !strings.HasPrefix(grammar, "root ::=") || !strings.Contains(grammar, `"\"bottom\""`) {
						t.Fatalf("expected a box grammar, got %q", grammar)
					}
					if len(input.Messages) != 1 || input.Messages[0].Role != "user" {
						t.Fatalf("expected one user message, got %v", input.Messages)
					}
					parts := input.Messages[0].Parts
					if len(parts) != 2 || parts[0].MIMEType != "image/png" || !bytes.Equal(parts[0].Data, data) ||
						!strings.Contains(parts[1].Text, instruction) || len(input.Tools) != 0 || input.ToolCalling {
						t.Fatalf("expected image then instruction, without tools; got %+v", input)
					}
					return tc.result
				},
			}
			local, g := initDiscoveryTest(t, t.TempDir(), op)
			def := testModelDefinition()
			def.Projector = &ProjectorOptions{GGUFPath: "manual.gguf", Images: true}
			model := local.DefineModel(g, def, nil)
			got, err := GenerateBoundingBox(t.Context(), g, model, pngPart(data), instruction)
			if tc.want != "" {
				if got != nil || err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("GenerateBoundingBox = (%v, %v), want nil result and %q error", got, err, tc.want)
				}
				return
			}
			if want := (BoundingBox{Right: 1000, Bottom: 1000}); err != nil || got == nil || *got != want {
				t.Fatalf("GenerateBoundingBox = (%+v, %v), want (%+v, nil)", got, err, want)
			}
		})
	}
}

func TestConstrainedMediaIncompatibleOptions(t *testing.T) {
	op := &testConstrainedMediaOperator{testVisionOperator: &testVisionOperator{
		testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
			t.Fatal("incompatible request must not load weights")
			return 0, nil
		}},
	}}
	local, g := initLoadingTest(t, op)
	def := testModelDefinition()
	def.Projector = &ProjectorOptions{GGUFPath: "manual.gguf", Images: true}
	model := local.DefineModel(g, def, nil)
	for _, tc := range []struct {
		name   string
		config GenerateContentConfig
		tools  []*ai.ToolDefinition
		cb     ai.ModelStreamCallback
		want   string
	}{
		{name: "tools", tools: []*ai.ToolDefinition{{Name: "click"}}, want: "does not support tool use"},
		{name: "stream", cb: func(context.Context, *ai.ModelResponseChunk) error { return nil }, want: "does not support streaming"},
		{name: "stop", config: GenerateContentConfig{StopSequences: ptr([]string{"}"})}, want: "does not support stop sequences"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := &ai.ModelRequest{
				Messages: []*ai.Message{ai.NewUserMessage(pngPart(solidPNG(t, 4, 2, color.White)))},
				Tools:    tc.tools,
				Config:   tc.config,
				Output: &ai.ModelOutputConfig{Constrained: true, Format: ai.OutputFormatJSON, Schema: map[string]any{
					"type": "object", "properties": map[string]any{}, "additionalProperties": false,
				}},
			}
			got, err := model.Generate(t.Context(), req, tc.cb)
			if got != nil || err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Generate = (%v, %v), want nil result and %q error", got, err, tc.want)
			}
		})
	}
}

func TestNativeBoundingBox(t *testing.T) {
	modelPath, projectorPath, imagePath := os.Getenv("YZMA_TEST_MODEL"), os.Getenv("YZMA_TEST_PROJECTOR"), os.Getenv("YZMA_TEST_VLA_IMAGE")
	if modelPath == "" || projectorPath == "" || imagePath == "" {
		t.Skip("set YZMA_TEST_MODEL and YZMA_TEST_PROJECTOR to the SmolVLM 256M pair and YZMA_TEST_VLA_IMAGE to domestic_llama.jpg")
	}
	f, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxImageBytes+1))
	if err != nil || len(data) > maxImageBytes {
		t.Fatalf("read JPEG fixture: %d bytes, error %v", len(data), err)
	}
	part := ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,"+base64.StdEncoding.EncodeToString(data))
	local := &Yzma{ModelsDir: filepath.Dir(modelPath)}
	g := genkit.Init(t.Context(), genkit.WithPlugins(local))
	t.Cleanup(func() {
		if err := local.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	model := local.DefineModel(g, ModelDefinition{
		Name: "local-box", Type: ModelTypeChat,
		Loading:   ModelOptions{GGUFPath: modelPath, NGPULayers: ptr(int32(0))},
		Projector: &ProjectorOptions{GGUFPath: projectorPath, Images: true},
		Context:   ContextOptions{ContextSize: ptr(uint32(4096)), Threads: ptr(int32(4)), BatchThreads: ptr(int32(4))},
		GenerationDefaults: &GenerateContentConfig{
			MaxOutputTokens: ptr(128), Temperature: ptr(0.0),
		},
	}, nil)
	box, err := GenerateBoundingBox(t.Context(), g, model, part, "Locate the llama in the image.")
	if err != nil || box == nil {
		t.Fatalf("GenerateBoundingBox = (%+v, %v), want a valid typed box", box, err)
	}
	t.Logf("Predicted box: %+v", *box)

	box, err = GenerateBoundingBox(t.Context(), g, model, part, "Return an empty box: left=0, top=0, right=0, bottom=0.")
	if box != nil || err == nil || !strings.Contains(err.Error(), "invalid bounding box") {
		t.Fatalf("empty prediction = (%+v, %v), want no box and a geometry error", box, err)
	}
	t.Logf("Rejected prediction: %v", err)
}
