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
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/hybridgroup/yzma/pkg/llama"
)

type testVisionOperator struct {
	*testModelOperator
	loadProjector func(context.Context, llama.Model, ModelDefinition) (ProjectorHandle, error)
	freeProjector func(context.Context, ProjectorHandle) error
	generate      func(context.Context, llama.Model, ProjectorHandle, MediaGenerationInput) (TextGenerationResult, error)
}

func (o *testVisionOperator) LoadProjector(ctx context.Context, model llama.Model, def ModelDefinition) (ProjectorHandle, error) {
	return o.loadProjector(ctx, model, def)
}

func (o *testVisionOperator) FreeProjector(ctx context.Context, handle ProjectorHandle) error {
	return o.freeProjector(ctx, handle)
}

func (o *testVisionOperator) GenerateMedia(ctx context.Context, model llama.Model, handle ProjectorHandle, input MediaGenerationInput, _ func(context.Context, string) error) (TextGenerationResult, error) {
	return o.generate(ctx, model, handle, input)
}

func solidPNG(t *testing.T, width, height int, c color.Color) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(img, img.Bounds(), image.NewUniform(c), image.Point{}, draw.Src)
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func pngPart(data []byte) *ai.Part {
	return ai.NewMediaPart("image/png", "data:image/png;base64,"+base64.StdEncoding.EncodeToString(data))
}

func TestVisionDiscovery(t *testing.T) {
	for _, tc := range []struct {
		name       string
		projectors []string
		want       string
	}{
		{name: "exact name", projectors: []string{"mmproj-vision.Q8.gguf", "other.gguf"}, want: "mmproj-vision.Q8.gguf"},
		{name: "unique metadata", projectors: []string{"other.gguf"}, want: "other.gguf"},
		{name: "missing"},
		{name: "ambiguous", projectors: []string{"one.gguf", "two.gguf"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeGGUFFixture(t, dir, "vision.Q8.gguf", map[string]ggufValue{
				"general.name":            {kind: ggufString, text: "SmolVLM Small"},
				"general.basename":        {kind: ggufString, text: "SmolVLM"},
				"llama.embedding_length":  {kind: 4, num: 576},
				"tokenizer.chat_template": {kind: ggufString, text: smolVLMTemplate},
			})
			writeProjector := func(name string, width uint64) {
				writeGGUF(t, dir, name, map[string]ggufValue{
					"general.architecture":       {kind: ggufString, text: "clip"},
					"general.name":               {kind: ggufString, text: "SmolVLM Small"},
					"clip.projector_type":        {kind: ggufString, text: "idefics3"},
					"clip.has_vision_encoder":    {kind: 7, num: 1},
					"clip.vision.projection_dim": {kind: 4, num: width},
				}, "vision.weight")
			}
			writeProjector("wrong-width.gguf", 512)
			for _, name := range tc.projectors {
				writeProjector(name, 576)
			}
			canLoad := false
			op := &testVisionOperator{testModelOperator: &testModelOperator{
				load: func(context.Context, ModelOptions) (llama.Model, error) {
					if !canLoad {
						t.Fatal("listing and resolving must not load weights")
					}
					return 42, nil
				},
			},
				loadProjector: func(_ context.Context, _ llama.Model, def ModelDefinition) (ProjectorHandle, error) {
					if got := def.Projector.GGUFPath; got != filepath.Join(dir, tc.want) {
						t.Errorf("paired projector = %q, want %q", got, tc.want)
					}
					return 17, nil
				},
				freeProjector: func(context.Context, ProjectorHandle) error { return nil },
			}
			plugin, g := initDiscoveryTest(t, dir, op)
			if resolved := Model(g, "vision.q8"); (resolved != nil) != (tc.want != "") {
				t.Fatalf("resolved decoder = %v, want projector %q", resolved != nil, tc.want)
			}
			descriptors := plugin.ListActions(t.Context())
			if tc.want == "" {
				if len(descriptors) != 0 {
					t.Fatalf("listed %d descriptors, want none", len(descriptors))
				}
				return
			}
			if len(descriptors) != 1 || descriptors[0].Name != "yzma/vision.q8" {
				t.Fatalf("listed %v, want one vision model", descriptors)
			}
			supports := descriptors[0].Metadata["model"].(map[string]any)["supports"].(map[string]any)
			if supports["media"] != true || supports["tools"] != false {
				t.Errorf("supports = %v, want media without tools", supports)
			}
			canLoad = true
			if err := plugin.Preload(t.Context(), "vision.q8"); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestVisionGenkitOrderAndLifecycle(t *testing.T) {
	red := solidPNG(t, 2, 2, color.RGBA{R: 255, A: 255})
	blue := solidPNG(t, 3, 3, color.RGBA{B: 255, A: 255})
	var events []string
	retryErr := errors.New("projector still busy")
	inferenceErr := errors.New("failed request")
	failFree := true
	requests := 0
	canceledCtx, cancel := context.WithCancel(t.Context())
	defer cancel()
	op := &testVisionOperator{
		testModelOperator: &testModelOperator{
			load: func(context.Context, ModelOptions) (llama.Model, error) {
				events = append(events, "load model")
				return 42, nil
			},
			free: func(context.Context, llama.Model) error {
				events = append(events, "free model")
				return nil
			},
		},
		loadProjector: func(_ context.Context, model llama.Model, def ModelDefinition) (ProjectorHandle, error) {
			if model != 42 || def.Projector.GGUFPath != "manual.gguf" {
				t.Fatalf("load projector for model %d with options %+v", model, def.Projector)
			}
			events = append(events, "load projector")
			return 17, nil
		},
		freeProjector: func(_ context.Context, handle ProjectorHandle) error {
			if handle != 17 {
				t.Fatalf("free projector handle = %d, want 17", handle)
			}
			events = append(events, "free projector")
			if failFree {
				failFree = false
				return retryErr
			}
			return nil
		},
		generate: func(ctx context.Context, model llama.Model, handle ProjectorHandle, input MediaGenerationInput) (TextGenerationResult, error) {
			if model != 42 || handle != 17 {
				t.Fatalf("generate with %d/%d, want 42/17", model, handle)
			}
			want := []MediaInputPart{
				{Text: "First: "}, {MIMEType: "image/png", Data: red},
				{Text: " then "}, {MIMEType: "image/png", Data: blue}, {Text: " Which is red?"},
			}
			if len(input.Messages) != 1 || input.Messages[0].Role != "user" {
				t.Fatalf("messages = %v, want one user message", input.Messages)
			}
			got := input.Messages[0].Parts
			if diff := cmp.Diff(want, got, cmpopts.IgnoreUnexported(MediaInputPart{})); diff != "" {
				t.Errorf("ordered parts (-want +got):\n%s", diff)
			}
			requests++
			switch requests {
			case 1:
				return TextGenerationResult{}, inferenceErr
			case 2:
				cancel()
				return TextGenerationResult{}, ctx.Err()
			}
			return TextGenerationResult{Text: "The first image.", InputTokens: 37, OutputTokens: 4}, nil
		},
	}
	plugin, g := initDiscoveryTest(t, t.TempDir(), op)
	def := testModelDefinition()
	def.Projector = &ProjectorOptions{GGUFPath: "manual.gguf", Images: true}
	model := plugin.DefineModel(g, def, nil)
	*def.Projector = ProjectorOptions{GGUFPath: "changed.gguf"} // Registration owns a snapshot.
	prompt := ai.NewUserMessage(ai.NewTextPart("First: "), pngPart(red), ai.NewTextPart(" then "), pngPart(blue), ai.NewTextPart(" Which is red?"))
	generate := func(ctx context.Context) (*ai.ModelResponse, error) {
		return genkit.Generate(ctx, g, ai.WithModel(model), ai.WithMessages(prompt))
	}
	if _, err := generate(t.Context()); !errors.Is(err, inferenceErr) {
		t.Fatalf("first generate error = %v, want %v", err, inferenceErr)
	}
	if _, err := generate(canceledCtx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled generation = %v, want context.Canceled", err)
	}
	result, err := generate(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if result.Text() != "The first image." || result.Usage.TotalTokens != 41 {
		t.Errorf("unexpected response: %+v", result)
	}
	if err := plugin.Close(t.Context()); !errors.Is(err, retryErr) {
		t.Fatalf("first close error = %v, want %v", err, retryErr)
	}
	if _, err := generate(t.Context()); !errors.Is(err, ErrClosed) {
		t.Fatalf("generation after closing = %v, want ErrClosed", err)
	}
	if err := plugin.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff([]string{"load model", "load projector", "free projector", "free projector", "free model"}, events); diff != "" {
		t.Errorf("lifecycle (-want +got):\n%s", diff)
	}
}

func TestVisionRejectsInvalidMediaBeforeLoad(t *testing.T) {
	valid := solidPNG(t, 2, 2, color.White)
	large := append(append([]byte{}, valid...), make([]byte, (3<<20)-len(valid))...)
	plugin, g := initDiscoveryTest(t, t.TempDir(), &testVisionOperator{testModelOperator: &testModelOperator{
		load: func(context.Context, ModelOptions) (llama.Model, error) {
			t.Fatal("an invalid request must not load weights")
			return 0, nil
		},
	}})
	def := testModelDefinition()
	def.Projector = &ProjectorOptions{GGUFPath: "manual.gguf", Images: true}
	model := plugin.DefineModel(g, def, nil)
	for _, tc := range []struct {
		name  string
		parts []*ai.Part
		want  string
	}{
		{"remote", []*ai.Part{ai.NewMediaPart("image/png", "https://example.com/image.png")}, "inline base64"},
		{"local", []*ai.Part{ai.NewMediaPart("image/png", "file:///tmp/image.png")}, "inline base64"},
		{"unsupported", []*ai.Part{ai.NewMediaPart("image/gif", "data:image/gif;base64,AAAA")}, "unsupported media MIME"},
		{"mime mismatch", []*ai.Part{ai.NewMediaPart("image/jpeg", pngPart(valid).Text)}, "content type does not match"},
		{"invalid base64", []*ai.Part{ai.NewMediaPart("image/png", "data:image/png;base64,%%%%")}, "invalid media base64"},
		{"wrong data", []*ai.Part{pngPart([]byte("not a PNG"))}, "not a valid image/png"},
		{"truncated", []*ai.Part{pngPart(valid[:len(valid)-10])}, "decode image/png"},
		{"dimensions", []*ai.Part{pngPart(solidPNG(t, maxImageEdge+1, 1, color.White))}, "image dimensions"},
		{"image bytes", []*ai.Part{pngPart(make([]byte, maxImageBytes+1))}, "4194304-byte limit"},
		{"request bytes", []*ai.Part{pngPart(large), pngPart(large), pngPart(large)}, "2097152-byte limit"},
		{"part count", []*ai.Part{pngPart(valid), pngPart(valid), pngPart(valid), pngPart(valid), pngPart(valid)}, "at most 4 media parts"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := genkit.Generate(t.Context(), g, ai.WithModel(model), ai.WithMessages(ai.NewUserMessage(tc.parts...)))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("generate error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestVisionRejectsStructuredOutputBeforeLoad(t *testing.T) {
	p, g := initLoadingTest(t, &testVisionOperator{testModelOperator: &testModelOperator{
		load: func(context.Context, ModelOptions) (llama.Model, error) {
			t.Fatal("unsupported output must not load weights")
			return 0, nil
		},
	}})
	def := testModelDefinition()
	def.Projector = &ProjectorOptions{GGUFPath: "manual.gguf", Images: true}
	model := p.DefineModel(g, def, nil)
	schema := map[string]any{"type": "object"}
	part := pngPart(solidPNG(t, 2, 2, color.White))
	for _, tc := range []struct {
		name   string
		output ai.ModelOutputConfig
		want   string
	}{
		{"JSON", ai.ModelOutputConfig{Format: ai.OutputFormatJSON}, "only unconstrained text output"},
		{"schema", ai.ModelOutputConfig{Schema: schema}, "only unconstrained text output"},
		{"constrained JSON", ai.ModelOutputConfig{Format: ai.OutputFormatJSON, Constrained: true, Schema: schema}, "does not support native constrained output"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			res, err := model.Generate(t.Context(), &ai.ModelRequest{
				Messages: []*ai.Message{ai.NewUserMessage(ai.NewTextPart("What is this?"), part)},
				Output:   &tc.output,
			}, nil)
			if res != nil || err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Generate = (%v, %v), want nil response and %q error", res, err, tc.want)
			}
		})
	}
}

func TestVisionTemplateCannotReorderMedia(t *testing.T) {
	input := MediaGenerationInput{
		TextGenerationInput: TextGenerationInput{
			Type: ModelTypeChat, ChatTemplate: `{{ messages[1].content }}{{ messages[0].content }}`,
		},
		Messages: []MediaMessage{
			{Role: "user", Parts: []MediaInputPart{{MIMEType: "image/png"}}},
			{Role: "user", Parts: []MediaInputPart{{MIMEType: "image/png"}}},
		},
	}
	if _, _, err := renderMediaPrompt(input, nil); err == nil || !strings.Contains(err.Error(), "ordered media slots") {
		t.Fatalf("render reordered prompt error = %v", err)
	}
}

func TestNativeVisionGeneration(t *testing.T) {
	modelPath, projectorPath := os.Getenv("YZMA_TEST_MODEL"), os.Getenv("YZMA_TEST_PROJECTOR")
	if modelPath == "" || projectorPath == "" {
		t.Skip("set YZMA_TEST_MODEL and YZMA_TEST_PROJECTOR to the SmolVLM 256M test pair")
	}
	local := &Yzma{ModelsDir: filepath.Dir(modelPath)}
	g := genkit.Init(t.Context(), genkit.WithPlugins(local))
	t.Cleanup(func() {
		if err := local.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	model := local.DefineModel(g, ModelDefinition{
		Name: "local-vision", Type: ModelTypeChat,
		Loading:   ModelOptions{GGUFPath: modelPath, NGPULayers: ptr(int32(0))},
		Projector: &ProjectorOptions{GGUFPath: projectorPath, Images: true},
		Context:   ContextOptions{ContextSize: ptr(uint32(4096)), Threads: ptr(int32(4)), BatchThreads: ptr(int32(4))},
	}, nil)
	red := pngPart(solidPNG(t, 256, 256, color.RGBA{R: 255, A: 255}))
	blue := pngPart(solidPNG(t, 256, 256, color.RGBA{B: 255, A: 255}))
	for _, tc := range []struct {
		first, second *ai.Part
		want          string
	}{
		{red, blue, "red"}, {blue, red, "blue"},
	} {
		response, err := genkit.Generate(t.Context(), g, ai.WithModel(model),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("First image: "), tc.first,
				ai.NewTextPart(" Second image: "), tc.second, ai.NewTextPart(" What color is the first image?"))),
			ai.WithConfig(&GenerateContentConfig{MaxOutputTokens: ptr(32), Temperature: ptr(0.0)}),
		)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("first color %s: %q", tc.want, response.Text())
		answer := strings.ToLower(response.Text())
		redAt, blueAt := strings.Index(answer, "red"), strings.Index(answer, "blue")
		firstColor := ""
		switch {
		case redAt >= 0 && (blueAt < 0 || redAt < blueAt):
			firstColor = "red"
		case blueAt >= 0:
			firstColor = "blue"
		}
		if firstColor != tc.want || response.Usage.InputTokens < 100 {
			t.Errorf("response %q (%d input tokens) did not identify first color %s",
				response.Text(), response.Usage.InputTokens, tc.want)
		}
	}
}
