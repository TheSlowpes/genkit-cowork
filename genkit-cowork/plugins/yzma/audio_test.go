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
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
	"github.com/hybridgroup/yzma/pkg/llama"
)

func wavFixture(samples int) []byte {
	le := binary.LittleEndian
	b := make([]byte, 44+samples*2)
	copy(b, "RIFF")
	le.PutUint32(b[4:8], uint32(len(b)-8))
	copy(b[8:], "WAVEfmt ")
	le.PutUint32(b[16:20], 16)
	le.PutUint16(b[20:22], 1)
	le.PutUint16(b[22:24], 1)
	le.PutUint32(b[24:28], 16000)
	le.PutUint32(b[28:32], 32000)
	le.PutUint16(b[32:34], 2)
	le.PutUint16(b[34:36], 16)
	copy(b[36:], "data")
	le.PutUint32(b[40:44], uint32(samples*2))
	return b
}

func wavPart(mime string, data []byte) *ai.Part {
	return ai.NewMediaPart(mime, "data:"+mime+";base64,"+base64.StdEncoding.EncodeToString(data))
}

func TestAudioInputGenkitOrder(t *testing.T) {
	first, second := wavFixture(100), wavFixture(120)
	first[44], second[44] = 17, 91
	op := &testVisionOperator{
		testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) { return 42, nil }},
		loadProjector: func(_ context.Context, _ llama.Model, def ModelDefinition) (ProjectorHandle, error) {
			if !def.Projector.Audio || def.Projector.Images {
				t.Fatalf("projector options = %+v, want audio only", def.Projector)
			}
			return 17, nil
		},
		freeProjector: func(context.Context, ProjectorHandle) error { return nil },
		generate: func(_ context.Context, _ llama.Model, _ ProjectorHandle, input MediaGenerationInput) (TextGenerationResult, error) {
			// The native path consumes these same spans and payloads in order.
			input.ChatTemplate = `{{ messages[0].content }}`
			spans, parts, err := renderMediaPrompt(input, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff([]string{"first ", " second ", " done"}, spans); diff != "" {
				t.Errorf("text spans (-want +got):\n%s", diff)
			}
			if len(parts) != 2 || parts[0].MIMEType != "audio/wav" || parts[1].MIMEType != "audio/x-wav" ||
				!bytes.Equal(parts[0].pcm16, first[44:]) || !bytes.Equal(parts[1].pcm16, second[44:]) {
				t.Fatal("rendered payloads did not preserve the two validated clips in order")
			}
			return TextGenerationResult{Text: "Transcript", InputTokens: 25, OutputTokens: 1}, nil
		},
	}
	p, g := initDiscoveryTest(t, t.TempDir(), op)
	def := testModelDefinition()
	def.Projector = &ProjectorOptions{GGUFPath: "audio.gguf", Audio: true}
	model := p.DefineModel(g, def, nil)
	res, err := genkit.Generate(t.Context(), g, ai.WithModel(model), ai.WithMessages(ai.NewUserMessage(
		ai.NewTextPart("first "), wavPart("audio/wav", first),
		ai.NewTextPart(" second "), wavPart("audio/x-wav", second), ai.NewTextPart(" done"),
	)))
	if err != nil {
		t.Fatal(err)
	}
	if got := res.Text(); got != "Transcript" {
		t.Errorf("response = %q, want Transcript", got)
	}
}

func TestAudioInputRejectsInvalidMediaBeforeLoad(t *testing.T) {
	const mime = "audio/wav"
	valid := wavFixture(100)
	stereo := bytes.Clone(valid)
	binary.LittleEndian.PutUint16(stereo[22:24], 2)
	badRate := bytes.Clone(valid)
	binary.LittleEndian.PutUint32(badRate[24:28], 44100)
	badChunk := bytes.Clone(valid)
	binary.LittleEndian.PutUint32(badChunk[40:44], ^uint32(0))
	p, g := initDiscoveryTest(t, t.TempDir(), &testVisionOperator{
		testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
			t.Fatal("invalid media must not load the model")
			return 0, nil
		}},
	})
	def := testModelDefinition()
	def.Projector = &ProjectorOptions{GGUFPath: "audio.gguf", Audio: true}
	model := p.DefineModel(g, def, nil)
	for _, tc := range []struct {
		name string
		part *ai.Part
		want string
	}{
		{"remote", ai.NewMediaPart(mime, "https://example.com/clip.wav"), "inline base64"},
		{"local", ai.NewMediaPart(mime, "file:///tmp/clip.wav"), "inline base64"},
		{"MIME mismatch", ai.NewMediaPart("audio/x-wav", wavPart(mime, valid).Text), "content type does not match"},
		{"unsupported format", wavPart("audio/mpeg", valid), "unsupported media MIME type"},
		{"bad base64", ai.NewMediaPart(mime, "data:audio/wav;base64,!!!!"), "invalid media base64"},
		{"wrong container", wavPart(mime, []byte("not WAV")), "invalid RIFF"},
		{"truncated", wavPart(mime, valid[:len(valid)-2]), "invalid RIFF"},
		{"truncated chunk", wavPart(mime, badChunk), "truncated WAV chunk"},
		{"empty audio", wavPart(mime, wavFixture(0)), "non-empty PCM"},
		{"stereo", wavPart(mime, stereo), "mono 16 kHz"},
		{"sample rate", wavPart(mime, badRate), "mono 16 kHz"},
		{"byte limit", wavPart(mime, make([]byte, maxAudioBytes+1)), "1048576-byte limit"},
		{"duration limit", wavPart(mime, wavFixture(audioSampleRate*maxAudioSeconds+1)), "30-second limit"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := genkit.Generate(t.Context(), g, ai.WithModel(model), ai.WithMessages(ai.NewUserMessage(tc.part)))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("generation error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestConstrainedMediaRejectsAudioBeforeLoad(t *testing.T) {
	op := &testConstrainedMediaOperator{testVisionOperator: &testVisionOperator{
		testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
			t.Fatal("constrained audio must not load weights")
			return 0, nil
		}},
	}}
	local, g := initLoadingTest(t, op)
	def := testModelDefinition()
	def.Projector = &ProjectorOptions{GGUFPath: "manual.gguf", Images: true, Audio: true}
	model := local.DefineModel(g, def, nil)
	res, err := model.Generate(t.Context(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(wavPart("audio/wav", wavFixture(100)))},
		Output: &ai.ModelOutputConfig{
			Constrained: true, Format: ai.OutputFormatJSON,
			Schema: map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false},
		},
	}, nil)
	if res != nil || err == nil || !strings.Contains(err.Error(), "constrained media output supports only image input") {
		t.Fatalf("Generate = (%v, %v), want no response and an image-only error", res, err)
	}
}

func TestAudioProjectorCompatibility(t *testing.T) {
	for _, tc := range []struct {
		name, decoder, projectorType string
		hasAudio, width              uint64
		want                         string
	}{
		{"matches", "Qwen3-ASR", "qwen3a", 1, 1024, ""},
		{"vision only", "Qwen3-ASR", "idefics3", 0, 1024, "no audio encoder"},
		{"other projector", "Qwen3-ASR", "ultravox", 1, 1024, "matching qwen3a projector"},
		{"unrelated same width", "Qwen3-VL", "qwen3a", 1, 1024, "Qwen3-ASR decoder"},
		{"wrong width", "Qwen3-ASR", "qwen3a", 1, 2048, "width does not match"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			header := ggufHeader{
				values: map[string]ggufValue{
					"general.architecture":      {kind: ggufString, text: "clip"},
					"clip.has_audio_encoder":    {kind: 7, num: tc.hasAudio},
					"clip.audio.projector_type": {kind: ggufString, text: tc.projectorType},
					"clip.audio.projection_dim": {kind: 4, num: tc.width},
				},
				tensors: map[string]bool{"a.conv_out.weight": true},
			}
			err := validateAudioPair(map[string]ggufValue{
				"general.architecture":     {kind: ggufString, text: "qwen3vl"},
				"general.basename":         {kind: ggufString, text: tc.decoder},
				"qwen3vl.embedding_length": {kind: 4, num: 1024},
			}, header)
			switch {
			case tc.want == "" && err != nil:
				t.Errorf("validate pair: %v", err)
			case tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)):
				t.Errorf("validate pair error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestNativeAudioInput(t *testing.T) {
	modelPath := os.Getenv("YZMA_TEST_AUDIO_MODEL")
	if modelPath == "" {
		t.Skip("run task test:yzma:audio:native with the pair and WAV sample in the plugin README")
	}
	projectorPath, audioPath := os.Getenv("YZMA_TEST_AUDIO_PROJECTOR"), os.Getenv("YZMA_TEST_AUDIO_WAV")
	if projectorPath == "" || audioPath == "" {
		t.Fatal("set YZMA_TEST_AUDIO_PROJECTOR and YZMA_TEST_AUDIO_WAV")
	}
	data, err := os.ReadFile(audioPath)
	if err != nil {
		t.Fatal(err)
	}
	p := &Yzma{ModelsDir: filepath.Dir(modelPath)}
	g := genkit.Init(t.Context(), genkit.WithPlugins(p))
	t.Cleanup(func() {
		if err := p.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	model := p.DefineModel(g, ModelDefinition{
		Name: "transcribe", Type: ModelTypeChat,
		Loading:   ModelOptions{GGUFPath: modelPath, NGPULayers: ptr(int32(0))},
		Projector: &ProjectorOptions{GGUFPath: projectorPath, Audio: true},
		Context:   ContextOptions{ContextSize: ptr(uint32(2048)), Threads: ptr(int32(2)), BatchThreads: ptr(int32(2))},
		ChatTemplate: `<|im_start|>system
<|im_end|>
<|im_start|>user
{{ messages[0].content }}<|im_end|>
<|im_start|>assistant
language English<asr_text>`,
	}, nil)
	res, err := genkit.Generate(t.Context(), g, ai.WithModel(model),
		ai.WithMessages(ai.NewUserMessage(wavPart("audio/wav", data))),
		ai.WithConfig(&GenerateContentConfig{MaxOutputTokens: ptr(96), Temperature: ptr(0.0)}),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("transcript (%d input tokens): %q", res.Usage.InputTokens, res.Text())
	if text := strings.ToLower(res.Text()); !strings.Contains(text, "fox") || !strings.Contains(text, "dog") || res.Usage.InputTokens < 40 {
		t.Errorf("expected words from the sample and audio tokens, got %q (%d input tokens)", res.Text(), res.Usage.InputTokens)
	}
}
