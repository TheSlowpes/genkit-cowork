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
	"os"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

type testSpeechOperator struct {
	*testModelOperator
	loadDecoder func(context.Context, llama.Model, string) (mtmd.Context, error)
	freeDecoder func(context.Context, mtmd.Context) error
	generate    func(context.Context, llama.Model, mtmd.Context, string) ([]byte, error)
}

func (o *testSpeechOperator) LoadSpeechDecoder(ctx context.Context, model llama.Model, path string, _ ModelOptions, _ ContextOptions) (mtmd.Context, error) {
	return o.loadDecoder(ctx, model, path)
}

func (o *testSpeechOperator) FreeSpeechDecoder(ctx context.Context, decoder mtmd.Context) error {
	return o.freeDecoder(ctx, decoder)
}

func (o *testSpeechOperator) GenerateSpeech(ctx context.Context, model llama.Model, decoder mtmd.Context, _ ContextOptions, text string) ([]byte, error) {
	return o.generate(ctx, model, decoder, text)
}

func speechDefinition() ModelDefinition {
	return ModelDefinition{
		Name:              "voice",
		Type:              ModelTypeSpeech,
		Loading:           ModelOptions{GGUFPath: "backbone.gguf"},
		SpeechDecoderPath: "decoder.gguf",
	}
}

func speechWAVFixture() []byte {
	// One nonzero PCM sample. All lengths in the RIFF header include it.
	return []byte("RIFF\x26\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" +
		"\x00\x7d\x00\x00\x00\xfa\x00\x00\x02\x00\x10\x00data\x02\x00\x00\x00\xff\x7f")
}

func speechMedia(t *testing.T, res *ai.ModelResponse) []byte {
	t.Helper()
	if res == nil || res.Message == nil || res.Message.Role != ai.RoleModel ||
		len(res.Message.Content) != 1 || res.FinishReason != ai.FinishReasonStop {
		t.Fatalf("expected one complete model message with stop reason, got %+v", res)
	}
	part := res.Message.Content[0]
	if !part.IsMedia() || part.ContentType != "audio/wav" {
		t.Fatalf("expected audio/wav media, got %+v", part)
	}
	data, ok := strings.CutPrefix(res.Media(), "data:audio/wav;base64,")
	if !ok {
		t.Fatal("expected inline base64 WAV data URI")
	}
	wav, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		t.Fatal(err)
	}
	return wav
}

func TestSpeechGenkitContract(t *testing.T) {
	var calls []string
	op := &testSpeechOperator{
		testModelOperator: &testModelOperator{
			load: func(_ context.Context, opts ModelOptions) (llama.Model, error) {
				calls = append(calls, "load "+opts.GGUFPath)
				return 17, nil
			},
			free: func(_ context.Context, model llama.Model) error {
				calls = append(calls, "free backbone")
				if model != 17 {
					t.Errorf("free backbone handle = %d, want 17", model)
				}
				return nil
			},
		},
		loadDecoder: func(_ context.Context, model llama.Model, path string) (mtmd.Context, error) {
			calls = append(calls, "load "+path)
			if model != 17 {
				t.Errorf("decoder backbone handle = %d, want 17", model)
			}
			return 42, nil
		},
		freeDecoder: func(_ context.Context, decoder mtmd.Context) error {
			calls = append(calls, "free decoder")
			if decoder != 42 {
				t.Errorf("free decoder handle = %d, want 42", decoder)
			}
			return nil
		},
		generate: func(_ context.Context, model llama.Model, decoder mtmd.Context, text string) ([]byte, error) {
			if model != 17 || decoder != 42 {
				t.Errorf("synthesize with handles (%d, %d), want (17, 42)", model, decoder)
			}
			calls = append(calls, "say "+text)
			return speechWAVFixture(), nil
		},
	}
	p, g := initLoadingTest(t, op)
	model := p.DefineModel(g, speechDefinition(), nil)
	supports := model.(api.Action).Desc().Metadata["model"].(map[string]any)["supports"].(map[string]any)
	if diff := cmp.Diff([]string{"media"}, supports["output"]); diff != "" {
		t.Errorf("output types (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"audio/wav"}, supports["contentType"]); diff != "" {
		t.Errorf("output content types (-want +got):\n%s", diff)
	}
	if supports["media"] != false {
		t.Errorf("media input support = %v, want false", supports["media"])
	}
	if len(calls) != 0 {
		t.Fatalf("registration loaded weights: %v", calls)
	}
	if err := p.Preload(t.Context(), "voice"); err != nil {
		t.Fatal(err)
	}

	for _, config := range []any{SpeechConfig{Voice: "default"}, map[string]any{"voice": "default"}, nil} {
		res, err := genkit.Generate(t.Context(), g, ai.WithModelName("yzma/voice"),
			ai.WithPrompt("Hello, Genkit."), ai.WithConfig(config))
		if err != nil {
			t.Fatal(err)
		}
		if wav := speechMedia(t, res); !bytes.Equal(wav, speechWAVFixture()) {
			t.Errorf("response WAV = %x, want %x", wav, speechWAVFixture())
		}
	}
	if err := p.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	want := []string{"load backbone.gguf", "load decoder.gguf", "say Hello, Genkit.", "say Hello, Genkit.",
		"say Hello, Genkit.", "free decoder", "free backbone"}
	if diff := cmp.Diff(want, calls); diff != "" {
		t.Errorf("native calls (-want +got):\n%s", diff)
	}
}

func TestSpeechRejectsRequestsBeforeLoading(t *testing.T) {
	p, g := initLoadingTest(t, &testSpeechOperator{
		testModelOperator: &testModelOperator{
			load: func(context.Context, ModelOptions) (llama.Model, error) {
				t.Fatal("invalid requests must not load weights")
				return 0, nil
			},
		},
	})
	p.DefineModel(g, speechDefinition(), nil)
	for _, tt := range []struct {
		name   string
		option ai.GenerateOption
		want   string
	}{
		{"voice", ai.WithConfig(SpeechConfig{Voice: "other"}), "unsupported speech voice"},
		{"sample rate", ai.WithConfig(map[string]any{"sampleRate": 16000}), "sampleRate"},
		{"stream", ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error {
			t.Error("speech must not send a chunk")
			return nil
		}), "does not support streaming"},
		{"JSON", ai.WithOutputFormat(ai.OutputFormatJSON), "only supports WAV output"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			res, err := genkit.Generate(t.Context(), g, ai.WithModelName("yzma/voice"), ai.WithPrompt("Hello"), tt.option)
			if err == nil || !strings.Contains(err.Error(), tt.want) || res != nil {
				t.Errorf("Generate = (%v, %v), want nil result and error containing %q", res, err, tt.want)
			}
		})
	}
	model := Model(g, "voice")
	for _, tt := range []struct {
		name string
		msg  *ai.Message
		want string
	}{
		{"audio input", ai.NewUserMessage(ai.NewMediaPart("audio/wav", "data:audio/wav;base64,AA==")), "does not support media"},
		{"empty", ai.NewUserTextMessage(" "), "non-empty"},
		{"long", ai.NewUserTextMessage(strings.Repeat("a", maxSpeechBytes+1)), "exceeds"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			res, err := model.Generate(t.Context(), &ai.ModelRequest{Messages: []*ai.Message{tt.msg}}, nil)
			if err == nil || !strings.Contains(err.Error(), tt.want) || res != nil {
				t.Errorf("Generate = (%v, %v), want nil result and error containing %q", res, err, tt.want)
			}
		})
	}
}

func TestSpeechFailureCanRetry(t *testing.T) {
	var loads, frees, generations int
	loadErr := errors.New("decoder failed")
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	op := &testSpeechOperator{
		testModelOperator: &testModelOperator{
			load: func(context.Context, ModelOptions) (llama.Model, error) {
				loads++
				return 1, nil
			},
			free: func(context.Context, llama.Model) error {
				frees++
				return nil
			},
		},
		loadDecoder: func(context.Context, llama.Model, string) (mtmd.Context, error) {
			if loads == 1 {
				return 0, loadErr
			}
			return 2, nil
		},
		freeDecoder: func(context.Context, mtmd.Context) error { return nil },
		generate: func(context.Context, llama.Model, mtmd.Context, string) ([]byte, error) {
			generations++
			switch generations {
			case 1:
				cancel()
			case 2:
				return speechWAVFixture()[:44], nil // No complete PCM samples.
			}
			return speechWAVFixture(), nil
		},
	}
	p, g := initLoadingTest(t, op)
	m := p.DefineModel(g, speechDefinition(), nil)
	req := &ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("Hello")}}
	if err := p.Preload(t.Context(), "voice"); !errors.Is(err, loadErr) || frees != 1 {
		t.Fatalf("preload = %v, frees = %d; want decoder failure and backbone rollback", err, frees)
	}
	if res, err := m.Generate(ctx, req, nil); res != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled generation = (%v, %v), want no audio and context.Canceled", res, err)
	}
	if res, err := m.Generate(t.Context(), req, nil); res != nil || err == nil {
		t.Fatalf("incomplete generation = (%v, %v), want no audio and error", res, err)
	}
	res, err := m.Generate(t.Context(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	speechMedia(t, res)
	if loads != 2 {
		t.Errorf("backbone loads = %d, want 2 (reuse after canceled request)", loads)
	}
}

// TestNativeSpeech exercises a local Soprano 1.1 backbone and matching decoder
// without pulling assets or requiring media input. See the plugin README.
func TestNativeSpeech(t *testing.T) {
	backbone := os.Getenv("YZMA_TEST_SPEECH_MODEL")
	if backbone == "" {
		t.Skip("set YZMA_TEST_SPEECH_MODEL to opt into native speech synthesis")
	}
	decoder, lib := os.Getenv("YZMA_TEST_SPEECH_DECODER"), os.Getenv("YZMA_LIB")
	if decoder == "" || lib == "" {
		t.Fatal("native speech requires both YZMA_TEST_SPEECH_DECODER and YZMA_LIB")
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	p := &Yzma{LibPath: lib, ModelsDir: t.TempDir()}
	g := genkit.Init(ctx, genkit.WithPlugins(p))
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer stop()
		if err := p.Close(cleanup); err != nil {
			t.Errorf("close native speech model: %v", err)
		}
	})
	model := speechDefinition()
	cpuLayers, threads, capacity := int32(0), int32(4), uint32(256)
	model.Loading.GGUFPath, model.SpeechDecoderPath = backbone, decoder
	model.Loading.NGPULayers = &cpuLayers
	model.Context = ContextOptions{Threads: &threads, BatchThreads: &threads, ContextSize: &capacity}
	p.DefineModel(g, model, nil)
	if res, err := genkit.Generate(ctx, g, ai.WithModelName("yzma/voice"),
		ai.WithPrompt(strings.Repeat("a b ", 256))); res != nil || err == nil {
		t.Fatalf("over-capacity prompt = (%v, %v), want no audio and error", res, err)
	}
	res, err := genkit.Generate(ctx, g, ai.WithModelName("yzma/voice"), ai.WithPrompt("Hello from Genkit."),
		ai.WithConfig(SpeechConfig{Voice: "default"}))
	if err != nil {
		t.Fatal(err)
	}
	wav := speechMedia(t, res)
	if len(wav) < 32044 || bytes.Count(wav[44:], []byte{0}) == len(wav)-44 {
		t.Fatalf("native WAV has %d bytes or no nonzero samples; want an audible clip of at least half a second", len(wav))
	}
	t.Logf("generated %d PCM samples at 32000 Hz", (len(wav)-44)/2)
}
