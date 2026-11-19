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
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
)

type testTextOperator struct {
	*testModelOperator
	generate func(context.Context, llama.Model, TextGenerationInput, func(context.Context, string) error) (TextGenerationResult, error)
}

func (o *testTextOperator) GenerateText(ctx context.Context, model llama.Model, input TextGenerationInput, onText func(context.Context, string) error) (TextGenerationResult, error) {
	return o.generate(ctx, model, input, onText)
}

func TestGenerateWithGenkit(t *testing.T) {
	wantMessages := []message.Message{
		message.Chat{Role: "system", Content: "Be concise."},
		message.Chat{Role: "user", Content: "Hello"},
		message.Chat{Role: "assistant", Content: "Hi!"},
		message.Chat{Role: "user", Content: "Explain Go."},
	}
	op := &testTextOperator{
		testModelOperator: &testModelOperator{
			load: func(context.Context, ModelOptions) (llama.Model, error) { return 42, nil },
		},
		generate: func(ctx context.Context, model llama.Model, input TextGenerationInput, onText func(context.Context, string) error) (TextGenerationResult, error) {
			if model != 42 {
				t.Errorf("model = %d, want 42", model)
			}
			if diff := cmp.Diff(wantMessages, input.Messages); diff != "" {
				t.Errorf("messages (-want +got):\n%s", diff)
			}
			if input.Sampler.Temp != 0 || input.Sampler.Seed != 0 ||
				input.MaxOutputTokens != 16 || input.ChatTemplate != "chatml" ||
				*input.Context.ContextSize != 4096 {
				t.Errorf("resolved input = %+v", input)
			}
			return TextGenerationResult{Text: "Go is a language.", InputTokens: 12, OutputTokens: 5, Limited: true}, nil
		},
	}
	plugin, g := initLoadingTest(t, op)
	def := testModelDefinition()
	def.ChatTemplate, def.Context.ContextSize = "chatml", ptr(uint32(4096))
	def.GenerationDefaults = &GenerateContentConfig{Temperature: ptr(0.7), MaxOutputTokens: ptr(16)}
	plugin.DefineModel(g, def, nil)
	user := ai.NewUserTextMessage("Explain ")
	user.Content = append(user.Content, ai.NewTextPart("Go."))
	response, err := genkit.Generate(t.Context(), g,
		ai.WithModelName("yzma/local-chat"),
		ai.WithMessages(ai.NewSystemTextMessage("Be concise."), ai.NewUserTextMessage("Hello"),
			ai.NewModelTextMessage("Hi!"), user),
		ai.WithConfig(&GenerateContentConfig{Temperature: ptr(0.0), Seed: ptr(uint32(0))}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() != "Go is a language." || response.Message.Role != ai.RoleModel ||
		response.FinishReason != ai.FinishReasonLength {
		t.Errorf("response = %+v", response)
	}
	if diff := cmp.Diff(&ai.GenerationUsage{
		InputTokens: 12, OutputTokens: 5, TotalTokens: 17,
		InputCharacters: 30, OutputCharacters: 17,
	}, response.Usage); diff != "" {
		t.Errorf("usage (-want +got):\n%s", diff)
	}
}

func TestGenerateStreamingAndCallbackErrors(t *testing.T) {
	op := &testTextOperator{
		testModelOperator: &testModelOperator{
			load: func(context.Context, ModelOptions) (llama.Model, error) { return 42, nil },
		},
		generate: func(ctx context.Context, model llama.Model, input TextGenerationInput, onText func(context.Context, string) error) (TextGenerationResult, error) {
			for _, text := range []string{"Hello ", "world"} {
				if onText != nil {
					if err := onText(ctx, text); err != nil {
						return TextGenerationResult{}, err
					}
				}
			}
			return TextGenerationResult{Text: "Hello world", InputTokens: 3, OutputTokens: 2}, nil
		},
	}
	plugin, g := initLoadingTest(t, op)
	model := plugin.DefineModel(g, testModelDefinition(), nil)
	request := ai.NewModelRequest(map[string]any{"temperature": 0}, ai.NewUserTextMessage("Hello"))
	var streamed strings.Builder
	response, err := model.Generate(t.Context(), request, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		for _, part := range chunk.Content {
			streamed.WriteString(part.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() != streamed.String() || response.Text() != "Hello world" ||
		response.FinishReason != ai.FinishReasonStop {
		t.Errorf("stream = %q, response = %+v", streamed.String(), response)
	}
	streamed.Reset()
	var final *ai.ModelResponse
	for value, err := range genkit.GenerateStream(t.Context(), g,
		ai.WithModelName("yzma/local-chat"), ai.WithPrompt("Hello"),
		ai.WithConfig(&GenerateContentConfig{Temperature: ptr(0.0)}),
	) {
		if err != nil {
			t.Fatal(err)
		}
		if value.Done {
			final = value.Response
		} else {
			streamed.WriteString(value.Chunk.Text())
		}
	}
	if final == nil || final.Text() != "Hello world" || streamed.String() != final.Text() {
		t.Errorf("GenerateStream returned incomplete text: %q, final=%v", streamed.String(), final)
	}
	callbackErr := errors.New("client disconnected")
	_, err = model.Generate(t.Context(), request, func(context.Context, *ai.ModelResponseChunk) error {
		return callbackErr
	})
	if !errors.Is(err, callbackErr) {
		t.Errorf("callback error = %v, want original error", err)
	}
	if err := plugin.Preload(t.Context(), "local-chat"); err != nil {
		t.Errorf("model reservation not released: %v", err)
	}
}

func TestTextMessagesRejectsUnsupportedContent(t *testing.T) {
	for name, req := range map[string]*ai.ModelRequest{
		"nil":         nil,
		"empty":       {},
		"nil message": {Messages: []*ai.Message{nil}},
		"empty text":  {Messages: []*ai.Message{ai.NewUserTextMessage("")}},
		"NUL":         {Messages: []*ai.Message{ai.NewUserTextMessage("a\x00b")}},
		"tool role":   {Messages: []*ai.Message{{Role: ai.RoleTool, Content: []*ai.Part{ai.NewTextPart("x")}}}},
		"nil part":    {Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{nil}}}},
		"media":       {Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewMediaPart("image/png", "x")}}}},
		"tools":       {Messages: []*ai.Message{ai.NewUserTextMessage("x")}, Tools: []*ai.ToolDefinition{{Name: "test"}}},
		"required":    {Messages: []*ai.Message{ai.NewUserTextMessage("x")}, ToolChoice: ai.ToolChoiceRequired},
		"JSON":        {Messages: []*ai.Message{ai.NewUserTextMessage("x")}, Output: &ai.ModelOutputConfig{Format: "json"}},
		"schema":      {Messages: []*ai.Message{ai.NewUserTextMessage("x")}, Output: &ai.ModelOutputConfig{Constrained: true}},
		"docs":        {Messages: []*ai.Message{ai.NewUserTextMessage("x")}, Docs: []*ai.Document{{}}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := textMessages(req); err == nil {
				t.Fatal("expected an unsupported-input error")
			}
		})
	}
}

func TestDecodeGenerationConfig(t *testing.T) {
	for name, cfg := range map[string]any{
		"typed": GenerateContentConfig{Temperature: ptr(0.0), Seed: ptr(uint32(0)), StopSequences: ptr([]string{})},
		"map":   map[string]any{"temperature": 0.0, "seed": 0, "stopSequences": []string{}},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := decodeGenerationConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if got.Temperature == nil || *got.Temperature != 0 || got.Seed == nil ||
				*got.Seed != 0 || got.StopSequences == nil || len(*got.StopSequences) != 0 {
				t.Errorf("explicit zero/empty config lost: %+v", got)
			}
		})
	}
	for _, cfg := range []any{
		map[string]any{"unknown": true},
		map[string]any{"maxOutputTokens": 1.5},
		map[string]any{"temperature": "hot"},
		"invalid",
	} {
		if _, err := decodeGenerationConfig(cfg); err == nil {
			t.Errorf("invalid config %v was accepted", cfg)
		}
	}
}

func TestRenderChatPrompt(t *testing.T) {
	want := strings.Repeat("x", 5000)
	var calls int
	got, err := renderChatPrompt("chatml", nil, func(template string, chat []llama.ChatMessage, add bool, buf []byte) int32 {
		calls++
		if template != "chatml" || !add {
			t.Error("template or assistant generation marker changed")
		}
		copy(buf, want)
		return int32(len(want))
	})
	if err != nil || got != want || calls != 2 {
		t.Fatalf("render = %q, %v, calls=%d; want resized prompt", got, err, calls)
	}
	if _, err := renderChatPrompt("unknown", nil, func(string, []llama.ChatMessage, bool, []byte) int32 {
		return -1
	}); err == nil {
		t.Error("unsupported template must return an error, not substitute another format")
	}
}

type scriptedTokenSession struct {
	tokens    []llama.Token
	pieces    map[llama.Token]string
	decoded   [][]llama.Token
	next      int
	decodeErr int
	onSample  func()
}

func (s *scriptedTokenSession) Decode(ctx context.Context, tokens []llama.Token) error {
	s.decoded = append(s.decoded, slices.Clone(tokens))
	if len(s.decoded) == s.decodeErr {
		return fmt.Errorf("decode failed")
	}
	return ctx.Err()
}

func (s *scriptedTokenSession) Sample(ctx context.Context) (llama.Token, error) {
	if s.onSample != nil {
		s.onSample()
	}
	if s.next >= len(s.tokens) {
		return 0, fmt.Errorf("token script exhausted")
	}
	token := s.tokens[s.next]
	s.next++
	return token, nil
}

func (s *scriptedTokenSession) IsEnd(token llama.Token) bool { return token == 99 }

func (s *scriptedTokenSession) Piece(token llama.Token) (string, error) {
	piece, ok := s.pieces[token]
	if !ok {
		return "", fmt.Errorf("missing token piece")
	}
	return piece, nil
}

func TestGenerateTokens(t *testing.T) {
	for _, tc := range []struct {
		name     string
		pieces   []string
		eos      bool
		stops    []string
		limit    int
		capacity int
		want     TextGenerationResult
	}{
		{"EOS", []string{"hello"}, true, nil, 8, 16, TextGenerationResult{Text: "hello", InputTokens: 3, OutputTokens: 2}},
		{"token limit", []string{"a", "b"}, false, nil, 2, 16, TextGenerationResult{Text: "ab", InputTokens: 3, OutputTokens: 2, Limited: true}},
		{"context limit", []string{"a"}, false, nil, 8, 4, TextGenerationResult{Text: "a", InputTokens: 3, OutputTokens: 1, Limited: true}},
		{"split stop", []string{"hello<ST", "OP>ignored"}, false, []string{"<STOP>"}, 8, 16, TextGenerationResult{Text: "hello", InputTokens: 3, OutputTokens: 2}},
		{"partial stop at EOS", []string{"hello<ST"}, true, []string{"<STOP>"}, 8, 16, TextGenerationResult{Text: "hello<ST", InputTokens: 3, OutputTokens: 2}},
		{"split UTF8", []string{"\xe2", "\x82", "\xac"}, true, nil, 8, 16, TextGenerationResult{Text: "€", InputTokens: 3, OutputTokens: 4}},
		{"incomplete UTF8", []string{"\xe2"}, false, nil, 1, 16, TextGenerationResult{Text: "\uFFFD", InputTokens: 3, OutputTokens: 1, Limited: true}},
		{"earliest stop", []string{"helloABCtail"}, false, []string{"BC", "ABC"}, 8, 16, TextGenerationResult{Text: "hello", InputTokens: 3, OutputTokens: 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			session := &scriptedTokenSession{pieces: make(map[llama.Token]string)}
			for i, piece := range tc.pieces {
				token := llama.Token(i + 1)
				session.tokens = append(session.tokens, token)
				session.pieces[token] = piece
			}
			if tc.eos {
				session.tokens = append(session.tokens, 99)
			}
			var streamed strings.Builder
			got, err := generateTokens(t.Context(), session, []llama.Token{10, 11, 12}, 2, tc.capacity,
				TextGenerationInput{MaxOutputTokens: tc.limit, StopSequences: tc.stops},
				func(ctx context.Context, text string) error {
					if !utf8.ValidString(text) {
						t.Errorf("invalid UTF-8 chunk: %q", text)
					}
					streamed.WriteString(text)
					return nil
				})
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("result (-want +got):\n%s", diff)
			}
			if streamed.String() != got.Text {
				t.Errorf("stream = %q, response = %q", streamed.String(), got.Text)
			}
			if diff := cmp.Diff([][]llama.Token{{10, 11}, {12}}, session.decoded[:2]); diff != "" {
				t.Errorf("prompt batching (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNativeTokenSessionRejectsExplicitPositionBatch(t *testing.T) {
	position := llama.Pos(12)
	session := nativeTokenSession{position: &position}
	err := session.Decode(t.Context(), []llama.Token{1, 2})
	if err == nil || !strings.Contains(err.Error(), "requires exactly one token") {
		t.Fatalf("Decode = %v, want a single-token requirement before calling llama.cpp", err)
	}
}

func TestGenerateTokensErrorsAndCancellation(t *testing.T) {
	for _, decodeAt := range []int{1, 2} {
		session := &scriptedTokenSession{
			tokens: []llama.Token{1, 99}, pieces: map[llama.Token]string{1: "a"}, decodeErr: decodeAt,
		}
		if _, err := generateTokens(t.Context(), session, []llama.Token{10}, 2, 16,
			TextGenerationInput{MaxOutputTokens: 8}, nil); err == nil {
			t.Errorf("decode error at call %d was ignored", decodeAt)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	session := &scriptedTokenSession{tokens: []llama.Token{1}, pieces: map[llama.Token]string{1: "a"}, onSample: cancel}
	_, err := generateTokens(ctx, session, []llama.Token{10}, 2, 16, TextGenerationInput{MaxOutputTokens: 8}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("cancellation = %v, want context.Canceled", err)
	}
	callbackErr := errors.New("disconnected")
	session = &scriptedTokenSession{tokens: []llama.Token{1}, pieces: map[llama.Token]string{1: "a"}}
	_, err = generateTokens(t.Context(), session, []llama.Token{10}, 2, 16,
		TextGenerationInput{MaxOutputTokens: 8}, func(context.Context, string) error { return callbackErr })
	if !errors.Is(err, callbackErr) {
		t.Errorf("callback error = %v, want original error", err)
	}
}

// TestNativeTextGeneration exercises request-local contexts and streaming with an
// existing GGUF file. No model or native library is downloaded by the test.
func TestNativeTextGeneration(t *testing.T) {
	path, lib := os.Getenv("YZMA_TEST_MODEL"), os.Getenv("YZMA_LIB")
	if path == "" || lib == "" {
		t.Skip("set YZMA_TEST_MODEL and YZMA_LIB to exercise native text generation")
	}
	plugin := &Yzma{LibPath: lib}
	g := genkit.Init(t.Context(), genkit.WithPlugins(plugin))
	t.Cleanup(func() {
		if err := plugin.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	def := testModelDefinition()
	def.Loading.GGUFPath, def.Loading.NGPULayers = path, ptr(int32(0))
	def.Context.ContextSize = ptr(uint32(512))
	if kind := os.Getenv("YZMA_TEST_TYPE"); kind != "" {
		def.Type = ModelType(kind)
	}
	def.ChatTemplate = os.Getenv("YZMA_TEST_TEMPLATE")
	model := plugin.DefineModel(g, def, nil)
	request := ai.NewModelRequest(&GenerateContentConfig{Temperature: ptr(0.0), MaxOutputTokens: ptr(8)},
		ai.NewUserTextMessage("Say hello."))
	for range 2 {
		var streamed strings.Builder
		result, err := model.Generate(t.Context(), request, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
			for _, part := range chunk.Content {
				streamed.WriteString(part.Text)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if result.Text() != streamed.String() || !utf8.ValidString(result.Text()) ||
			result.Usage.InputTokens <= 0 || result.Usage.OutputTokens <= 0 || result.Usage.OutputTokens > 8 {
			t.Errorf("invalid native response: %+v, streamed=%q", result, streamed.String())
		}
	}
}
