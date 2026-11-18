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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
)

func TestPrepareCompletionPrompt(t *testing.T) {
	const raw = "Once upon a time\n<|im_start|>literal text "
	input := TextGenerationInput{
		Type:     ModelTypeCompletion,
		Messages: []message.Message{message.Chat{Role: "user", Content: raw}},
	}
	got, parseSpecial, err := prepareTextPrompt(input, func() string {
		t.Fatal("completion must not read chat-template metadata")
		return ""
	}, func(string, []llama.ChatMessage, bool, []byte) int32 {
		t.Fatal("completion must not apply a chat template")
		return 0
	})
	if err != nil || got != raw || parseSpecial {
		t.Fatalf("completion prompt = %q, parseSpecial=%v, err=%v", got, parseSpecial, err)
	}
}

func TestPrepareChatPrompt(t *testing.T) {
	for _, override := range []string{"", "override"} {
		t.Run("template="+override, func(t *testing.T) {
			readMetadata := false
			input := TextGenerationInput{
				Type: ModelTypeChat, ChatTemplate: override,
				Messages: []message.Message{message.Chat{Role: "user", Content: "Hello"}},
			}
			got, parseSpecial, err := prepareTextPrompt(input, func() string {
				readMetadata = true
				return "embedded"
			}, func(template string, chat []llama.ChatMessage, add bool, buf []byte) int32 {
				want := "embedded"
				if override != "" {
					want = override
				}
				if template != want || !add || len(chat) != 1 {
					t.Errorf("unexpected chat formatting: %q, add=%v, messages=%d", template, add, len(chat))
				}
				return int32(copy(buf, "<assistant>"))
			})
			if err != nil || got != "<assistant>" || !parseSpecial || readMetadata != (override == "") {
				t.Fatalf("chat prompt = %q, parseSpecial=%v, metadata=%v, err=%v", got, parseSpecial, readMetadata, err)
			}
		})
	}
	_, _, err := prepareTextPrompt(TextGenerationInput{Type: ModelTypeChat},
		func() string { return "" }, nil)
	if err == nil {
		t.Error("chat without a template must fail")
	}
}

func TestCompletionThroughGenkit(t *testing.T) {
	const raw = "Once upon a time"
	var calls int
	op := &testTextOperator{
		testModelOperator: &testModelOperator{
			load: func(context.Context, ModelOptions) (llama.Model, error) { return 42, nil },
		},
		generate: func(ctx context.Context, model llama.Model, input TextGenerationInput, onText func(context.Context, string) error) (TextGenerationResult, error) {
			calls++
			if input.Type != ModelTypeCompletion || input.ChatTemplate != "" ||
				len(input.Messages) != 1 || input.Messages[0].GetContent()["content"] != raw {
				t.Errorf("completion input changed: %+v", input)
			}
			if onText != nil {
				if err := onText(ctx, " there was a village."); err != nil {
					return TextGenerationResult{}, err
				}
			}
			return TextGenerationResult{Text: " there was a village.", InputTokens: 4, OutputTokens: 5}, nil
		},
	}
	plugin, g := initLoadingTest(t, op)
	def := testModelDefinition()
	def.Type = ModelTypeCompletion
	model := plugin.DefineModel(g, def, nil)
	var streamed strings.Builder
	for value, err := range genkit.GenerateStream(t.Context(), g,
		ai.WithModelName("yzma/local-chat"), ai.WithPrompt(raw),
	) {
		if err != nil {
			t.Fatal(err)
		}
		if value.Done {
			if value.Response.Text() != streamed.String() || value.Response.Text() != " there was a village." {
				t.Errorf("completion response = %q, stream=%q", value.Response.Text(), streamed.String())
			}
		} else {
			streamed.WriteString(value.Chunk.Text())
		}
	}
	for name, messages := range map[string][]*ai.Message{
		"system":    {ai.NewSystemTextMessage("Be concise")},
		"assistant": {ai.NewModelTextMessage("An old response")},
		"history":   {ai.NewUserTextMessage("First"), ai.NewUserTextMessage("Second")},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := model.Generate(t.Context(), ai.NewModelRequest(nil, messages...), nil); err == nil {
				t.Error("completion must reject system messages, assistant messages, and history")
			}
		})
	}
	if calls != 1 {
		t.Errorf("inference calls = %d, want 1; invalid requests must not reach inference", calls)
	}
}

func TestModelTypeValidationAndCapabilities(t *testing.T) {
	op := &testModelOperator{
		load: func(context.Context, ModelOptions) (llama.Model, error) {
			t.Error("registration must not load weights")
			return 42, nil
		},
	}
	plugin, g := initLoadingTest(t, op)
	for _, kind := range []ModelType{"", "unknown", ModelTypeChat, ModelTypeCompletion} {
		t.Run("type="+string(kind), func(t *testing.T) {
			def := testModelDefinition()
			def.Name, def.Type = "model-"+string(kind), kind
			valid := kind == ModelTypeChat || kind == ModelTypeCompletion
			defer func() {
				if recovered := recover(); (recovered == nil) != valid {
					t.Errorf("registration panic = %v, valid=%v", recovered, valid)
				}
			}()
			plugin.DefineModel(g, def, nil)
			supports := modelSupports(kind, false)
			if supports.Multiturn != (kind == ModelTypeChat) || supports.SystemRole != (kind == ModelTypeChat) ||
				supports.Tools || supports.Media {
				t.Errorf("incorrect capabilities for %s: %+v", kind, supports)
			}
		})
	}
	for _, tc := range []struct {
		name string
		def  ModelDefinition
		opts *ai.ModelOptions
	}{
		{"completion template", ModelDefinition{Name: "bad-template", Type: ModelTypeCompletion, ChatTemplate: "chatml"}, nil},
		{"completion tools", ModelDefinition{Name: "bad-tools", Type: ModelTypeCompletion, ToolCalling: true}, nil},
		{"completion named template", ModelDefinition{Name: "bad-name", Type: ModelTypeCompletion, ChatTemplateName: "tool_use"}, nil},
		{"conflicting templates", ModelDefinition{Name: "both", Type: ModelTypeChat, ChatTemplate: "chatml", ChatTemplateName: "tool_use"}, nil},
		{"capability override", ModelDefinition{Name: "bad-caps", Type: ModelTypeCompletion}, &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Error("contradictory mode settings must be rejected")
				}
			}()
			plugin.DefineModel(g, tc.def, tc.opts)
		})
	}
}
