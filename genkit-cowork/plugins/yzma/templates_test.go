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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
)

func TestJinjaPromptWithToolsAndHistory(t *testing.T) {
	const source = `{{ bos_token }}{% macro show(call) %}{{ call.function.name }}:{{ call.function.arguments.a + 1 }}:{{ call.id }}{% endmacro %}{% for tool in tools %}tool={{ tool.function.name }};{% endfor %}{% for msg in messages %}{% if msg.tool_calls %}{{ show(msg.tool_calls[0]) }}{% elif msg.role == 'tool' %}result={{ msg.content.sum }}:{{ msg.tool_call_id }}{% else %}{{ msg.content }}{% endif %};{% endfor %}{% if add_generation_prompt %}assistant{% endif %}`
	input := TextGenerationInput{
		Type: ModelTypeChat, ToolCalling: true, ChatTemplate: source, bosToken: "<bos>",
		Messages: []message.Message{
			message.Chat{Role: "user", Content: "Add"},
			templateMessage{"assistant", map[string]any{"content": "", "tool_calls": []map[string]any{
				{"id": "call-1", "function": map[string]any{"name": "add", "arguments": map[string]any{"a": 2}}},
			}}},
			templateMessage{"tool", map[string]any{"tool_call_id": "call-1", "content": struct {
				Sum int `json:"sum"`
			}{5}}},
		},
		Tools: []message.ToolDefinition{{Type: "function", Function: message.ToolFunctionDefinition{Name: "add"}}},
	}
	got, special, err := prepareTextPrompt(input, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != "<bos>tool=add;Add;add:3:call-1;result=5:call-1;assistant" || !special {
		t.Errorf("rendered prompt = %q, parseSpecial=%v", got, special)
	}
}

func TestJinjaRejectsIgnoredTools(t *testing.T) {
	_, err := renderJinjaPrompt(`{% for m in messages %}{{ m.content }}{% endfor %}`, TextGenerationInput{
		Messages: []message.Message{message.Chat{Role: "user", Content: "Use add"}},
		Tools:    []message.ToolDefinition{{Type: "function", Function: message.ToolFunctionDefinition{Name: "add"}}},
	})
	if err == nil {
		t.Error("templates must not silently ignore offered tool definitions")
	}
}

// TestNativeEmbeddedTemplate loads vocabulary/metadata only, not model weights.
// It tests the actual embedded Jinja source even for models too large for CI RAM.
func TestNativeEmbeddedTemplate(t *testing.T) {
	path, lib := os.Getenv("YZMA_TEST_MODEL"), os.Getenv("YZMA_LIB")
	if path == "" || lib == "" {
		t.Skip("set YZMA_TEST_MODEL and YZMA_LIB to test an embedded template")
	}
	if err := (defaultModelOperator{}).Init(t.Context(), lib); err != nil {
		t.Fatal(err)
	}
	params := llama.ModelDefaultParams()
	params.VocabOnly, params.NGpuLayers = 1, 0
	model, err := llama.ModelLoadFromFile(path, params)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := llama.ModelFree(model); err != nil {
			t.Error(err)
		}
	}()
	source := embeddedChatTemplate(model, "", false)
	if source == "" {
		t.Skip("this GGUF has no embedded chat template")
	}
	if embeddedChatTemplate(model, "__yzma_missing_template__", false) != "" {
		t.Error("a missing explicit template name must not silently select the default")
	}
	vocab := llama.ModelGetVocab(model)
	input := TextGenerationInput{
		Type: ModelTypeChat, bosToken: tokenSpelling(vocab, llama.VocabBOS(vocab)),
		eosToken: tokenSpelling(vocab, llama.VocabEOS(vocab)),
		Messages: []message.Message{message.Chat{Role: "user", Content: "Hello"}},
	}
	prompt, _, err := prepareTextPrompt(input, func() string { return source }, llama.ChatApplyTemplate)
	if err != nil || !strings.Contains(prompt, "Hello") {
		t.Fatalf("embedded prompt = %q, error=%v", prompt, err)
	}
	t.Logf("Rendered embedded template: %d source bytes, %d prompt bytes", len(source), len(prompt))
	architecture, _ := llama.ModelMetaValStr(model, "general.architecture")
	if architecture != "gemma4" {
		return
	}
	// Exercise discovery with the real canonical template and no companion.
	modelPath, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.Symlink(modelPath, filepath.Join(dir, "gemma.gguf")); err != nil {
		t.Fatal(err)
	}
	discovered, ok := discoverModels(t.Context(), dir, nil)["gemma"]
	if !ok || discovered.Type != ModelTypeChat || discovered.Projector != nil || discovered.ToolCalling {
		t.Fatalf("discovered Gemma = %+v, present = %v; want text-only chat without tools", discovered, ok)
	}
	input.ToolCalling = true
	input.Tools = []message.ToolDefinition{{Type: "function", Function: message.ToolFunctionDefinition{
		Name: "add", Description: "Add two integers",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{
			"a": map[string]any{"type": "integer"}, "b": map[string]any{"type": "integer"},
		}, "required": []string{"a", "b"}},
	}}}
	prompt, _, err = prepareTextPrompt(input, func() string { return source }, llama.ChatApplyTemplate)
	if err != nil || !strings.Contains(prompt, "add") {
		t.Fatalf("embedded tool template failed: %v, prompt=%q", err, prompt)
	}
	input.Messages = append(input.Messages,
		templateMessage{"assistant", map[string]any{
			"content": "", "tool_calls": []map[string]any{
				{"id": "call-1", "type": "function", "function": map[string]any{
					"name": "add", "arguments": map[string]any{"a": 2, "b": 3},
				}},
			},
		}},
		templateMessage{"tool", map[string]any{
			"name": "add", "tool_call_id": "call-1", "content": map[string]any{"sum": 5},
		}},
	)
	prompt, _, err = prepareTextPrompt(input, func() string { return source }, llama.ChatApplyTemplate)
	if err != nil || !strings.Contains(prompt, "call:add{a:2,b:3}") ||
		!strings.Contains(prompt, "response:add{sum:5}") {
		t.Fatalf("embedded tool history lost types or results: %v, prompt=%q", err, prompt)
	}
}
