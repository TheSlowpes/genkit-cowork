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
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ardanlabs/jinja"
	"github.com/hybridgroup/yzma/pkg/llama"
	yzmatemplate "github.com/hybridgroup/yzma/pkg/template"
)

// embeddedChatTemplate reads the native accessor first and then GGUF metadata.
// Named tool templates are preferred only when tools are offered and no explicit
// template name was selected. A missing explicit name never falls back silently.
func embeddedChatTemplate(model llama.Model, name string, tools bool) string {
	read := func(name string) string {
		if source := llama.ModelChatTemplate(model, name); source != "" {
			return source
		}
		key := "tokenizer.chat_template"
		if name != "" {
			key += "." + name
		}
		source, _ := llama.ModelMetaValStr(model, key)
		return source
	}
	if name == "" && tools {
		if source := read("tool_use"); source != "" {
			return source
		}
	}
	return read(name)
}

func builtinChatTemplate(name string) (string, bool) {
	return yzmatemplate.BuiltinTemplate(name)
}

// renderJinjaPrompt uses the same Jinja engine as Yzma's template package.
// Its higher-level ApplyWithOptions does not accept bos_token and eos_token,
// so this adapter supplies the complete context required by embedded templates.
// Dynamic JSON maps are necessary for model-specific Jinja fields and schemas.
func renderJinjaPrompt(source string, input TextGenerationInput) (string, error) {
	template, err := jinja.Compile(source)
	if err != nil {
		return "", fmt.Errorf("compile chat template: %w", err)
	}
	messages := make([]map[string]any, len(input.Messages))
	for i, msg := range input.Messages {
		content := make(map[string]any)
		for key, value := range msg.GetContent() {
			content[key] = value
		}
		content["role"] = msg.GetRole()
		messages[i] = content
	}
	tools := make([]map[string]any, len(input.Tools))
	for i, tool := range input.Tools {
		tools[i] = tool.GetContent()
	}
	values := map[string]any{
		"messages": messages, "tools": tools,
		"bos_token": input.bosToken, "eos_token": input.eosToken,
		"add_generation_prompt": true, "enable_thinking": false,
		"preserve_thinking": true,
	}
	// Normalize JSON-compatible Go tool outputs and schemas to values understood
	// by the Jinja engine, including structs returned by Genkit tools.
	raw, err := json.Marshal(values)
	if err != nil {
		return "", fmt.Errorf("encode template context: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&values); err != nil {
		return "", err
	}
	normalized, err := jinjaNumbers(values)
	if err != nil {
		return "", err
	}
	values = normalized.(map[string]any)
	prompt, err := template.Render(values)
	if err != nil {
		return "", fmt.Errorf("render chat template: %w", err)
	}
	if len(input.Tools) > 0 {
		values["tools"] = []any{}
		withoutTools, withoutErr := template.Render(values)
		if withoutErr == nil && withoutTools == prompt {
			return "", fmt.Errorf("chat template ignores tool definitions; select a tool-aware embedded template or provide an override")
		}
	}
	if prompt == "" || strings.ContainsRune(prompt, '\x00') {
		return "", fmt.Errorf("chat template produced empty text or a NUL byte")
	}
	return prompt, nil
}

// jinjaNumbers converts json.Number to the engine's supported numeric types,
// preserving int64 values instead of rounding tool results through float64.
func jinjaNumbers(value any) (any, error) {
	switch value := value.(type) {
	case json.Number:
		if !strings.ContainsAny(string(value), ".eE") {
			n, err := value.Int64()
			if err != nil {
				return nil, fmt.Errorf("template integer is outside int64: %w", err)
			}
			return n, nil
		}
		return value.Float64()
	case map[string]any:
		for key, child := range value {
			normalized, err := jinjaNumbers(child)
			if err != nil {
				return nil, err
			}
			value[key] = normalized
		}
	case []any:
		for i, child := range value {
			normalized, err := jinjaNumbers(child)
			if err != nil {
				return nil, err
			}
			value[i] = normalized
		}
	}
	return value, nil
}

func tokenSpelling(vocab llama.Vocab, token llama.Token) string {
	if token < 0 {
		return ""
	}
	return llama.VocabGetText(vocab, token)
}
