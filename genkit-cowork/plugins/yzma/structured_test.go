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
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
	"github.com/hybridgroup/yzma/pkg/llama"
)

type constrainedTestOperator struct {
	*testTextOperator
	generateConstrained func(TextGenerationInput, string) TextGenerationResult
}

func (o *constrainedTestOperator) GenerateConstrainedText(
	_ context.Context, _ llama.Model, input TextGenerationInput, grammar string,
) (TextGenerationResult, error) {
	return o.generateConstrained(input, grammar), nil
}

type structuredResult struct {
	Category string `json:"category" jsonschema:"enum=parcel,enum=letter"`
	Fragile  []bool `json:"fragile"`
	Details  struct {
		Pieces int `json:"pieces"`
	} `json:"details"`
	Label string `json:"label,omitempty"`
}

func TestGenerateDataWithConstraint(t *testing.T) {
	const valid = `{"category":"parcel","details":{"pieces":2},"fragile":[true,false]}`
	for _, tc := range []struct {
		name   string
		result TextGenerationResult
		label  string
		err    string
	}{
		{name: "typed output", result: TextGenerationResult{Text: valid}},
		{
			name:   "Markdown inside a string is preserved",
			result: TextGenerationResult{Text: valid[:len(valid)-1] + `,"label":"` + "```json {} ```" + `"}`},
			label:  "```json {} ```",
		},
		{name: "missing property", result: TextGenerationResult{Text: `{"category":"parcel","fragile":[]}`}, err: "does not match the schema"},
		{name: "extra property", result: TextGenerationResult{Text: `{"category":"parcel","details":{"pieces":2,"extra":false},"fragile":[]}`}, err: "does not match the schema"},
		{name: "invalid enum", result: TextGenerationResult{Text: strings.Replace(valid, "parcel", "crate", 1)}, err: "does not match the schema"},
		{name: "wrong item type", result: TextGenerationResult{Text: strings.Replace(valid, "true", `"true"`, 1)}, err: "does not match the schema"},
		{name: "truncated JSON", result: TextGenerationResult{Text: valid[:len(valid)-1]}, err: "one complete JSON"},
		{name: "trailing text", result: TextGenerationResult{Text: valid + " extra"}, err: "one complete JSON"},
		{name: "fenced JSON", result: TextGenerationResult{Text: "```json\n" + valid + "\n```"}, err: "one complete JSON"},
		{name: "exhausted after complete JSON", result: TextGenerationResult{Text: valid, Limited: true}, err: "incomplete constrained output"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			op := &constrainedTestOperator{
				testTextOperator: &testTextOperator{
					testModelOperator: &testModelOperator{
						load: func(context.Context, ModelOptions) (llama.Model, error) { return 42, nil },
					},
				},
				generateConstrained: func(input TextGenerationInput, grammar string) TextGenerationResult {
					for _, rule := range []string{`root ::=`, `"\"category\""`, `"\"parcel\""`, `"\"letter\""`, `jinteger ws`, `jboolean ws`} {
						if !strings.Contains(grammar, rule) {
							t.Errorf("missing %q in grammar:\n%s", rule, grammar)
						}
					}
					if input.ToolCalling || len(input.Tools) != 0 || len(input.Messages) != 1 ||
						input.Messages[0].GetContent()["content"] != "Classify this letter." {
						t.Errorf("expected unchanged text prompt without tools, got %+v", input)
					}
					return tc.result
				},
			}
			plugin, g := initLoadingTest(t, op)
			def := testModelDefinition()
			def.ToolCalling = true
			model := plugin.DefineModel(g, def, nil)
			got, response, err := genkit.GenerateData[structuredResult](t.Context(), g,
				ai.WithModel(model), ai.WithPrompt("Classify this letter."))
			if tc.err != "" {
				if err == nil || !strings.Contains(err.Error(), tc.err) || got != nil {
					t.Fatalf("output = %+v, err = %v; want no output and error containing %q", got, err, tc.err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := structuredResult{Category: "parcel", Fragile: []bool{true, false}, Label: tc.label}
			want.Details.Pieces = 2
			if diff := cmp.Diff(&want, got); diff != "" {
				t.Errorf("output (-want +got):\n%s", diff)
			}
			if response.Request.Output == nil || !response.Request.Output.Constrained || response.FinishReason != ai.FinishReasonStop {
				t.Errorf("response = %+v; want a completed native constrained request", response)
			}
		})
	}
}

func TestGenerateDataRejectsUnsupportedRequests(t *testing.T) {
	op := &constrainedTestOperator{testTextOperator: &testTextOperator{
		testModelOperator: &testModelOperator{
			load: func(context.Context, ModelOptions) (llama.Model, error) {
				t.Error("unsupported request reached model loading")
				return 0, nil
			},
		},
	}}
	plugin, g := initLoadingTest(t, op)
	def := testModelDefinition()
	def.ToolCalling = true
	model := plugin.DefineModel(g, def, nil)
	tool := genkit.DefineTool(g, "unused", "An unused test tool.", func(_ *ai.ToolContext, _ struct{}) (string, error) {
		t.Error("unsupported request executed a tool")
		return "", nil
	})
	for _, tc := range []struct {
		name string
		opts []ai.GenerateOption
		err  string
	}{
		{"tools", []ai.GenerateOption{ai.WithTools(tool)}, "cannot be combined with tools"},
		{"tools with none", []ai.GenerateOption{ai.WithTools(tool), ai.WithToolChoice(ai.ToolChoiceNone)}, "cannot be combined with tools"},
		{"custom constraints", []ai.GenerateOption{ai.WithCustomConstrainedOutput()}, "requires native constraints"},
		{"stops", []ai.GenerateOption{ai.WithConfig(&GenerateContentConfig{StopSequences: ptr([]string{"}"})})}, "does not support stop sequences"},
		{"streaming", []ai.GenerateOption{ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error {
			t.Error("unsupported request streamed output")
			return nil
		})}, "does not support streaming"},
		{"tool history", []ai.GenerateOption{ai.WithMessages(&ai.Message{Role: ai.RoleTool, Content: []*ai.Part{ai.NewTextPart("tool output")}})}, `unsupported role "tool"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := append([]ai.GenerateOption{ai.WithModel(model), ai.WithPrompt("Classify mail.")}, tc.opts...)
			_, _, err := genkit.GenerateData[structuredResult](t.Context(), g, opts...)
			if err == nil || !strings.Contains(err.Error(), tc.err) {
				t.Fatalf("error = %v, want %q", err, tc.err)
			}
		})
	}

	type patternedResult struct {
		Label string `json:"label" jsonschema:"pattern=^ABC$"`
	}
	_, _, err := genkit.GenerateData[patternedResult](t.Context(), g, ai.WithModel(model), ai.WithPrompt("Say ABC."))
	if err == nil || !strings.Contains(err.Error(), `keyword "pattern" is not supported`) {
		t.Fatalf("error = %v, want unsupported pattern", err)
	}
}

func TestConstraintRejectsUnsupportedSchemaShapes(t *testing.T) {
	for _, tc := range []struct {
		name, err string
		schema    map[string]any
	}{
		{"array root", "root must be a closed object", map[string]any{"type": "array", "items": map[string]any{"type": "string"}}},
		{"open object", "additionalProperties: false", map[string]any{"type": "object", "properties": map[string]any{}}},
		{"undefined required key", "unique declared", map[string]any{
			"type": "object", "additionalProperties": false, "properties": map[string]any{}, "required": []string{"typo"},
		}},
		{"external reference", `keyword "$ref"`, map[string]any{
			"type": "object", "additionalProperties": false, "properties": map[string]any{"link": map[string]any{"$ref": "https://example.invalid/schema"}},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := compileConstraint(tc.schema); err == nil || !strings.Contains(err.Error(), tc.err) {
				t.Fatalf("error = %v, want %q", err, tc.err)
			}
		})
	}
}

// TestNativeStructuredOutput uses an existing GGUF and llama.cpp installation.
// The contradictory prompt makes unconstrained success unlikely, while the
// schema itself makes the expected result independent of model accuracy.
func TestNativeStructuredOutput(t *testing.T) {
	path, lib := os.Getenv("YZMA_TEST_MODEL"), os.Getenv("YZMA_LIB")
	if path == "" || lib == "" {
		t.Skip("set YZMA_TEST_MODEL and YZMA_LIB to exercise native constrained output")
	}
	plugin := &Yzma{LibPath: lib}
	g := genkit.Init(t.Context(), genkit.WithPlugins(plugin))
	t.Cleanup(func() {
		if err := plugin.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	def := testModelDefinition()
	def.Type = ModelTypeCompletion
	def.Loading.GGUFPath, def.Loading.NGPULayers = path, ptr(int32(0))
	def.Context.ContextSize = ptr(uint32(512))
	def.Context.Threads, def.Context.BatchThreads = ptr(int32(4)), ptr(int32(4))
	model := plugin.DefineModel(g, def, nil)
	type status struct {
		Alt     string `json:"alt,omitempty" jsonschema:"enum=X"`
		Code    string `json:"code" jsonschema:"enum=YZMA"`
		Details struct {
			OK bool `json:"ok"`
		} `json:"details"`
	}
	for _, temperature := range []float64{0, 0.7} {
		got, resp, err := genkit.GenerateData[status](t.Context(), g,
			ai.WithModel(model), ai.WithPrompt("The output is a single line of plain text: goodbye."),
			ai.WithConfig(&GenerateContentConfig{
				Temperature: ptr(temperature), MaxOutputTokens: ptr(64), Seed: ptr(uint32(7)),
			}),
		)
		if err != nil {
			t.Fatalf("temperature %v: %v", temperature, err)
		}
		if got == nil || got.Code != "YZMA" || resp.FinishReason != ai.FinishReasonStop {
			t.Errorf("temperature %v: output = %+v, response = %+v; want code YZMA and stop", temperature, got, resp)
		}
	}
	_, _, err := genkit.GenerateData[status](t.Context(), g,
		ai.WithModel(model), ai.WithPrompt("Say goodbye."),
		ai.WithConfig(&GenerateContentConfig{Temperature: ptr(0.0), MaxOutputTokens: ptr(1)}),
	)
	if err == nil || !strings.Contains(err.Error(), "incomplete constrained output") {
		t.Fatalf("error = %v, want incomplete constrained output at the token limit", err)
	}
}
