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
	"encoding/json"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/hybridgroup/yzma/pkg/llama"
)

func testAddDefinition() *ai.ToolDefinition {
	return &ai.ToolDefinition{Name: "add", InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{
			"a": map[string]any{"type": "integer"},
			"b": map[string]any{"type": "integer"},
		}, "required": []string{"a", "b"}, "additionalProperties": false,
	}}
}

func TestParseToolFamilies(t *testing.T) {
	for name, raw := range map[string]string{
		"Gemma": `<|tool_call>call:add{a:2,b:3}<tool_call|>`,
		"JSON":  `<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>`,
		"Qwen":  `<tool_call><function=add><parameter=a>2</parameter><parameter=b>3</parameter></function></tool_call>`,
	} {
		t.Run(name, func(t *testing.T) {
			parts, err := parsedToolParts(TextGenerationResult{Text: raw}, []*ai.ToolDefinition{testAddDefinition()}, ai.ToolChoiceAuto)
			if err != nil {
				t.Fatal(err)
			}
			var requests []*ai.ToolRequest
			for _, part := range parts {
				if part.IsToolRequest() {
					requests = append(requests, part.ToolRequest)
				}
			}
			if len(requests) != 1 {
				t.Fatalf("tool requests = %v, want one", requests)
			}
			raw, err := json.Marshal(requests[0].Input)
			if err != nil {
				t.Fatal(err)
			}
			if requests[0].Name != "add" || requests[0].Ref == "" || string(raw) != `{"a":2,"b":3}` {
				t.Errorf("tool request = %+v, arguments=%s", requests[0], raw)
			}
		})
	}
}

func TestToolCallsRejectUnsafeOutputs(t *testing.T) {
	for name, result := range map[string]TextGenerationResult{
		"truncated": {Text: `<tool_call>{"name":"add","arguments":{"a":2,"b":3`},
		"limited":   {Text: `<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>`, Limited: true},
		"unknown":   {Text: `<tool_call>{"name":"other","arguments":{}}</tool_call>`},
		"missing":   {Text: `<tool_call>{"name":"add","arguments":{"a":2}}</tool_call>`},
		"bad type":  {Text: `<tool_call>{"name":"add","arguments":{"a":"oops","b":3}}</tool_call>`},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := parsedToolParts(result, []*ai.ToolDefinition{testAddDefinition()}, ai.ToolChoiceAuto); err == nil {
				t.Fatal("invalid tool output must not produce an executable request")
			}
		})
	}
}

func TestToolArgumentTypes(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{
		"string": map[string]any{"type": "string"}, "bool": map[string]any{"type": "boolean"},
		"array": map[string]any{"type": "array"}, "object": map[string]any{"type": "object"},
	}}
	args, err := typedToolArguments(map[string]string{
		"string": "0123", "bool": "false", "array": `[1,true,"x"]`, "object": `{"nested":42}`,
	}, schema)
	if err != nil {
		t.Fatal(err)
	}
	if args["string"] != "0123" || args["bool"] != false {
		t.Errorf("schema types lost: %+v", args)
	}
}

func TestToolCallingRoundTrip(t *testing.T) {
	var generations, executions int
	op := &testTextOperator{
		testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
			return 42, nil
		}},
		generate: func(ctx context.Context, model llama.Model, input TextGenerationInput, onText func(context.Context, string) error) (TextGenerationResult, error) {
			generations++
			if onText != nil {
				t.Error("tool syntax must be buffered until parsing, not streamed as text")
			}
			if len(input.Tools) != 1 || !input.ToolCalling {
				t.Errorf("tool definitions missing: %+v", input.Tools)
			}
			if generations == 1 {
				return TextGenerationResult{Text: `<|tool_call>call:add{a:2,b:3}<tool_call|>`}, nil
			}
			history := input.Messages
			if len(history) != 3 || history[1].GetRole() != "assistant" || history[2].GetRole() != "tool" {
				t.Fatalf("tool history missing: %+v", history)
			}
			call := history[1].GetContent()["tool_calls"].([]map[string]any)[0]
			result := history[2].GetContent()
			if call["id"] != result["tool_call_id"] || result["name"] != "add" {
				t.Error("tool call references were not preserved")
			}
			return TextGenerationResult{Text: "The answer is 5."}, nil
		},
	}
	plugin, g := initLoadingTest(t, op)
	def := testModelDefinition()
	def.ToolCalling = true
	plugin.DefineModel(g, def, nil)
	type addInput struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	add := genkit.DefineTool(g, "add", "Add integers", func(ctx *ai.ToolContext, in addInput) (int, error) {
		executions++
		if in.A != 2 || in.B != 3 {
			t.Errorf("tool input = %+v", in)
		}
		return in.A + in.B, nil
	})
	var text strings.Builder
	for value, err := range genkit.GenerateStream(t.Context(), g,
		ai.WithModelName("yzma/local-chat"), ai.WithPrompt("Use add to calculate 2+3"),
		ai.WithTools(add),
	) {
		if err != nil {
			t.Fatal(err)
		}
		if !value.Done {
			for _, part := range value.Chunk.Content {
				if part.IsText() {
					text.WriteString(part.Text)
				}
			}
		} else if value.Response.Text() != "The answer is 5." {
			t.Errorf("final text = %q", value.Response.Text())
		}
	}
	if executions != 1 || generations != 2 || strings.Contains(text.String(), "call:add") {
		t.Errorf("executions=%d generations=%d streamed=%q", executions, generations, text.String())
	}
}

func TestEmptyGemmaToolCall(t *testing.T) {
	parts, err := parsedToolParts(TextGenerationResult{Text: `<|tool_call>call:ping{}<tool_call|>`},
		[]*ai.ToolDefinition{{Name: "ping", InputSchema: map[string]any{"type": "object"}}}, ai.ToolChoiceAuto)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 1 || !parts[0].IsToolRequest() || parts[0].ToolRequest.Name != "ping" {
		t.Errorf("empty-argument tool call = %+v", parts)
	}
	parts, err = parsedToolParts(TextGenerationResult{
		Text: `<|tool_call>call:ping{}<tool_call|><|tool_call>call:add{a:2,b:3}<tool_call|>`,
	}, []*ai.ToolDefinition{
		{Name: "ping", InputSchema: map[string]any{"type": "object"}}, testAddDefinition(),
	}, ai.ToolChoiceAuto)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 || !parts[0].IsToolRequest() || !parts[1].IsToolRequest() {
		t.Errorf("parallel empty/non-empty calls = %+v, want both calls", parts)
	}
}

func TestToolDefinitionsResolveReferences(t *testing.T) {
	schema := map[string]any{
		"$ref": "#/$defs/Input",
		"$defs": map[string]any{
			"Input": map[string]any{"type": "object", "properties": map[string]any{
				"a": map[string]any{"$ref": "#/$defs/Integer"},
			}, "required": []string{"a"}},
			"Integer": map[string]any{"type": "integer"},
		},
	}
	tools, err := convertTools([]*ai.ToolDefinition{{Name: "test", InputSchema: schema}})
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 || tools[0].Function.Name != "test" {
		t.Fatalf("converted definitions = %+v", tools)
	}
	properties := tools[0].Function.Parameters["properties"].(map[string]any)
	if properties["a"].(map[string]any)["type"] != "integer" || schema["$ref"] != "#/$defs/Input" {
		t.Error("reference expansion failed or mutated the original schema")
	}
	if _, err := typedToolArguments(map[string]string{"a": "2"}, schema); err != nil {
		t.Errorf("original referenced schema failed validation: %v", err)
	}
}

func TestToolHistoryRejectsMismatchedReferences(t *testing.T) {
	req := ai.NewModelMessage(ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "add", Ref: "call-1", Input: map[string]any{"a": 2, "b": 3},
	}))
	input := ai.NewModelRequest(nil, ai.NewUserTextMessage("Add"), req,
		ai.NewMessage(ai.RoleTool, nil, ai.NewToolResponsePart(&ai.ToolResponse{
			Name: "add", Ref: "wrong-id", Output: 5,
		})))
	if _, err := generationMessages(input, true); err == nil {
		t.Error("mismatched tool response must not be fed to the model")
	}
}

func TestCompleteToolOutput(t *testing.T) {
	for raw, want := range map[string]bool{
		`<tool_call>{"name":"echo","arguments":{"text":"{braces}"}}</tool_call>`: true,
		`<|tool_call>call:echo{text:<|"|>a } brace<|"|>}<tool_call|>`:            true,
		`<tool_call>{"name":"echo","arguments":{"text":"unfinished`:              false,
		`call:echo{text:[1,2}`: false,
	} {
		if got := completeToolOutput(raw); got != want {
			t.Errorf("completeToolOutput(%q) = %v, want %v", raw, got, want)
		}
	}
}

func TestRequiredAndDisabledToolChoices(t *testing.T) {
	if _, err := parsedToolParts(TextGenerationResult{Text: "No tool needed"},
		[]*ai.ToolDefinition{testAddDefinition()}, ai.ToolChoiceRequired); err == nil {
		t.Error("required tool choice must not silently return plain text")
	}
	if _, err := parsedToolParts(TextGenerationResult{Text: `call:add{a:2,b:3}`},
		[]*ai.ToolDefinition{testAddDefinition()}, ai.ToolChoiceNone); err == nil {
		t.Error("disabled tools must not produce executable calls")
	}
}
