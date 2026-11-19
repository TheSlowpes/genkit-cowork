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
	"os"
	"path/filepath"
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
	args, err := untypedToolArguments(map[string]string{
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
	if err := validateToolArguments(map[string]any{"a": json.Number("2")}, schema); err != nil {
		t.Errorf("original referenced schema failed validation: %v", err)
	}
}

func TestJSONToolCallsPreserveWireTypes(t *testing.T) {
	def := &ai.ToolDefinition{Name: "echo", InputSchema: map[string]any{
		"type": "object", "required": []string{"value"}, "additionalProperties": false,
		"properties": map[string]any{"value": map[string]any{"type": "string"}},
	}}
	for name, wrap := range map[string]func(string) string{
		"standard": func(args string) string {
			return `<tool_call>{"name":"echo","arguments":` + args + `}</tool_call>`
		},
		"phi": func(args string) string {
			return `<|tool_call>{"name":"echo","arguments":` + args + `}</tool_call>`
		},
		"inline": func(args string) string { return `{"name":"echo","args":` + args + `}` },
		"lfm": func(args string) string {
			return `<|tool_call_start|>[{"name":"echo","arguments":` + args + `}]<|tool_call_end|>`
		},
		"mistral": func(args string) string { return `[TOOL_CALLS]echo[ARGS]` + args },
		"gpt":     func(args string) string { return `.echo <|message|>` + args },
	} {
		t.Run(name, func(t *testing.T) {
			parse := func(args string) ([]*ai.Part, error) {
				return parsedToolParts(TextGenerationResult{Text: wrap(args)}, []*ai.ToolDefinition{def}, ai.ToolChoiceAuto)
			}
			for _, args := range []string{`{"value":42}`, `{"value":false}`, `{"value":null}`,
				`{"arguments":{"value":"42"}}`, `{"value":"one","value":"two"}`, `"{}"`, `null`} {
				if parts, err := parse(args); err == nil || parts != nil {
					t.Errorf("parse(%s) = (%v, %v), want no parts and error", args, parts, err)
				}
			}
			parts, err := parse(`{"value":"</tool_call>42"}`)
			if err != nil || len(parts) != 1 || parts[0].ToolRequest == nil ||
				parts[0].ToolRequest.Input.(map[string]any)["value"] != "</tool_call>42" {
				t.Fatalf("parse valid string = (%+v, %v)", parts, err)
			}
		})
	}
}

func TestJSONToolCallsKeepStructureAndRejectBadBatches(t *testing.T) {
	def := &ai.ToolDefinition{Name: "echo", InputSchema: map[string]any{"type": "object"}}
	const good = `<tool_call>{"name":"echo","arguments":{"arguments":{"n":9007199254740993},"list":[null,"42",false]}}</tool_call>`
	parts, err := parsedToolParts(TextGenerationResult{Text: good}, []*ai.ToolDefinition{def}, ai.ToolChoiceAuto)
	if err != nil || len(parts) != 1 || parts[0].ToolRequest == nil {
		t.Fatalf("parse = (%v, %v)", parts, err)
	}
	args, err := json.Marshal(parts[0].ToolRequest.Input)
	if err != nil || string(args) != `{"arguments":{"n":9007199254740993},"list":[null,"42",false]}` {
		t.Fatalf("arguments = %s, err = %v", args, err)
	}
	for _, raw := range []string{
		good + `<tool_call>{"name":"echo","arguments":{}}`,
		good + `<tool_call>{"name":"echo","arguments":{},"args":{}}</tool_call>`,
		good + `<tool_call>{"name":"echo","arguments":{"n":tru}}</tool_call>`,
		good + `<|tool_call>call:echo{}<tool_call|>`,
		`<tool_call>{"name":"echo","arguments":{"nested":{"n":1,"n":2}}}</tool_call>`,
		`<tool_call>{"name":"add","arguments":{"a":"2","b":3}}</tool_call>`,
	} {
		if parts, err := parsedToolParts(TextGenerationResult{Text: raw}, []*ai.ToolDefinition{def, testAddDefinition()}, ai.ToolChoiceAuto); err == nil || parts != nil {
			t.Errorf("parse(%q) = (%v, %v), want no parts and error", raw, parts, err)
		}
	}
}

func TestInlineNamedJSONWithoutArgumentsRemainsText(t *testing.T) {
	const raw = `{"name":"Ada","age":37}`
	parts, err := parsedToolParts(TextGenerationResult{Text: raw}, nil, ai.ToolChoiceAuto)
	if err != nil || len(parts) != 1 || !parts[0].IsText() || parts[0].Text != raw {
		t.Fatalf("parse = (%v, %v), want the original text", parts, err)
	}
}

// TestNativeToolCalling uses Gemma 4 to request a real Genkit tool, then reads
// back a value that is present only in the tool result.
func TestNativeToolCalling(t *testing.T) {
	path, lib := os.Getenv("YZMA_TEST_TOOL_MODEL"), os.Getenv("YZMA_LIB")
	if path == "" || lib == "" {
		t.Skip("set YZMA_TEST_TOOL_MODEL to the Gemma 4 GGUF and set YZMA_LIB")
	}
	plugin := &Yzma{LibPath: lib, ModelsDir: filepath.Dir(path)}
	g := genkit.Init(t.Context(), genkit.WithPlugins(plugin))
	t.Cleanup(func() {
		if err := plugin.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	def := testModelDefinition()
	def.Loading.GGUFPath, def.Loading.NGPULayers = path, ptr(int32(0))
	def.Context = ContextOptions{ContextSize: ptr(uint32(2048)), Threads: ptr(int32(2)), BatchThreads: ptr(int32(2))}
	def.GenerationDefaults = &GenerateContentConfig{Temperature: ptr(0.0), MaxOutputTokens: ptr(256)}
	def.ToolCalling = true
	model := plugin.DefineModel(g, def, nil)
	type lookupInput struct {
		City string `json:"city"`
	}
	type lookupOutput struct {
		Code string `json:"code"`
	}
	const code = "violet-742"
	executions := 0
	lookup := genkit.DefineTool(g, "lookup_code", "Look up the secret code for a city",
		func(ctx *ai.ToolContext, input lookupInput) (lookupOutput, error) {
			executions++
			if input.City != "Oslo" {
				t.Errorf("tool input = %+v, want Oslo", input)
			}
			return lookupOutput{Code: code}, nil
		})
	result, err := genkit.Generate(t.Context(), g, ai.WithModel(model), ai.WithTools(lookup),
		ai.WithPrompt("Call lookup_code with city Oslo, then reply with the code returned by the tool. Do not guess."))
	if err != nil {
		t.Fatal(err)
	}
	var requests []*ai.ToolRequest
	var responses []*ai.ToolResponse
	for _, msg := range result.History() {
		for _, part := range msg.Content {
			if part.IsToolRequest() {
				requests = append(requests, part.ToolRequest)
			}
			if part.IsToolResponse() {
				responses = append(responses, part.ToolResponse)
			}
		}
	}
	if executions != 1 || len(requests) != 1 || len(responses) != 1 {
		t.Fatalf("executions = %d, calls = %d, tool replies = %d; final text = %q",
			executions, len(requests), len(responses), result.Text())
	}
	call, reply := requests[0], responses[0]
	output, err := json.Marshal(reply.Output)
	if call.Ref == "" || call.Ref != reply.Ref || call.Name != "lookup_code" || reply.Name != call.Name ||
		err != nil || string(output) != `{"code":"`+code+`"}` || !strings.Contains(result.Text(), code) {
		t.Fatalf("call = %+v, reply = %+v, final text = %q", call, reply, result.Text())
	}
	t.Logf("matched lookup_code call and result %s; final text: %q", call.Ref, result.Text())
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
