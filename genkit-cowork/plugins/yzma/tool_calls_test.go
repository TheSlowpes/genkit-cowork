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

func TestQwenToolCallsValidateWholeEnvelope(t *testing.T) {
	tool := &ai.ToolDefinition{Name: "run", InputSchema: map[string]any{
		"type": "object", "additionalProperties": false, "properties": map[string]any{
			"text": map[string]any{"type": "string"}, "dryRun": map[string]any{"type": "boolean"},
		},
	}}
	const wrapped = `<tool_call><function=run><parameter=text>}</parameter></function></tool_call>`
	const bare = `<function=run><parameter=dryRun>true</parameter></function>`
	parse := func(raw string, limited bool) ([]*ai.Part, error) {
		return parsedToolParts(TextGenerationResult{Text: raw, Limited: limited}, []*ai.ToolDefinition{tool}, ai.ToolChoiceAuto)
	}
	parts, err := parse("Working on it.\n"+wrapped+"\n"+bare, false)
	if err != nil || len(parts) != 3 || parts[0].Text != "Working on it." ||
		parts[1].ToolRequest == nil || parts[2].ToolRequest == nil {
		t.Fatalf("Qwen calls = (%v, %v), want text and two requests", parts, err)
	}
	if parts[1].ToolRequest.Input.(map[string]any)["text"] != "}" ||
		parts[2].ToolRequest.Input.(map[string]any)["dryRun"] != true {
		t.Errorf("Qwen arguments = %v and %v", parts[1].ToolRequest.Input, parts[2].ToolRequest.Input)
	}
	for name, raw := range map[string]string{
		"unclosed optional parameter": `<function=run><parameter=dryRun>true</function>`,
		"unclosed after valid call":   wrapped + `<function=run><parameter=dryRun>true</function>`,
		"partial trailing function":   wrapped + `<function`,
		"next tag must not close it":  `<function=run><parameter=text>hi<parameter=dryRun>true</parameter></function>`,
		"unclosed wrapper":            `<tool_call>` + bare,
		"misplaced parameter":         wrapped + `<parameter=dryRun>true</parameter>`,
		"unrecognized body text":      `<function=run>dryRun=true</function>`,
		"duplicate parameter":         `<function=run><parameter=text>a</parameter><parameter=text>b</parameter></function>`,
	} {
		t.Run(name, func(t *testing.T) {
			if parts, err := parse(raw, false); err == nil || parts != nil {
				t.Errorf("parse = (%v, %v), want no requests and an error", parts, err)
			}
		})
	}
	if parts, err := parse(wrapped, true); err == nil || parts != nil {
		t.Errorf("limited Qwen output = (%v, %v), want no requests and an error", parts, err)
	}
	if parts, err := parse("Partial answer.", true); err != nil || len(parts) != 1 || parts[0].Text != "Partial answer." {
		t.Errorf("limited plain text = (%v, %v), want text without a tool error", parts, err)
	}
	// Qwen-looking text inside another format's arguments isn't an envelope.
	const gemma = `<|tool_call>call:run{text:<|"|><parameter=dryRun><|"|>}<tool_call|>`
	parts, err = parse(gemma, false)
	if err != nil || len(parts) != 1 || parts[0].ToolRequest == nil ||
		parts[0].ToolRequest.Input.(map[string]any)["text"] != "<parameter=dryRun>" {
		t.Errorf("Gemma with literal tag text = (%v, %v)", parts, err)
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
		"harmony": func(args string) string {
			return `<|channel|>commentary to=functions.echo <|constrain|>json<|message|>` + args + `<|call|>`
		},
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

func TestHarmonyToolCallFraming(t *testing.T) {
	const header = `<|channel|>commentary to=functions.add <|constrain|>json<|message|>`
	const args = `{"a":2,"b":3}`
	const good = header + args + `<|call|>`
	const preamble = `<|channel|>analysis<|message|>Use the add tool.<|end|>` +
		`<|start|>assistant<|channel|>commentary<|message|>Calculating.<|end|><|start|>assistant`
	const role = `<|start|>assistant to=functions.add<|channel|>commentary<|message|>` + args + `<|call|>`
	parse := func(raw string, limited bool) ([]*ai.Part, error) {
		return parsedToolParts(TextGenerationResult{Text: raw, Limited: limited},
			[]*ai.ToolDefinition{testAddDefinition()}, ai.ToolChoiceAuto)
	}
	parts, err := parse(preamble+good+role, false)
	if err != nil || len(parts) != 3 || parts[0].Text != "Calculating." {
		t.Fatalf("parse = (%v, %v), want the preamble and two requests", parts, err)
	}
	for _, part := range parts[1:] {
		if part.ToolRequest == nil || part.ToolRequest.Name != "add" {
			t.Fatalf("part = %v, want add request", part)
		}
		got := part.ToolRequest.Input.(map[string]any)
		if got["a"] != json.Number("2") || got["b"] != json.Number("3") {
			t.Errorf("arguments = %#v, want the original JSON numbers", got)
		}
	}
	const reasoning = `<|channel|>analysis<|message|><function=add><parameter=a>2</parameter>` +
		`<parameter=b>3</parameter></function><|end|>`
	parts, err = parse(reasoning+`<|start|>assistant<|channel|>final<|message|>Done.<|return|>`, false)
	if err != nil || len(parts) != 1 || !parts[0].IsText() || parts[0].Text != "Done." {
		t.Fatalf("parse = (%v, %v), want only final text", parts, err)
	}
	for name, result := range map[string]TextGenerationResult{
		"no terminator":     {Text: header + args},
		"wrong terminator":  {Text: header + args + `<|end|>`},
		"extra body":        {Text: header + args + `{}` + `<|call|>`},
		"limited":           {Text: good, Limited: true},
		"broken batch":      {Text: good + header + `{"a":2,"b":`},
		"invalid schema":    {Text: good + header + `{"a":"2","b":3}<|call|>`},
		"unknown recipient": {Text: good + strings.Replace(header, "functions.", "python.", 1) + args + `<|call|>`},
	} {
		t.Run(name, func(t *testing.T) {
			if parts, err := parse(result.Text, result.Limited); err == nil || parts != nil {
				t.Errorf("parse = (%v, %v), want no parts and an error", parts, err)
			}
		})
	}
}

func TestJSONToolCallsKeepStructureAndRejectBadBatches(t *testing.T) {
	def := &ai.ToolDefinition{Name: "echo", InputSchema: map[string]any{"type": "object"}}
	const large = `<tool_call>{"name":"echo","arguments":{"arguments":{"n":9007199254740993},"list":[null,"42",false],"parameters":{"x":1}}}</tool_call>`
	calls, _, err := parseWireCalls(large)
	if err != nil || len(calls) != 1 {
		t.Fatalf("parse = (%v, %v)", calls, err)
	}
	args, err := json.Marshal(calls[0].args)
	if err != nil || string(args) != `{"arguments":{"n":9007199254740993},"list":[null,"42",false],"parameters":{"x":1}}` {
		t.Fatalf("arguments = %s, err = %v", args, err)
	}
	good := strings.Replace(large, "9007199254740993", "42", 1)
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

func TestGenkitToolDispatchPreservesNestedArguments(t *testing.T) {
	type payload struct {
		ID int64 `json:"id"`
	}
	type input struct {
		Arguments  []payload `json:"arguments"`
		Parameters payload   `json:"parameters"`
	}
	const exact = int64(9007199254740992)
	for _, tc := range []struct {
		name, number string
		wantError    bool
	}{
		{name: "exact", number: "9007199254740992"},
		{name: "rounded integer", number: "9007199254740993", wantError: true},
		{name: "rounded integer hidden by short float spelling", number: "1000000000000000100", wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			op := &testTextOperator{
				testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
					return 42, nil
				}},
				generate: func(_ context.Context, _ llama.Model, input TextGenerationInput, _ func(context.Context, string) error) (TextGenerationResult, error) {
					if input.Messages[len(input.Messages)-1].GetRole() == "tool" {
						return TextGenerationResult{Text: "Done."}, nil
					}
					return TextGenerationResult{
						Text: `<tool_call>{"name":"echo","arguments":{"arguments":[{"id":` + tc.number +
							`}],"parameters":{"id":42}}}</tool_call>`,
					}, nil
				},
			}
			plugin, g := initLoadingTest(t, op)
			def := testModelDefinition()
			def.ToolCalling = true
			model := plugin.DefineModel(g, def, nil)
			executed := false
			tool := genkit.DefineTool(g, "echo", "Echo the nested ID", func(_ *ai.ToolContext, in input) (string, error) {
				executed = true
				if len(in.Arguments) != 1 || in.Arguments[0].ID != exact || in.Parameters.ID != 42 {
					t.Errorf("tool got %+v; want arguments ID %d and parameters ID 42", in, exact)
				}
				return "ok", nil
			})
			resp, err := genkit.Generate(t.Context(), g, ai.WithModel(model), ai.WithTools(tool), ai.WithPrompt("Use echo."))
			if tc.wantError {
				if executed || err == nil || !strings.Contains(err.Error(), "lose precision") {
					t.Fatalf("executed = %v, error = %v; want a precision error before execution", executed, err)
				}
			} else if err != nil || !executed || resp.Text() != "Done." {
				t.Fatalf("response = %v, error = %v, executed = %v; want completed tool chain", resp, err, executed)
			}
		})
	}
}

func TestGenkitToolDispatchRejectsNullObjectMembers(t *testing.T) {
	for _, tc := range []struct {
		name, args, wantInput string
	}{
		{name: "top-level null", args: `{"optional":null}`},
		{name: "nested null", args: `{"nested":{"optional":null}}`},
		{name: "object in array with null", args: `{"list":[{"optional":null}]}`},
		{name: "absent optional field", args: `{}`, wantInput: `{}`},
		{name: "null array element", args: `{"list":[null,{}]}`, wantInput: `{"list":[null,{}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			op := &testTextOperator{
				testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
					return 42, nil
				}},
				generate: func(_ context.Context, _ llama.Model, input TextGenerationInput, _ func(context.Context, string) error) (TextGenerationResult, error) {
					if input.Messages[len(input.Messages)-1].GetRole() == "tool" {
						return TextGenerationResult{Text: "Done."}, nil
					}
					return TextGenerationResult{Text: `{"name":"echo","arguments":` + tc.args + `}`}, nil
				},
			}
			plugin, g := initLoadingTest(t, op)
			def := testModelDefinition()
			def.ToolCalling = true
			model := plugin.DefineModel(g, def, nil)
			object := map[string]any{"type": "object", "properties": map[string]any{
				"optional": map[string]any{"type": "null"},
			}}
			schema := map[string]any{"type": "object", "properties": map[string]any{
				"optional": map[string]any{"type": "null"},
				"nested":   object,
				"list": map[string]any{"type": "array", "items": map[string]any{
					"anyOf": []any{map[string]any{"type": "null"}, object},
				}},
			}}
			executions := 0
			tool := genkit.DefineTool(g, "echo", "Echo the input", func(_ *ai.ToolContext, in any) (string, error) {
				executions++
				body, err := json.Marshal(in)
				if err != nil || string(body) != tc.wantInput {
					t.Errorf("tool input = (%s, %v), want %s", body, err, tc.wantInput)
				}
				return "ok", nil
			}, ai.WithInputSchema(schema))
			resp, err := genkit.Generate(t.Context(), g, ai.WithModel(model), ai.WithTools(tool), ai.WithPrompt("Use echo."))
			if tc.wantInput == "" {
				if executions != 0 || err == nil || !strings.Contains(err.Error(), "null object member") {
					t.Fatalf("executions = %d, error = %v; want a null transport error before execution", executions, err)
				}
			} else if err != nil || executions != 1 || resp.Text() != "Done." {
				t.Fatalf("response = %v, error = %v, executions = %d; want completed tool chain", resp, err, executions)
			}
		})
	}
}

func TestGenkitGLMToolDispatch(t *testing.T) {
	type input struct {
		Text  string `json:"text"`
		Extra string `json:"extra,omitempty"`
	}
	const text = `{"name":"Ada"}`
	const extra = `{"name": "Grace", "labels": [" A ", "B"]}`
	const key = `echo<arg_key>text</arg_key>`
	const value = `<arg_value>` + text + `</arg_value>`
	for _, tc := range []struct {
		name, raw string
		want      input
	}{
		{name: "space", raw: key + " " + value, want: input{Text: text}},
		{
			name: "tab and multiple arguments",
			raw:  key + "\t" + value + "<arg_key>extra</arg_key> \t<arg_value>" + extra + "</arg_value>",
			want: input{Text: text, Extra: extra},
		},
		{name: "non-whitespace gap", raw: key + " stray " + value},
		{name: "line break", raw: key + "\n" + value},
		{name: "unclosed value", raw: key + " <arg_value>" + text},
		{
			name: "tool envelope inside the argument",
			raw:  key + ` <arg_value>{"name":"echo","args":{"text":"not the GLM argument"}}</arg_value>`,
			want: input{Text: `{"name":"echo","args":{"text":"not the GLM argument"}}`},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			op := &testTextOperator{
				testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
					return 42, nil
				}},
				generate: func(_ context.Context, _ llama.Model, in TextGenerationInput, _ func(context.Context, string) error) (TextGenerationResult, error) {
					if in.Messages[len(in.Messages)-1].GetRole() == "tool" {
						return TextGenerationResult{Text: "Done."}, nil
					}
					return TextGenerationResult{Text: tc.raw}, nil
				},
			}
			plugin, g := initLoadingTest(t, op)
			def := testModelDefinition()
			def.ToolCalling = true
			model := plugin.DefineModel(g, def, nil)
			var got input
			calls := 0
			tool := genkit.DefineTool(g, "echo", "Echo the input", func(_ *ai.ToolContext, in input) (string, error) {
				calls++
				got = in
				return "ok", nil
			})
			resp, err := genkit.Generate(t.Context(), g, ai.WithModel(model), ai.WithTools(tool), ai.WithPrompt("Use echo."))
			if tc.want == (input{}) {
				if err == nil || calls != 0 {
					t.Fatalf("calls = %d, input = %+q, error = %v; want an error before dispatch", calls, got, err)
				}
			} else if err != nil || calls != 1 || got != tc.want || resp.Text() != "Done." {
				t.Fatalf("calls = %d, input = %+q, error = %v; want one call with %+q and a completed tool chain",
					calls, got, err, tc.want)
			}
		})
	}
}

func TestNamedJSONProseSurvivesToolMarkupCleaning(t *testing.T) {
	const prose = `{"name":"Ada","meta":{"args":{"x":1}}}`
	const nestedCall = `{"name":"Ada","meta":{"name":"add","args":{"a":2,"b":3}}}`
	for _, tc := range []struct {
		name, raw, want string
		wantCall        bool
	}{
		{name: "text only", raw: "Profile: " + prose + " \n", want: "Profile: " + prose},
		{name: "nested call shape is text", raw: nestedCall, want: nestedCall},
		{name: "reasoning stays hidden", raw: "<think>" + prose + "</think>" + prose, want: prose},
		{
			name:     "with JSON tool call",
			raw:      prose + ` <tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call> ` + prose + " \n",
			want:     prose + "  " + prose,
			wantCall: true,
		},
		{
			name:     "with Gemma fallback call",
			raw:      prose + ` <|tool_call>call:add{a:2,b:3}<tool_call|> ` + prose + " \n",
			want:     prose + "  " + prose,
			wantCall: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parts, err := parsedToolParts(TextGenerationResult{Text: tc.raw}, []*ai.ToolDefinition{testAddDefinition()}, ai.ToolChoiceAuto)
			if err != nil || len(parts) == 0 || !parts[0].IsText() || parts[0].Text != tc.want {
				t.Fatalf("parse = (%v, %v), want text %q", parts, err, tc.want)
			}
			if tc.wantCall {
				if len(parts) != 2 || !parts[1].IsToolRequest() || parts[1].ToolRequest.Name != "add" {
					t.Fatalf("parts = %v, want text and one add request", parts)
				}
			} else if len(parts) != 1 {
				t.Fatalf("parts = %v, want only text", parts)
			}
		})
	}
	malformed := prose + ` {"name":"add","args":{"a":2,"b":`
	if parts, err := parsedToolParts(TextGenerationResult{Text: malformed}, []*ai.ToolDefinition{testAddDefinition()}, ai.ToolChoiceAuto); err == nil || parts != nil {
		t.Errorf("parse = (%v, %v), want an error and no parts for a malformed call after prose", parts, err)
	}
}

func TestToolDelimitersInProse(t *testing.T) {
	const qwen = `<function=add><parameter=a>2</parameter><parameter=b>3</parameter></function>`
	const inline = `{"name":"add","args":{"a":2,"b":3}}`
	parse := func(raw string) ([]*ai.Part, error) {
		return parsedToolParts(TextGenerationResult{Text: raw}, []*ai.ToolDefinition{testAddDefinition()}, ai.ToolChoiceAuto)
	}
	for _, raw := range []string{
		"Use the <parameter> delimiter and close with </parameter>.",
		`{"name":"Ada","age":37}`,
		`For example, {"name": <value>}.`,
		`An example starts with {"name":`,
		`{"name":"args"`,
		`{"name":<value>,"note":"argument key \"args\":"}`,
	} {
		if parts, err := parse(raw); err != nil || len(parts) != 1 || !parts[0].IsText() || parts[0].Text != raw {
			t.Errorf("parse(%q) = (%v, %v), want the original text", raw, parts, err)
		}
	}
	for _, call := range []string{qwen, inline} {
		const prose = `The <parameter> delimiter goes before {"name": <value>}.`
		if parts, err := parse(prose + "\n" + call); err != nil || len(parts) != 2 ||
			parts[0].Text != prose || !parts[1].IsToolRequest() {
			t.Errorf("parse with preamble = (%v, %v), want text and a request", parts, err)
		}
	}
	for _, raw := range []string{
		`{"name":"add","arguments":`,
		`{"name":<value>,"args":{}}`,
		`{"name":"add","arguments`,
		`{"name":"add","arguments"}`,
		inline + `{"name":"add","args":{"a":2,"b":`,
		inline + `{"name":42,"arguments":{"a":2,"b":3}}`,
		inline + `<function=add><parameter=a>2</function>`,
		inline + qwen,
		qwen + `<parameter=a>2</parameter>`,
	} {
		if parts, err := parse(raw); err == nil || parts != nil {
			t.Errorf("parse(%q) = (%v, %v), want no parts and an error", raw, parts, err)
		}
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
