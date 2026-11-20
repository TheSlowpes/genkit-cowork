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
	"github.com/google/go-cmp/cmp"
	"github.com/hybridgroup/yzma/pkg/llama"
)

const (
	wireTestJSON    = `<tool_call>{"name":"echo","args":{"text":"ok"}}</tool_call>`
	wireTestGemma   = `<|tool_call>call:echo{text:<|"|>ok<|"|>}<tool_call|>`
	wireTestGLM     = `echo<arg_key>text</arg_key><arg_value>ok</arg_value>`
	wireTestQwen    = `<function=echo><parameter=text>ok</parameter></function>`
	wireTestMistral = `[TOOL_CALLS]echo[ARGS]{"text":"ok"}`
	wireTestPython  = `<|tool_call_start|>[echo(text='ok')]<|tool_call_end|>`
)

func TestToolWireRejectsWholeMalformedBatch(t *testing.T) {
	for _, tc := range []struct {
		name, good, bad string
	}{
		{"JSON trailing partial wrapper", wireTestJSON, `<tool_call`},
		{"JSON trailing partial args first", wireTestJSON, `{"args":{"text":`},
		{"JSON missing comma", wireTestJSON, `{"name":"echo" "args":{"text":"bad"}}`},
		{"JSON invalid UTF-16", wireTestJSON, `{"name":"echo","args":{"text":"\ud800"}}`},
		{"JSON unsupported wrapper alias", wireTestJSON, `<toolcall>{"name":"echo","args":{}}</toolcall>`},
		{"Phi missing close", wireTestJSON, `<|tool_call>{"name":"echo","args":{"text":"bad"}}`},
		{"Qwen partial trailing tag", wireTestQwen, `<functio`},
		{"Qwen stray leading parameter", wireTestQwen, `<parameter=text>bad</parameter>`},
		{"Gemma missing closer", wireTestGemma, `<|tool_call>call:echo{text:<|"|>bad<|"|>}`},
		{"Gemma missing argument value", wireTestGemma, `call:echo{text:<|"|>bad<|"|>,items}`},
		{"Gemma duplicate field", wireTestGemma, `call:echo{text:<|"|>bad<|"|>,"text":"worse"}`},
		{"Gemma duplicate nested field", wireTestGemma, `call:echo{object:{value:1,value:2}}`},
		{"Gemma missing comma", wireTestGemma, `call:echo{text:<|"|>bad<|"|>items:[]}`},
		{"Gemma missing closing object", wireTestGemma, `call:echo{text:[1,2}`},
		{"GLM duplicate key", wireTestGLM, wireTestGLM + `<arg_key>text</arg_key><arg_value>duplicate</arg_value>`},
		{"GLM missing optional value", wireTestGLM, wireTestGLM + `<arg_key>items</arg_key><arg_value>partial`},
		{"LFM2 Python missing wrapper close", wireTestPython, `<|tool_call_start|>[echo(text='bad')]`},
		{"LFM2 Python missing paren", wireTestPython, `<|tool_call_start|>[echo(text='bad']<|tool_call_end|>`},
		{"LFM2 Python invalid second item", wireTestPython, `<|tool_call_start|>[echo(text='bad'), ping]<|tool_call_end|>`},
		{"LFM2 Python duplicate argument", wireTestPython, `<|tool_call_start|>[echo(text='x',text='y')]<|tool_call_end|>`},
		{"LFM2 Python duplicate dictionary key", wireTestPython, `<|tool_call_start|>[echo(object={'value':1,"value":2})]<|tool_call_end|>`},
		{"LFM2 Python unknown escape", wireTestPython, `<|tool_call_start|>[echo(text='\q')]<|tool_call_end|>`},
		{"LFM2 Python unsupported expression", wireTestPython, `<|tool_call_start|>[echo(text=bad())]<|tool_call_end|>`},
		{"LFM2 JSON invalid list member", "", `<|tool_call_start|>[{"name":"echo","args":{"text":"ok"}},null]<|tool_call_end|>`},
		{"Mistral leading args marker", wireTestMistral, `[ARGS]`},
		{"Mistral unattached argument object", wireTestMistral, `[ARGS]{"text":"bad"}`},
		{"Mistral unsupported legacy array", wireTestMistral, `[TOOL_CALLS][{"name":"echo","arguments":{"text":"bad"}}]`},
		{"mixed Gemma and GLM", wireTestGemma, wireTestGLM},
		{"mixed Python and Gemma", wireTestPython, wireTestGemma},
		{"mixed Mistral and wrapped JSON", wireTestMistral, wireTestJSON},
		{"mixed LFM2 encodings", wireTestPython, `<|tool_call_start|>{"name":"echo","args":{}}<|tool_call_end|>`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A truncated marker is only recognizable at the end of the
			// stream. (Complete malformed frames are also covered by the
			// dispatch test with the malformed span first.)
			raw := tc.good + "\n" + tc.bad
			if tc.name == "Qwen stray leading parameter" || tc.name == "Mistral leading args marker" {
				raw = tc.bad + "\n" + tc.good
			}
			calls, text, err := parseWireCalls(raw)
			if calls != nil || text != "" || err == nil {
				t.Errorf("parse = (%+v, %q, %v), want an error and no output", calls, text, err)
			}
		})
	}
}

func TestToolWireVisibleTextUsesFrameSpans(t *testing.T) {
	const object = `{"payload":{"name":"echo","args":{"text":"nested"}},"example":"<tool_call>"}`
	const array = `[{"name":"echo","args":{"text":"nested"}}]`
	for _, tc := range []struct {
		name, raw, want string
		calls           int
	}{
		{"generic object and array", object + " " + array, object + " " + array, 0},
		{"ordinary args-only data", `{"args":[1,2]}`, `{"args":[1,2]}`, 0},
		{"unclosed prose does not expose nested inline JSON", `Related [2 notes. {"name":"echo","args":{"text":"nested"}}`, `Related [2 notes. {"name":"echo","args":{"text":"nested"}}`, 0},
		{"ordinary prose before and after", "status: 200 " + wireTestGemma + " After [a bracket", "status: 200  After [a bracket", 1},
		{"ordinary call prefix", "We call: this a demo. " + wireTestGemma, "We call: this a demo.", 1},
		{"ordinary call at end", "The next call: echo", "The next call: echo", 0},
		{"named JSON prose nonfirst", `{"note":"<tool_call>","name":"Ada"}` + wireTestQwen, `{"note":"<tool_call>","name":"Ada"}`, 1},
		{"bare Gemma empty and nonempty", `call:ping{} call:echo{text:"ok"}`, "", 2},
		{"wrapped and inline JSON", wireTestJSON + `{"args":{"text":"ok"},"na\u006de":"echo"}`, "", 2},
		{"LFM singleton JSON", `<|tool_call_start|>{"name":"echo","args":{"text":"ok"}}<|tool_call_end|>`, "", 1},
		{"LFM literal bracket prose", "Before " + wireTestPython + " After [bracket", "Before  After [bracket", 1},
		{"GLM suffix and literal brace", `echo<arg_key>text</arg_key><arg_value>{</arg_value> After`, "After", 1},
		{"GLM prose mention", "In GLM, write <arg_key> followed by a key.", "In GLM, write <arg_key> followed by a key.", 0},
		{"Gemma next turn", "Before<turn|><|turn>user" + wireTestGemma, "Before", 0},
		{"Phi next turn", "Before<|end|>" + wireTestJSON, "Before", 0},
		{"Harmony visible unchannelled", "One. <|message|>Two.", "One. Two.", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls, text, err := parseWireCalls(tc.raw)
			if err != nil || text != tc.want || len(calls) != tc.calls {
				t.Errorf("parse = (%+v, %q, %v), want %d calls and %q", calls, text, err, tc.calls, tc.want)
			}
		})
	}
}

func TestGenkitToolFrameDispatch(t *testing.T) {
	type object struct {
		Value string `json:"value"`
	}
	type input struct {
		Text   string   `json:"text"`
		Items  []string `json:"items,omitempty"`
		Object *object  `json:"object,omitempty"`
	}
	const done = "Done."
	const harmonyExample = `Example: <|start|>assistant<|channel|>commentary to=functions.echo<|message|>{"text":"<|return|>"}<|call|> Done.`
	const harmonyCall = `<|channel|>commentary to=functions.echo<|message|>{"text":"ok"}<|call|>`
	const openProse = " Related [2 notes. "
	for _, tc := range []struct {
		name, raw string
		wantInput *input
		wantText  string
		wantErr   bool
		errText   string
	}{
		{
			name: "inline reordered escaped key and Unicode",
			raw:  `{"args":{"text":"<|im_end|> \ud83d\ude80"},"na\u006de":"echo"}`, wantInput: &input{Text: "<|im_end|> 🚀"},
		},
		{
			name:      "Phi quoted wrapper and turn delimiter",
			raw:       `<|tool_call>{"name":"echo","args":{"text":"</tool_call><|end|>"}}</tool_call>`,
			wantInput: &input{Text: "</tool_call><|end|>"},
		},
		{
			name:      "Qwen literal spaces and nested marker",
			raw:       `<function=echo><parameter=text>  hi  </parameter><parameter=object>{"value":"</parameter><tool_call>"}</parameter></function>`,
			wantInput: &input{Text: "  hi  ", Object: &object{Value: "</parameter><tool_call>"}},
		},
		{
			name:      "Qwen unambiguous quoted closing tag",
			raw:       `<function=echo><parameter=text>"left</parameter>right"</parameter></function>`,
			wantInput: &input{Text: `"left</parameter>right"`},
		},
		{
			name:      "Gemma arrays, nested marker, whitespace and JSON escapes",
			raw:       `<|tool_call>call:echo{text:<|"|>say <tool_call|> <|tool_call_start|> [TOOL_CALLS]<|"|> ,items:[<|"|> a <|"|>,"b\nc"],object:{value:<|"|><think><|"|>}}<tool_call|>`,
			wantInput: &input{Text: "say <tool_call|> <|tool_call_start|> [TOOL_CALLS]", Items: []string{" a ", "b\nc"}, Object: &object{Value: "<think>"}},
		},
		{
			name:      "GLM horizontal spacing and multiline value",
			raw:       "echo<arg_key>text</arg_key> \t<arg_value> { \n x\n</arg_value>",
			wantInput: &input{Text: " { \n x\n"},
		},
		{
			name:      "GLM unambiguous quoted closing tag",
			raw:       `echo<arg_key>text</arg_key><arg_value>"left</arg_value>right"</arg_value>`,
			wantInput: &input{Text: `"left</arg_value>right"`},
		},
		{
			name:      "LFM2 Python exact escapes and nested literals",
			raw:       `<|tool_call_start|>[echo(text='\x41\u0062\' <|tool_call_end|>',items=[' one ', "two"],object={'value': 'line\nnext'})]<|tool_call_end|>`,
			wantInput: &input{Text: "Ab' <|tool_call_end|>", Items: []string{" one ", "two"}, Object: &object{Value: "line\nnext"}},
		},
		{
			name:      "LFM2 JSON marker and whitespace in array",
			raw:       `<|tool_call_start|>[ {"name":"echo","args":{"text":"<|tool_call_end|>","items":[" a "]}} ]<|tool_call_end|>`,
			wantInput: &input{Text: "<|tool_call_end|>", Items: []string{" a "}},
		},
		{
			name:      "Devstral paired markers",
			raw:       `[TOOL_CALLS]echo[ARGS]{"text":"[ARGS][/TOOL_CALLS]"}[/TOOL_CALLS]`,
			wantInput: &input{Text: "[ARGS][/TOOL_CALLS]"},
		},
		{
			name:      "Harmony role recipient and quoted call end",
			raw:       `<|start|>assistant to=functions.echo<|channel|>commentary<|message|>{"text":"<|call|>"}<|call|>`,
			wantInput: &input{Text: "<|call|>"},
		},
		{
			name: "visible call after hidden call",
			raw:  "<think>" + wireTestJSON + "</think>" + wireTestGemma, wantInput: &input{Text: "ok"},
		},
		{
			name:     "hidden think with closer inside argument",
			raw:      `<think><tool_call>{"name":"echo","args":{"text":"</think>"}}</tool_call>` + wireTestJSON + "</think>" + done,
			wantText: done,
		},
		{name: "orphan think closer", raw: wireTestJSON + "</think>" + done, wantText: done},
		{name: "Gemma hidden thought", raw: "<|channel>thought\n" + wireTestGemma + "<channel|>" + done, wantText: done},
		{name: "Gemma orphan thought closer", raw: wireTestGemma + "<channel|>" + done, wantText: done},
		{
			name: "Harmony reasoning and final literal example",
			raw: `<|channel|>analysis<|message|>` + wireTestJSON + `<|end|>` +
				`<|channel|>final<|message|>` + harmonyExample + `<|return|>` + wireTestJSON,
			wantText: harmonyExample,
		},
		{name: "ChatML end", raw: done + `<|im_end|>` + wireTestJSON, wantText: done},
		{name: "nested JSON is prose", raw: `{"payload":{"name":"echo","args":{"text":"ok"}}}`, wantText: `{"payload":{"name":"echo","args":{"text":"ok"}}}`},
		{
			name:     "unclosed hidden child cannot reveal a call",
			raw:      done + `<think><tool_call>{"name":"echo","args":` + `"</think>"` + wireTestJSON,
			wantText: done,
		},
		{
			name:     "fully framed hidden duplicate preserves answer",
			raw:      `<think><tool_call>{"name":"echo","args":{"text":"first","text":"</think>"}}</tool_call></think>` + done,
			wantText: done,
		},
		{
			name:    "unclosed prose bracket before truncated explicit call",
			raw:     wireTestJSON + openProse + `<tool_call>{"name":"echo","args":`,
			wantErr: true,
			errText: "invalid JSON tool call body",
		},
		{
			name:    "unclosed prose bracket before mixed explicit call",
			raw:     wireTestJSON + openProse + wireTestGemma,
			wantErr: true,
			errText: "mixed JSON and Gemma",
		},
		{
			name:    "Qwen quote spanning duplicate parameters",
			raw:     `<function=echo><parameter=text>"left</parameter><parameter=text>right"</parameter></function>`,
			wantErr: true,
			errText: "ambiguous </parameter>",
		},
		{
			name:    "GLM quote spanning duplicate arguments",
			raw:     `echo<arg_key>text</arg_key><arg_value>"left</arg_value><arg_key>text</arg_key><arg_value>right"</arg_value>`,
			wantErr: true,
			errText: "ambiguous </arg_value>",
		},
		{
			name:    "Harmony truncated shorthand after valid call",
			raw:     harmonyCall + ".echo <|messag",
			wantErr: true,
			errText: "incomplete or misplaced tool call delimiter",
		},
		{
			name:    "Qwen nested duplicates after valid call",
			raw:     wireTestQwen + `<function=echo><parameter=text>bad</parameter><parameter=object>{"value":"a","va\u006cue":"b"}</parameter></function>`,
			wantErr: true,
		},
		{name: "Gemma malformed optional pair after valid call", raw: wireTestGemma + `call:echo{text:"bad",items}`, wantErr: true},
		{name: "GLM incomplete value before valid call", raw: `echo<arg_key>text</arg_key><arg_value>bad` + "\n" + wireTestGLM, wantErr: true},
		{name: "mixed legacy after valid call", raw: wireTestGLM + "\n" + wireTestPython, wantErr: true},
		{name: "Mistral dangling args after valid call", raw: wireTestMistral + `[ARGS]{"text":"bad"}`, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			op := &testTextOperator{
				testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
					return 42, nil
				}},
				generate: func(_ context.Context, _ llama.Model, in TextGenerationInput, _ func(context.Context, string) error) (TextGenerationResult, error) {
					if in.Messages[len(in.Messages)-1].GetRole() == "tool" {
						return TextGenerationResult{Text: done}, nil
					}
					return TextGenerationResult{Text: tc.raw}, nil
				},
			}
			plugin, g := initLoadingTest(t, op)
			def := testModelDefinition()
			def.ToolCalling = true
			model := plugin.DefineModel(g, def, nil)
			var submitted []input
			tool := genkit.DefineTool(g, "echo", "Echo the input", func(_ *ai.ToolContext, got input) (string, error) {
				submitted = append(submitted, got)
				return "ok", nil
			})
			resp, err := genkit.Generate(t.Context(), g, ai.WithModel(model), ai.WithTools(tool), ai.WithPrompt("Use echo."))
			if tc.wantErr {
				if err == nil || len(submitted) != 0 || !strings.Contains(err.Error(), tc.errText) {
					t.Fatalf("dispatch = (%v, %v), want error containing %q and zero calls", submitted, err, tc.errText)
				}
				return
			}
			wantText := tc.wantText
			var want []input
			if tc.wantInput != nil {
				want = []input{*tc.wantInput}
				wantText = done
			}
			if diff := cmp.Diff(want, submitted); diff != "" {
				t.Errorf("handler inputs (-want +got):\n%s", diff)
			}
			if err != nil || resp.Text() != wantText {
				t.Fatalf("response = (%v, %v), want text %q", resp, err, wantText)
			}
		})
	}
}
