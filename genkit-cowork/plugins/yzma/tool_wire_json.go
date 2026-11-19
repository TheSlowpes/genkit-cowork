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
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/hybridgroup/yzma/pkg/message"
)

type wireToolCall struct {
	name    string
	args    map[string]any
	untyped map[string]string
}

const wireToolName = `[a-zA-Z_][\w./-]*`

var (
	toolSyntax = regexp.MustCompile(`<\|tool_call(?:_start\|>|>)|</?tool_call>|<function=|<arg_key>|\[TOOL_CALLS\]|(?:\.` +
		wireToolName + `\s*)?<\|message\|>|\{\s*"name"\s*:|call:\s*` + wireToolName + `\s*\{`)
	lfmPython = regexp.MustCompile(`^\[\s*` + wireToolName + `\s*\(`)
)

// parseJSONToolCalls consumes JSON directly. Yzma's parser stringifies values,
// rounds numbers, flattens objects, and may skip malformed calls in a batch.
// Use its parser only for non-JSON formats, and never mix the two paths.
func parseJSONToolCalls(raw string) ([]wireToolCall, string, error) {
	var calls []wireToolCall
	var text strings.Builder
	nonJSON := false
	for {
		loc := toolSyntax.FindStringIndex(raw)
		if loc == nil {
			break
		}
		text.WriteString(raw[:loc[0]])
		marker := raw[loc[0]:loc[1]]
		rest := strings.TrimSpace(raw[loc[1]:])
		var name, closeTag string
		inline := false
		switch {
		case marker == "<tool_call>" || marker == "<|tool_call>":
			if strings.HasPrefix(rest, "<function=") || (marker == "<|tool_call>" && strings.HasPrefix(rest, "call:")) {
				nonJSON = true
				raw = rest
				continue
			}
			closeTag = "</tool_call>"
		case marker == "</tool_call>" || strings.HasPrefix(marker, "call:") || marker == "<function=" || marker == "<arg_key>":
			nonJSON = true
			raw = rest
			continue
		case marker == "<|tool_call_start|>":
			if lfmPython.MatchString(rest) {
				nonJSON = true
				raw = rest
				continue
			}
			closeTag = "<|tool_call_end|>"
		case marker == "[TOOL_CALLS]":
			before, after, ok := strings.Cut(rest, "[ARGS]")
			name, rest = strings.TrimSpace(before), after
			if !ok || name == "" || strings.ContainsAny(name, " \t\r\n[]{}<>") {
				return nil, "", fmt.Errorf("yzma: malformed Mistral tool call")
			}
		case strings.HasSuffix(marker, "<|message|>"):
			// <|message|> is also used on ordinary GPT assistant turns.
			name = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(marker, "."), "<|message|>"))
			if name == "" {
				text.WriteString(marker)
				raw = raw[loc[1]:]
				continue
			}
		default: // Inline {"name": ...} form includes the opening brace.
			rest = raw[loc[0]:]
			inline = true
		}

		decoder := json.NewDecoder(strings.NewReader(rest))
		decoder.UseNumber()
		value, err := readWireJSON(decoder, 0)
		if err != nil {
			return nil, "", fmt.Errorf("yzma: malformed JSON tool call: %w", err)
		}
		raw = rest[decoder.InputOffset():]
		if closeTag != "" {
			var ok bool
			raw, ok = strings.CutPrefix(strings.TrimSpace(raw), closeTag)
			if !ok {
				return nil, "", fmt.Errorf("yzma: missing %s after JSON tool call", closeTag)
			}
		}
		switch v := value.(type) {
		case map[string]any:
			if name != "" {
				calls = append(calls, wireToolCall{name: name, args: v})
				continue
			}
			if inline {
				_, hasArgs := v["arguments"]
				_, hasAlias := v["args"]
				if !hasArgs && !hasAlias {
					text.WriteString(rest[:decoder.InputOffset()])
					continue
				}
			}
		case []any:
			if closeTag == "<|tool_call_end|>" && len(v) > 0 {
				for _, item := range v {
					call, err := jsonCallObject(item)
					if err != nil {
						return nil, "", err
					}
					calls = append(calls, call)
				}
				continue
			}
		}
		if name != "" {
			return nil, "", fmt.Errorf("yzma: tool arguments must be a JSON object")
		}
		call, err := jsonCallObject(value)
		if err != nil {
			return nil, "", err
		}
		calls = append(calls, call)
	}
	if nonJSON && len(calls) > 0 {
		return nil, "", fmt.Errorf("yzma: mixed or unmatched tool call syntax")
	}
	text.WriteString(raw)
	return calls, message.StripMarkup(text.String()), nil
}

func jsonCallObject(value any) (wireToolCall, error) {
	call, ok := value.(map[string]any)
	if !ok {
		return wireToolCall{}, fmt.Errorf("yzma: expected a JSON tool call object")
	}
	name, _ := call["name"].(string)
	args, hasArgs := call["arguments"]
	alias, hasAlias := call["args"]
	if !hasArgs {
		args = alias
	}
	object, ok := args.(map[string]any)
	if strings.TrimSpace(name) == "" || !ok || (hasArgs && hasAlias) {
		return wireToolCall{}, fmt.Errorf("yzma: tool call requires a name and exactly one JSON arguments object")
	}
	return wireToolCall{name: name, args: object}, nil
}

// readWireJSON preserves original numbers and rejects duplicate keys at every
// depth; accepting the last occurrence could change which call executes.
func readWireJSON(d *json.Decoder, depth int) (any, error) {
	if depth > 100 {
		return nil, fmt.Errorf("JSON tool call exceeds 100 levels")
	}
	token, err := d.Token()
	if err != nil {
		return nil, err
	}
	switch token {
	case json.Delim('{'):
		object := make(map[string]any)
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return nil, err
			}
			name, ok := key.(string)
			if !ok {
				return nil, fmt.Errorf("expected JSON object key")
			}
			if _, exists := object[name]; exists {
				return nil, fmt.Errorf("duplicate JSON key %q", name)
			}
			object[name], err = readWireJSON(d, depth+1)
			if err != nil {
				return nil, err
			}
		}
		_, err := d.Token()
		return object, err
	case json.Delim('['):
		array := []any{}
		for d.More() {
			item, err := readWireJSON(d, depth+1)
			if err != nil {
				return nil, err
			}
			array = append(array, item)
		}
		_, err := d.Token()
		return array, err
	default:
		return token, nil
	}
}
