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
	"unicode/utf8"
)

// Qwen parameter values are literal text apart from the call's own delimiters.
var qwenDelimiter = regexp.MustCompile(`</?(?:function|parameter)(?:[=>\s]|$)|<\|?/?tool_call[|>]`)

// parseQwenEnvelope handles both wrapped and bare functions. All functions in
// a wrapper must close before the wrapper closes.
func parseQwenEnvelope(raw string, wrapped bool) ([]wireToolCall, string, error) {
	var calls []wireToolCall
	for {
		call, tail, err := parseQwenFunction(raw)
		if err != nil {
			return nil, "", err
		}
		calls = append(calls, call)
		if !wrapped {
			return calls, tail, nil
		}
		raw = strings.TrimSpace(tail)
		if tail, ok := strings.CutPrefix(raw, "</tool_call>"); ok {
			return calls, tail, nil
		}
		if !strings.HasPrefix(raw, "<function=") {
			return nil, "", fmt.Errorf("yzma: missing </tool_call> after Qwen function")
		}
	}
}

func parseQwenFunction(raw string) (wireToolCall, string, error) {
	name, body, ok := qwenOpenTag(raw, "<function=")
	if !ok {
		return wireToolCall{}, "", fmt.Errorf("yzma: malformed Qwen function tag")
	}
	args := make(map[string]string)
	for {
		body = strings.TrimSpace(body)
		if tail, ok := strings.CutPrefix(body, "</function>"); ok {
			return wireToolCall{name: name, untyped: args}, tail, nil
		}
		key, value, ok := qwenOpenTag(body, "<parameter=")
		if !ok {
			return wireToolCall{}, "", fmt.Errorf("yzma: expected a Qwen parameter or </function> in %q", name)
		}
		arg, tail, closed := taggedValue(value, qwenDelimiter, "</parameter>")
		if !closed {
			return wireToolCall{}, "", fmt.Errorf("yzma: missing </parameter> for Qwen parameter %q", key)
		}
		if _, duplicate := args[key]; duplicate {
			return wireToolCall{}, "", fmt.Errorf("yzma: duplicate Qwen parameter %q", key)
		}
		// Qwen templates put a line break at both tag edges. Other
		// whitespace belongs to the argument, not to the framing.
		for _, newline := range []string{"\r\n", "\n"} {
			if content, ok := strings.CutPrefix(arg, newline); ok {
				if content, ok := strings.CutSuffix(content, newline); ok {
					arg = content
					break
				}
			}
		}
		args[key] = arg
		body = tail
	}
}

const (
	glmKeyOpen   = "<arg_key>"
	glmKeyClose  = "</arg_key>"
	glmValueOpen = "<arg_value>"
	glmValueEnd  = "</arg_value>"
	glmStart     = `^[ \t]*` + wireToolName + `[ \t]*` + glmKeyOpen
)

var glmDelimiter = regexp.MustCompile(`</?arg_(?:key|value)(?:[>\s]|$)`)

func parseGLMCall(raw string) (wireToolCall, string, error) {
	name, _, _ := strings.Cut(raw, glmKeyOpen)
	args := map[string]string{}
	body := raw[len(name):]
	for {
		key, rest, ok := strings.Cut(body[len(glmKeyOpen):], glmKeyClose)
		if !ok || strings.TrimSpace(key) == "" || !utf8.ValidString(key) || strings.ContainsAny(key, "<>\r\n") {
			return wireToolCall{}, body, fmt.Errorf("malformed GLM argument key")
		}
		value, ok := strings.CutPrefix(strings.TrimLeft(rest, " \t"), glmValueOpen)
		if !ok {
			return wireToolCall{}, body, fmt.Errorf("missing GLM value for argument %q", key)
		}
		arg, tail, ok := taggedValue(value, glmDelimiter, glmValueEnd)
		if !ok {
			return wireToolCall{}, body, fmt.Errorf("unclosed GLM value for argument %q", key)
		}
		if _, exists := args[key]; exists {
			return wireToolCall{}, body, fmt.Errorf("duplicate GLM argument %q", key)
		}
		args[key] = arg
		body = strings.TrimLeft(tail, " \t")
		if !strings.HasPrefix(body, glmKeyOpen) {
			return wireToolCall{name: strings.TrimSpace(name), untyped: args}, tail, nil
		}
	}
}

// taggedValue respects literal delimiters in a leading JSON value. JSON
// syntax is enough to locate the boundary; duplicate and type checks happen
// later, when the offered schema says the value should be decoded as JSON.
func taggedValue(raw string, delimiter *regexp.Regexp, close string) (string, string, bool) {
	body := strings.TrimLeft(raw, " \r\n\t")
	start := 0
	if body != "" && strings.ContainsRune(`{["`, rune(body[0])) {
		d := json.NewDecoder(strings.NewReader(body))
		var value json.RawMessage
		if d.Decode(&value) == nil {
			start = len(raw) - len(body) + int(d.InputOffset())
		}
	}
	loc := delimiter.FindStringIndex(raw[start:])
	if loc == nil || !strings.HasPrefix(raw[start+loc[0]:], close) {
		return "", raw, false
	}
	end := start + loc[0]
	return raw[:end], raw[end+len(close):], true
}

func qwenOpenTag(raw, prefix string) (string, string, bool) {
	tail, ok := strings.CutPrefix(raw, prefix)
	if !ok {
		return "", "", false
	}
	name, body, closed := strings.Cut(tail, ">")
	name = strings.TrimSpace(name)
	return name, body, closed && name != "" && utf8.ValidString(name) && !strings.Contains(name, "<")
}
