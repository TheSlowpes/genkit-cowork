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
	"fmt"
	"regexp"
	"strings"
)

const qwenTagPattern = `</?(?:function|parameter)(?:[=>\s]|$)`

// Parameter values are literal text apart from the call's own delimiters.
var qwenDelimiter = regexp.MustCompile(qwenTagPattern + `|</?\|?tool_call(?:[=>\s]|$)`)

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
		delimiter := qwenDelimiter.FindStringIndex(value)
		const closeTag = "</parameter>"
		if delimiter == nil || !strings.HasPrefix(value[delimiter[0]:], closeTag) {
			return wireToolCall{}, "", fmt.Errorf("yzma: missing </parameter> for Qwen parameter %q", key)
		}
		if _, duplicate := args[key]; duplicate {
			return wireToolCall{}, "", fmt.Errorf("yzma: duplicate Qwen parameter %q", key)
		}
		args[key] = strings.TrimSpace(value[:delimiter[0]])
		body = value[delimiter[0]+len(closeTag):]
	}
}

func qwenOpenTag(raw, prefix string) (string, string, bool) {
	tail, ok := strings.CutPrefix(raw, prefix)
	if !ok {
		return "", "", false
	}
	name, body, closed := strings.Cut(tail, ">")
	name = strings.TrimSpace(name)
	return name, body, closed && name != "" && !strings.Contains(name, "<")
}
