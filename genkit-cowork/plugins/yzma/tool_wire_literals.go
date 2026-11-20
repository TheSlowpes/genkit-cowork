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
	"strconv"
	"strings"
	"unicode/utf8"
)

// literalReader shares the container grammar between Gemma and the supported
// subset of LFM2 Python literals. Neither dialect searches ahead for a missing
// separator or treats arbitrary bare expressions as strings.
type literalReader struct {
	tail   string
	python bool
}

func (r *literalReader) take(token string) bool {
	r.tail = strings.TrimSpace(r.tail)
	rest, ok := strings.CutPrefix(r.tail, token)
	if ok {
		r.tail = rest
	}
	return ok
}

func (r *literalReader) name() string {
	r.tail = strings.TrimSpace(r.tail)
	name := wireNamePrefix.FindString(r.tail)
	r.tail = r.tail[len(name):]
	return name
}

func (r *literalReader) key(quoted bool) (string, error) {
	r.tail = strings.TrimSpace(r.tail)
	if quoted {
		if value, found, err := r.string(); found {
			return value, err
		}
		if r.python {
			return "", fmt.Errorf("Python dictionary keys must be strings")
		}
	}
	if name := r.name(); name != "" {
		return name, nil
	}
	return "", fmt.Errorf("expected argument key")
}

func (r *literalReader) object(close, assign string, quoted bool, depth int) (map[string]any, error) {
	args := map[string]any{}
	if r.take(close) {
		return args, nil
	}
	for {
		key, err := r.key(quoted)
		if err != nil {
			return nil, err
		}
		if !r.take(assign) {
			return nil, fmt.Errorf("expected %s after argument %q", assign, key)
		}
		if _, exists := args[key]; exists {
			return nil, fmt.Errorf("duplicate argument %q", key)
		}
		args[key], err = r.value(depth + 1)
		if err != nil {
			return nil, fmt.Errorf("argument %q: %w", key, err)
		}
		if done, err := r.next(close); done || err != nil {
			return args, err
		}
	}
}

// next consumes the separator or the closing token for one container item.
func (r *literalReader) next(close string) (bool, error) {
	if r.take(close) {
		return true, nil
	}
	if !r.take(",") {
		return false, fmt.Errorf("expected comma or %s", close)
	}
	return r.python && r.take(close), nil
}

func (r *literalReader) value(depth int) (any, error) {
	if depth > 100 {
		return nil, fmt.Errorf("tool argument exceeds 100 levels")
	}
	switch {
	case r.take("{"):
		return r.object("}", ":", true, depth)
	case r.take("["):
		items := []any{}
		if r.take("]") {
			return items, nil
		}
		for {
			item, err := r.value(depth + 1)
			if err != nil {
				return nil, err
			}
			items = append(items, item)
			if done, err := r.next("]"); done || err != nil {
				return items, err
			}
		}
	}
	if value, found, err := r.string(); found {
		return value, err
	}
	if r.python {
		switch {
		case r.take("True"):
			return true, nil
		case r.take("False"):
			return false, nil
		case r.take("None"):
			return nil, nil
		}
	}
	value, tail, err := splitWireJSON(r.tail)
	switch value.(type) {
	case json.Number, bool, nil:
		if err == nil {
			r.tail = tail
			return value, nil
		}
	}
	return nil, fmt.Errorf("expected a literal value")
}

func (r *literalReader) string() (string, bool, error) {
	if !r.python {
		for _, mark := range []string{`<|"|>`, `<">`, `<|>`} {
			if body, ok := strings.CutPrefix(r.tail, mark); ok {
				value, tail, closed := strings.Cut(body, mark)
				if !closed || !utf8.ValidString(value) {
					return "", true, fmt.Errorf("unclosed or invalid Gemma string")
				}
				r.tail = tail
				return value, true, nil
			}
		}
	}
	if !strings.HasPrefix(r.tail, `"`) && !(r.python && strings.HasPrefix(r.tail, "'")) {
		return "", false, nil
	}
	end, ok := quotedEnd(r.tail)
	if !ok {
		return "", true, fmt.Errorf("unclosed string")
	}
	raw := r.tail[:end]
	r.tail = r.tail[end:]
	if r.python {
		value, err := decodePythonString(raw[1 : end-1])
		return value, true, err
	}
	value, _, err := splitWireJSON(raw)
	if err != nil {
		return "", true, err
	}
	return value.(string), true, nil
}

func decodePythonString(body string) (string, error) {
	var out strings.Builder
	if !utf8.ValidString(body) || strings.ContainsAny(body, "\r\n") {
		return "", fmt.Errorf("invalid Python string")
	}
	for len(body) != 0 {
		text, escape, found := strings.Cut(body, `\`)
		out.WriteString(text)
		if !found {
			break
		}
		if escape == "" || !strings.ContainsRune(`'"\\abfnrtvxuU`, rune(escape[0])) {
			return "", fmt.Errorf("unsupported Python escape")
		}
		if escape[0] == '\'' || escape[0] == '"' {
			out.WriteByte(escape[0])
			body = escape[1:]
			continue
		}
		ch, _, tail, err := strconv.UnquoteChar(body[len(text):], 0)
		if err != nil {
			return "", fmt.Errorf("invalid Python escape: %w", err)
		}
		out.WriteRune(ch)
		body = tail
	}
	return out.String(), nil
}

func parseGemmaCall(raw string, wrapped bool) (wireToolCall, string, error) {
	r := literalReader{tail: raw}
	name := r.name()
	if name == "" || !r.take("{") {
		return wireToolCall{}, r.tail, fmt.Errorf("expected Gemma tool name and argument object")
	}
	args, err := r.object("}", ":", true, 0)
	if err == nil && wrapped && !r.take("<tool_call|>") {
		err = fmt.Errorf("missing <tool_call|> after Gemma call")
	}
	return wireToolCall{name: name, args: args, legacy: true}, r.tail, err
}

func parseLFMPython(raw string) ([]wireToolCall, string, error) {
	r := literalReader{tail: raw, python: true}
	if !r.take("[") {
		return nil, r.tail, fmt.Errorf("expected LFM2 Python call list")
	}
	var calls []wireToolCall
	for {
		name := r.name()
		if name == "" || !r.take("(") {
			return nil, r.tail, fmt.Errorf("expected a named call in LFM2 Python list")
		}
		args, err := r.object(")", "=", false, 0)
		if err != nil {
			return nil, r.tail, err
		}
		calls = append(calls, wireToolCall{name: name, args: args, legacy: true})
		done, err := r.next("]")
		if err != nil {
			return nil, r.tail, err
		}
		if done {
			return calls, r.tail, nil
		}
	}
}
