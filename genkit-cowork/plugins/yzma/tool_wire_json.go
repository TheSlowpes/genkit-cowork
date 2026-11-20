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

// inlineJSONSpan classifies a JSON-shaped span without decoding its values.
// It also identifies call keys in broken objects. In either case a nested
// value or a quoted marker can never become a separate outer frame.
func inlineJSONSpan(raw string) (end int, candidate, complete bool) {
	var stack []byte
	var name, args, keyPosition bool
	var previous byte
	for i := 0; i < len(raw); {
		ch := raw[i]
		switch ch {
		case '"':
			end, closed := quotedEnd(raw[i:])
			// A missing comma is still a malformed candidate when the
			// next quoted token has its own colon. A value after a colon
			// is never a candidate key.
			if len(stack) == 1 && stack[0] == '{' && (keyPosition ||
				previous != ':' && strings.HasPrefix(strings.TrimLeft(raw[i+end:], " \t\r\n"), ":")) {
				var key string
				if closed {
					_ = json.Unmarshal([]byte(raw[i:i+end]), &key)
				} else {
					key = raw[i+1:]
				}
				name = name || key == "name"
				args = args || key == "args" || key == "arguments"
				keyPosition = false
			}
			i += end
		case '{', '[':
			stack = append(stack, raw[i])
			keyPosition = i == 0
			i++
		case '}', ']':
			if (stack[len(stack)-1] == '{') != (raw[i] == '}') {
				return len(raw), args, false
			}
			stack = stack[:len(stack)-1]
			i++
			if len(stack) == 0 {
				return i, name && args, true
			}
		case ',':
			keyPosition = len(stack) == 1
			i++
		case ' ', '\t', '\r', '\n':
			i++
			continue
		default:
			keyPosition = false
			i++
		}
		previous = ch
	}
	return len(raw), args, false
}

// quotedEnd returns the extent of a JSON or Python quoted token, even when
// its escape syntax is invalid. Dialect-specific decoding happens separately.
func quotedEnd(raw string) (int, bool) {
	for i := 1; i < len(raw); i++ {
		if raw[i] == raw[0] {
			return i + 1, true
		}
		if raw[i] == '\\' {
			i++
		}
	}
	return len(raw), false
}

func jsonCallObject(value any) (wireToolCall, error) {
	call, ok := value.(map[string]any)
	if !ok {
		return wireToolCall{}, fmt.Errorf("expected a JSON tool call object")
	}
	name, _ := call["name"].(string)
	args, hasArgs := call["arguments"]
	alias, hasAlias := call["args"]
	if !hasArgs {
		args = alias
	}
	object, ok := args.(map[string]any)
	if strings.TrimSpace(name) == "" || !ok || (hasArgs && hasAlias) {
		return wireToolCall{}, fmt.Errorf("tool call requires a name and exactly one JSON arguments object")
	}
	return wireToolCall{name: name, args: object}, nil
}

func splitWireJSON(raw string) (any, string, error) {
	d := json.NewDecoder(strings.NewReader(raw))
	d.UseNumber()
	value, err := readWireJSON(d, raw, 0)
	return value, raw[d.InputOffset():], err
}

// readWireJSON preserves numbers and rejects duplicate keys at every depth.
func readWireJSON(d *json.Decoder, raw string, depth int) (any, error) {
	if depth > 100 {
		return nil, fmt.Errorf("JSON tool call exceeds 100 levels")
	}
	switch token, err := wireJSONToken(d, raw); {
	case err != nil:
		return nil, err
	case token == json.Delim('{'):
		object := make(map[string]any)
		for d.More() {
			key, err := wireJSONToken(d, raw)
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
			object[name], err = readWireJSON(d, raw, depth+1)
			if err != nil {
				return nil, err
			}
		}
		_, err := d.Token()
		return object, err
	case token == json.Delim('['):
		array := []any{}
		for d.More() {
			item, err := readWireJSON(d, raw, depth+1)
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

func wireJSONToken(d *json.Decoder, raw string) (json.Token, error) {
	start := d.InputOffset()
	token, err := d.Token()
	if _, ok := token.(string); ok {
		spelling := raw[start:d.InputOffset()]
		spelling = spelling[strings.IndexByte(spelling, '"'):]
		if !validJSONStringUnicode(spelling) {
			return nil, fmt.Errorf("invalid Unicode in JSON string")
		}
	}
	return token, err
}

// encoding/json substitutes U+FFFD for invalid UTF-8 or an unpaired UTF-16
// surrogate. Reject those spellings rather than dispatching a changed string.
func validJSONStringUnicode(raw string) bool {
	if !utf8.ValidString(raw) {
		return false
	}
	for i := 1; i+1 < len(raw); i++ {
		if raw[i] != '\\' {
			continue
		}
		i++
		if raw[i] != 'u' {
			continue
		}
		n, _ := strconv.ParseUint(raw[i+1:i+5], 16, 16) // JSON decoding checked the syntax.
		i += 4
		if n >= 0xdc00 && n <= 0xdfff {
			return false
		}
		if n >= 0xd800 && n <= 0xdbff {
			if !strings.HasPrefix(raw[i+1:], `\u`) {
				return false
			}
			low, _ := strconv.ParseUint(raw[i+3:i+7], 16, 16)
			if low < 0xdc00 || low > 0xdfff {
				return false
			}
			i += 6
		}
	}
	return true
}
