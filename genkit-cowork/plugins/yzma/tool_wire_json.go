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
	"strconv"
	"strings"
	"unicode"

	"github.com/google/uuid"
	"github.com/hybridgroup/yzma/pkg/message"
)

type wireToolCall struct {
	name    string
	args    map[string]any
	untyped map[string]string
}

// toolProse shields complete JSON objects classified as text from Yzma's
// family parser and substring-based markup cleaner. Only restore text that
// survives cleaning, so blocks containing a placeholder can still be removed.
type toolProse struct {
	source       string
	prefix       string
	replacements []string
	glmValues    map[string][]string // Per-key replacements inside confirmed GLM values.
	masked       strings.Builder
	end          int
}

func (p *toolProse) shield(start, end int) string {
	if p.prefix == "" {
		for {
			p.prefix = "\x00" + uuid.NewString() + "_"
			if !strings.Contains(p.source, p.prefix) {
				break
			}
		}
	}
	token := p.prefix + strconv.Itoa(len(p.replacements)/2) + "\x00"
	p.replacements = append(p.replacements, token, p.source[start:end])
	p.masked.WriteString(p.source[p.end:start])
	p.masked.WriteString(token)
	p.end = end
	return token
}

func (p *toolProse) fallbackInput() string {
	if p.end == 0 {
		return p.source
	}
	return p.masked.String() + p.source[p.end:]
}

func (p *toolProse) clean(raw string) string {
	text := message.StripMarkup(raw)
	if len(p.replacements) > 0 {
		text = strings.NewReplacer(p.replacements...).Replace(text)
	}
	return text
}

const (
	wireToolName      = `[a-zA-Z_][\w./-]*`
	harmonyCallEnd    = "<|call|>"
	harmonyTurnEnd    = "<|return|>"
	harmonyMessageEnd = "<|end|>"
	harmonyAssistant  = `(?:<\|start\|>assistant)?`
	harmonyJSONBody   = `\s*(?:<\|constrain\|>\s*json\s*)?<\|message\|>`
)

var (
	harmonyRecipient     = regexp.MustCompile(`^functions\.(` + wireToolName + `)` + harmonyJSONBody)
	harmonyRoleRecipient = regexp.MustCompile(`^functions\.(` + wireToolName + `)\s*<\|channel\|>commentary` + harmonyJSONBody)
	harmonyBoundary      = regexp.MustCompile(`<\|(?:end|return|start|channel|call)\|>`)
	toolSyntax           = regexp.MustCompile(`<\|tool_call(?:_start\|>|>)|</?tool_call>|` + qwenTagPattern + `|<arg_key>|\[TOOL_CALLS\]|` +
		`<\|start\|>assistant\s+to=|` + harmonyAssistant + `<\|channel\|>[a-z]+\s+to=|` +
		harmonyAssistant + `<\|channel\|>(?:analysis|commentary|final)\s*<\|message\|>|<\|call\|>|(?:\.` +
		wireToolName + `\s*)?<\|message\|>|\{\s*"name"\s*:|call:\s*` + wireToolName + `\s*\{`)
	inlineJSONToken = regexp.MustCompile(`"(?:[^"\\]|\\(?s:.))*"|[{}\[\]]`)
	lfmPython       = regexp.MustCompile(`^\[\s*` + wireToolName + `\s*\(`)
)

// parseWireCalls reads JSON and Qwen envelopes before falling back to the
// family parsers. A non-nil empty slice means a Harmony text frame was
// consumed and must not be reinterpreted as a legacy tool call.
func (p *toolProse) parseWireCalls() ([]wireToolCall, string, error) {
	// Keep each scanner tail a suffix of input, even when a parser trims space.
	input := strings.TrimRightFunc(p.source, unicode.IsSpace)
	raw := input
	var calls, qwenCalls []wireToolCall
	var text strings.Builder
	var glmKey string
	var glmValueStart, glmValueEnd int
	nonJSON, harmonyText := false, false
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
		case marker == "<function=" || ((marker == "<tool_call>" || marker == "<|tool_call>") &&
			strings.HasPrefix(rest, "<function=")):
			wrapped := marker != "<function="
			if !wrapped {
				rest = raw[loc[0]:]
			}
			parsed, tail, err := parseQwenEnvelope(rest, wrapped)
			if err != nil {
				return nil, "", err
			}
			qwenCalls = append(qwenCalls, parsed...)
			raw = tail
			continue
		case strings.HasPrefix(marker, "<function") || strings.HasPrefix(marker, "</function") ||
			strings.HasPrefix(marker, "<parameter") || strings.HasPrefix(marker, "</parameter"):
			if len(qwenCalls) > 0 {
				return nil, "", fmt.Errorf("yzma: misplaced Qwen delimiter %s", marker)
			}
			text.WriteString(marker)
			raw = raw[loc[1]:]
			continue
		case marker == "<tool_call>" || marker == "<|tool_call>":
			if marker == "<|tool_call>" && strings.HasPrefix(rest, "call:") {
				nonJSON = true
				raw = rest
				continue
			}
			closeTag = "</tool_call>"
		case strings.HasPrefix(marker, "call:"):
			nonJSON = true
			start := loc[1] - 1 // Includes the opening brace matched by toolSyntax.
			if end, complete := scanToolStructure(raw[start:], true); complete {
				raw = raw[start+end:]
			} else {
				raw = rest // Leave incomplete structure to the family fallback.
			}
			continue
		case marker == "<arg_key>":
			nonJSON = true
			// GLM's parser treats everything in an argument value as text.
			// Recognize horizontal spacing, but don't search past stray text.
			if key, afterKey, ok := strings.Cut(raw[loc[1]:], "</arg_key>"); ok {
				afterGap := strings.TrimLeft(afterKey, " \t")
				if value, ok := strings.CutPrefix(afterGap, "<arg_value>"); ok {
					if body, tail, ok := strings.Cut(value, "</arg_value>"); ok &&
						!strings.Contains(raw[loc[1]:len(raw)-len(tail)], "\n") {
						if afterGap == afterKey {
							raw = tail
							continue
						}
						// Keep the scanner's existing mixed-syntax checks for
						// spaced pairs. Restore shielded data only as an argument.
						glmKey = key
						glmValueStart = len(input) - len(value)
						glmValueEnd = glmValueStart + len(body)
					}
				}
			}
			raw = rest
			continue
		case marker == "</tool_call>":
			nonJSON = true
			raw = rest
			continue
		case marker == "<|tool_call_start|>":
			if lfmPython.MatchString(rest) {
				nonJSON = true
				raw = rest
				if end, complete := scanToolStructure(rest, true); complete {
					raw = rest[end:]
				}
				continue
			}
			closeTag = "<|tool_call_end|>"
		case marker == "[TOOL_CALLS]":
			before, after, ok := strings.Cut(rest, "[ARGS]")
			name, rest = strings.TrimSpace(before), after
			if !ok || name == "" || strings.ContainsAny(name, " \t\r\n[]{}<>") {
				return nil, "", fmt.Errorf("yzma: malformed Mistral tool call")
			}
		case strings.HasSuffix(marker, "to="):
			// Harmony allows the recipient in either the role or channel
			// section. Only commentary calls to functions use our tool schema.
			var header []string
			switch {
			case strings.Contains(marker, "<|channel|>commentary"):
				header = harmonyRecipient.FindStringSubmatch(rest)
			case !strings.Contains(marker, "<|channel|>"):
				header = harmonyRoleRecipient.FindStringSubmatch(rest)
			}
			if header == nil {
				return nil, "", fmt.Errorf("yzma: unsupported or malformed Harmony tool call header")
			}
			name, rest, closeTag = header[1], rest[len(header[0]):], harmonyCallEnd
		case strings.Contains(marker, "<|channel|>"):
			// Reasoning can contain tool syntax as text. Only a recipient
			// in the header introduces a Harmony call.
			harmonyText = true
			body := raw[loc[1]:]
			raw = ""
			if boundary := harmonyBoundary.FindStringIndex(body); boundary != nil {
				switch body[boundary[0]:boundary[1]] {
				case harmonyMessageEnd, harmonyTurnEnd:
					raw = body[boundary[1]:]
					body = body[:boundary[0]]
				default:
					return nil, "", fmt.Errorf("yzma: unclosed Harmony message")
				}
			}
			if !strings.Contains(marker, "<|channel|>analysis") {
				text.WriteString(body)
			}
			continue
		case marker == harmonyCallEnd:
			return nil, "", fmt.Errorf("yzma: misplaced Harmony call terminator")
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
			if inline && !hasMalformedInlineArguments(rest) {
				text.WriteString(marker)
				raw = raw[loc[1]:]
				continue
			}
			return nil, "", fmt.Errorf("yzma: malformed JSON tool call: %w", err)
		}
		raw = rest[decoder.InputOffset():]
		if closeTag != "" {
			var ok bool
			raw, ok = strings.CutPrefix(strings.TrimSpace(raw), closeTag)
			if !ok {
				return nil, "", fmt.Errorf("yzma: missing %s after JSON tool call", closeTag)
			}
		} else if strings.HasSuffix(marker, "<|message|>") {
			// The older .name shorthand is complete after its JSON object.
			if tail, ok := strings.CutPrefix(strings.TrimSpace(raw), harmonyCallEnd); ok {
				raw = tail
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
					start := len(input) - len(rest)
					end := start + int(decoder.InputOffset())
					token := p.shield(start, end)
					if start >= glmValueStart && end <= glmValueEnd {
						if p.glmValues == nil {
							p.glmValues = make(map[string][]string)
						}
						p.glmValues[glmKey] = append(p.glmValues[glmKey], token, input[start:end])
					}
					text.WriteString(token)
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
	if (nonJSON && (len(calls) > 0 || len(qwenCalls) > 0 || harmonyText)) || (len(calls) > 0 && len(qwenCalls) > 0) {
		return nil, "", fmt.Errorf("yzma: mixed or unmatched tool call syntax")
	}
	calls = append(calls, qwenCalls...)
	if calls == nil && harmonyText {
		calls = []wireToolCall{}
	}
	if calls == nil {
		return nil, "", nil // The family fallback needs the original markup.
	}
	text.WriteString(raw)
	return calls, p.clean(text.String()), nil
}

// hasMalformedInlineArguments looks for an argument key even when JSON
// decoding fails before reaching it. Quoted values and nested keys don't
// turn ordinary named data into a tool call.
func hasMalformedInlineArguments(raw string) bool {
	depth := 0
	for {
		loc := inlineJSONToken.FindStringIndex(raw)
		if loc == nil {
			// A complete key name followed by EOF is visible even if its
			// closing quote was never generated.
			key, afterComma := strings.CutPrefix(strings.TrimSpace(raw), ",")
			key = strings.TrimSpace(key)
			return depth == 1 && afterComma && (key == `"args` || key == `"arguments`)
		}
		token := raw[loc[0]:loc[1]]
		before := strings.TrimSpace(raw[:loc[0]])
		raw = raw[loc[1]:]
		switch token[0] {
		case '{', '[':
			depth++
		case '}', ']':
			depth--
			if depth == 0 {
				return false
			}
		case '"':
			if depth == 1 {
				tail := strings.TrimSpace(raw)
				if strings.HasSuffix(before, ",") || (strings.HasPrefix(tail, ":") && !strings.HasSuffix(before, ":")) {
					var key string
					if json.Unmarshal([]byte(token), &key) == nil && (key == "arguments" || key == "args") {
						return true
					}
				}
			}
		}
	}
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
