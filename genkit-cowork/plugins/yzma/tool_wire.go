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
	"unicode"
)

const (
	wireToolName = `[a-zA-Z_][\w./-]*`
	jsonOpen     = `<tool_call>`
	phiOpen      = `<|tool_call>`
	jsonClose    = `</tool_call>`
	lfmOpen      = `<|tool_call_start|>`
	lfmClose     = `<|tool_call_end|>`
	mistralOpen  = `[TOOL_CALLS]`
	mistralArgs  = `[ARGS]`
	mistralClose = `[/TOOL_CALLS]`
)

type wireToolCall struct {
	name    string
	args    map[string]any
	untyped map[string]string
	legacy  bool // Gemma and LFM Python keep the legacy integer dispatch limit.
}

type wireFamily string

const (
	wireJSON      wireFamily = "JSON"
	wireQwen      wireFamily = "Qwen"
	wireGemma     wireFamily = "Gemma"
	wireGLM       wireFamily = "GLM"
	wireLFMJSON   wireFamily = "LFM2 JSON"
	wireLFMPython wireFamily = "LFM2 Python"
	wireMistral   wireFamily = "Mistral"
	wireHarmony   wireFamily = "Harmony"
)

var (
	wireNamePrefix = regexp.MustCompile(`^` + wireToolName)
	// Only look for new frames in unclaimed text. The matched prefix is
	// intentionally short; each family consumes its own body and closing token.
	wireToken = regexp.MustCompile(`(?m)` + glmStart + `|\bcall:(?:\s*` + wireToolName + `\s*\{|` + wireToolName + `\z)|` +
		`\.` + wireToolName + `\s*<\|message\|>|` +
		`\{\s*["}]|\[\s*(?:[[{"\]0-9-]|(?:true|false|null)\b)|["<\[]`)
	lfmPythonStart = regexp.MustCompile(`^\[\s*` + wireToolName + `\s*\(`)
	turnBoundary   = regexp.MustCompile(`^(?:<\|?(?:im_start|im_end)\|?>|<\|(?:user|assistant|system|eot_id|end|return|endoftext)\|>|` +
		`<\|?turn\|?>|</turn>|<end_of_turn>|<start_of_turn>|</s>|<\|?turnend\|?>|<\|tool_response_start\|>)`)
	qwenProseTag = regexp.MustCompile(`^</?(?:function|parameter)>`)
	// These variants were accepted only by upstream detection/repair, not by
	// this grammar. Treat them as candidates so their contents cannot dispatch.
	unmatchedCallTag = regexp.MustCompile(`^(?:<\|?/?tool_?call(?:[_|>]|$)|</?(?:function|parameter)(?:[=\s>]|$)|` +
		`</?arg_(?:key|value)(?:[>\s]|$)|<\|call\|>|\[(?:/?TOOL_CALLS|ARGS)\])`)
	hiddenOpen = regexp.MustCompile(`^(?:<think>|<\|channel>thought\b|<channel>thought\b|` +
		`<\|?tool_?(?:response|result|code)\|?>)`)
	gemmaChannel = regexp.MustCompile(`^(?:<\|channel>[a-zA-Z_]\w*|<channel>[a-zA-Z_]\w*:)`)
)

type wireTurn struct {
	calls  []wireToolCall
	text   strings.Builder
	kind   wireFamily
	hidden []string
	// Plain Qwen delimiter names appear in prose. They only contradict a
	// Qwen call if the same turn also contains real Qwen function syntax.
	qwenDelimiters bool
	gemmaVisible   bool
}

// parseWireCalls walks the generated turn once at frame boundaries. Each
// branch consumes a source span as text, hidden framing, or calls. There is
// no second pass over the arguments or over the resulting visible text.
func parseWireCalls(source string) ([]wireToolCall, string, error) {
	turn := wireTurn{}
	source = strings.TrimRightFunc(source, unicode.IsSpace)
	for raw := source; raw != ""; {
		loc := wireToken.FindStringIndex(raw)
		if loc == nil {
			turn.visible(raw)
			break
		}
		turn.visible(raw[:loc[0]])
		token := raw[loc[0]:loc[1]]
		raw = raw[loc[0]:]
		if mark := hiddenOpen.FindString(raw); mark != "" {
			turn.hidden = append(turn.hidden, hiddenClose(mark))
			raw = raw[len(mark):]
			continue
		}
		if end := turn.closeHidden(raw); end != 0 {
			raw = raw[end:]
			continue
		}
		if turnBoundary.MatchString(raw) {
			break
		}
		if len(turn.hidden) == 0 && harmonyFrameStart(raw) {
			tail, err := turn.harmony(raw)
			if err != nil {
				return nil, "", fmt.Errorf("yzma: %w", err)
			}
			raw = tail
			continue
		}
		if channel := gemmaChannel.FindString(raw); channel != "" {
			if len(turn.hidden) == 0 {
				turn.gemmaVisible = true
			}
			raw = raw[len(channel):]
			continue
		}
		if tail, ok := strings.CutPrefix(raw, "<s>"); ok {
			raw = tail
			continue
		}
		if token == `"` {
			if n, ok := quotedEnd(raw); ok {
				turn.visible(raw[:n])
				raw = raw[n:]
				continue
			}
		}
		if strings.HasSuffix(token, glmKeyOpen) && len(turn.hidden) == 0 {
			start := len(source) - len(raw)
			if start > 0 && source[start-1] != '\n' {
				return nil, "", fmt.Errorf("yzma: GLM tool name must begin a line")
			}
		}
		kind, calls, tail, err := readCallSpan(raw, token)
		switch {
		case err != nil && len(turn.hidden) == 0:
			return nil, "", fmt.Errorf("yzma: %w", err)
		case err != nil:
			// Hidden frames ignore validity, not boundaries. An unknown
			// boundary yields an empty tail rather than exposing payload.
			raw = tail
		case tail != raw:
			if kind == "" {
				turn.visible(raw[:len(raw)-len(tail)])
			} else if err := turn.add(kind, calls); err != nil {
				return nil, "", err
			}
			raw = tail
		default:
			if len(turn.hidden) == 0 {
				if tag := qwenProseTag.FindString(raw); tag != "" {
					// <parameter> in a prose example is not <parameter=...>.
					turn.qwenDelimiters = turn.qwenDelimiters || tag != "<parameter>"
				} else if (unmatchedCallTag.MatchString(raw) || partialCallTag(raw)) && !isGLMProseTag(raw, turn.kind) {
					return nil, "", fmt.Errorf("yzma: incomplete or misplaced tool call delimiter")
				}
			}
			turn.visible(token)
			raw = raw[len(token):]
		}
	}
	if turn.kind == wireQwen && turn.qwenDelimiters {
		return nil, "", fmt.Errorf("yzma: misplaced Qwen delimiter")
	}
	return turn.calls, strings.TrimSpace(turn.text.String()), nil
}

// An isolated tag followed by prose is not a GLM call. A standalone tag,
// or one with attached key/value syntax, is a misplaced delimiter.
func isGLMProseTag(raw string, kind wireFamily) bool {
	tail, ok := strings.CutPrefix(raw, glmKeyOpen)
	if !ok || kind == wireGLM {
		return false
	}
	line, _, _ := strings.Cut(tail, "\n")
	return strings.HasPrefix(line, " ") && strings.TrimSpace(line) != "" && !glmDelimiter.MatchString(line)
}

func (t *wireTurn) visible(text string) {
	if len(t.hidden) == 0 {
		t.text.WriteString(text)
	}
}

func (t *wireTurn) add(family wireFamily, calls []wireToolCall) error {
	if len(t.hidden) != 0 {
		return nil
	}
	if t.kind != "" && t.kind != family {
		return fmt.Errorf("yzma: mixed %s and %s tool calls", t.kind, family)
	}
	t.kind = family
	t.calls = append(t.calls, calls...)
	return nil
}

func hiddenClose(open string) string {
	switch open {
	case "<think>":
		return "</think>"
	case "<|channel>thought":
		return "<channel|>"
	case "<channel>thought":
		return "<channel>"
	default:
		name := strings.Trim(open, "<|>")
		if strings.HasPrefix(open, "<|") {
			return "<" + name + "|>"
		}
		return "</" + name + ">"
	}
}

// An orphan reasoning closer can follow a prompt-injected opening token.
// Drop the prefix including any tentative calls, never just its text.
func (t *wireTurn) closeHidden(raw string) int {
	if n := len(t.hidden); n != 0 {
		close := t.hidden[n-1]
		if strings.HasPrefix(raw, close) {
			t.hidden = t.hidden[:n-1]
			return len(close)
		}
		return 0
	}
	for _, close := range []string{"</think>", "<channel|>", "<channel>"} {
		if strings.HasPrefix(raw, close) {
			if close == "<channel>" && gemmaChannel.MatchString(raw) {
				return 0 // This is a decoded visible channel opener.
			}
			if close == "</think>" || !t.gemmaVisible {
				*t = wireTurn{}
			}
			t.gemmaVisible = false
			return len(close)
		}
	}
	return 0
}

func partialCallTag(raw string) bool {
	for _, tag := range []string{jsonOpen, jsonClose, phiOpen, "<tool_call|>", lfmOpen, lfmClose,
		"<function=", "</function>", "<parameter=", "</parameter>", "<arg_key>", "</arg_key>",
		"<arg_value>", "</arg_value>", mistralOpen, mistralArgs, mistralClose,
		harmonyMessage, harmonyCallEnd, "<|start|>", "<|channel|>"} {
		if len(raw) >= 4 && strings.HasPrefix(tag, raw) {
			return true
		}
	}
	return false
}

func harmonyFrameStart(raw string) bool {
	return strings.HasPrefix(raw, "<|start|>") || strings.HasPrefix(raw, "<|channel|>") ||
		strings.HasPrefix(raw, harmonyMessage)
}

// resumeAfterBrokenJSON finds a protocol boundary in an unclosed non-call
// span. Bare JSON nested in that span stays prose, and quoted markers stay
// literal. Only explicit framing can regain control of the turn scanner.
func resumeAfterBrokenJSON(raw string) string {
	for raw = raw[1:]; raw != ""; {
		loc := wireToken.FindStringIndex(raw)
		if loc == nil {
			break
		}
		raw = raw[loc[0]:]
		switch raw[0] {
		case '"':
			end, _ := quotedEnd(raw)
			raw = raw[end:]
			continue
		case '<':
			if harmonyFrameStart(raw) || hiddenOpen.MatchString(raw) || turnBoundary.MatchString(raw) ||
				strings.HasPrefix(raw, "</think>") || strings.HasPrefix(raw, "<channel|>") ||
				strings.HasPrefix(raw, "<channel>") ||
				(!qwenProseTag.MatchString(raw) && (unmatchedCallTag.MatchString(raw) || partialCallTag(raw))) {
				return raw
			}
		case '[':
			if unmatchedCallTag.MatchString(raw) || partialCallTag(raw) {
				return raw
			}
		case '{':
			// A bare object is not an explicit protocol boundary.
		default:
			return raw // Named Gemma, GLM, or Harmony shorthand.
		}
		raw = raw[1:]
	}
	return ""
}

// On error, a nonempty tail is safe to resume at only if the end of the
// entire frame is known independently of the error.
func readCallSpan(raw, token string) (wireFamily, []wireToolCall, string, error) {
	switch {
	case strings.HasPrefix(raw, jsonOpen), strings.HasPrefix(raw, phiOpen):
		open := jsonOpen
		if strings.HasPrefix(raw, phiOpen) {
			open = phiOpen
		}
		body := strings.TrimSpace(raw[len(open):])
		if strings.HasPrefix(body, "<function=") {
			calls, tail, err := parseQwenEnvelope(body, true)
			return wireQwen, calls, tail, err
		}
		if body, ok := strings.CutPrefix(body, "call:"); open == phiOpen && ok {
			call, tail, err := parseGemmaCall(body, true)
			if err != nil {
				return wireGemma, nil, "", err
			}
			return wireGemma, []wireToolCall{call}, tail, err
		}
		return jsonCallSpan(wireJSON, body, "", jsonClose)
	case strings.HasPrefix(raw, "<function="):
		calls, tail, err := parseQwenEnvelope(raw, false)
		return wireQwen, calls, tail, err
	case strings.HasPrefix(token, "call:"):
		call, tail, err := parseGemmaCall(raw[len("call:"):], false)
		if err != nil {
			return wireGemma, nil, "", err
		}
		return wireGemma, []wireToolCall{call}, tail, err
	case strings.HasSuffix(token, glmKeyOpen):
		call, tail, err := parseGLMCall(raw)
		if err != nil {
			return wireGLM, nil, "", err
		}
		return wireGLM, []wireToolCall{call}, tail, err
	case strings.HasPrefix(raw, lfmOpen):
		body := strings.TrimSpace(raw[len(lfmOpen):])
		if lfmPythonStart.MatchString(body) {
			calls, tail, err := parseLFMPython(body)
			if err == nil {
				tail, err = requireWireClose(tail, lfmClose)
			}
			if err != nil {
				return wireLFMPython, nil, "", err
			}
			return wireLFMPython, calls, tail, err
		}
		return jsonCallSpan(wireLFMJSON, body, "", lfmClose)
	case strings.HasPrefix(raw, mistralOpen):
		body := strings.TrimSpace(raw[len(mistralOpen):])
		name := wireNamePrefix.FindString(body)
		args, ok := strings.CutPrefix(strings.TrimSpace(body[len(name):]), mistralArgs)
		if name == "" || !ok {
			return wireMistral, nil, "", fmt.Errorf("expected [TOOL_CALLS]name[ARGS] for Mistral")
		}
		kind, calls, tail, err := jsonCallSpan(wireMistral, args, name, "")
		if after, ok := strings.CutPrefix(strings.TrimSpace(tail), mistralClose); ok {
			tail = after
		}
		return kind, calls, tail, err
	case strings.HasPrefix(token, "."):
		name := strings.TrimSpace(token[1 : len(token)-len("<|message|>")])
		kind, calls, tail, err := jsonCallSpan(wireHarmony, raw[len(token):], name, "")
		if after, ok := strings.CutPrefix(strings.TrimSpace(tail), harmonyCallEnd); ok {
			tail = after
		}
		return kind, calls, tail, err
	case len(token) > 1 && (token[0] == '{' || token[0] == '['):
		end, candidate, complete := inlineJSONSpan(raw)
		if !candidate {
			if !complete {
				return "", nil, resumeAfterBrokenJSON(raw), nil
			}
			return "", nil, raw[end:], nil
		}
		kind, calls, tail, err := jsonCallSpan(wireJSON, raw[:end], "", "")
		if err == nil && strings.TrimSpace(tail) != "" {
			err = fmt.Errorf("extra text inside inline JSON call")
		}
		return kind, calls, raw[end:], err
	default:
		return "", nil, raw, nil
	}
}

func jsonCallSpan(kind wireFamily, raw, name, close string) (wireFamily, []wireToolCall, string, error) {
	value, tail, err := splitWireJSON(raw)
	if err != nil {
		tail = ""
		body := strings.TrimSpace(raw)
		if body != "" && (body[0] == '{' || body[0] == '[') {
			if end, _, complete := inlineJSONSpan(body); complete {
				if close == "" {
					tail = body[end:]
				} else if after, closeErr := requireWireClose(body[end:], close); closeErr == nil {
					tail = after
				}
			}
		}
		return kind, nil, tail, fmt.Errorf("invalid %s tool call body: %w", kind, err)
	}
	if close != "" {
		tail, err = requireWireClose(tail, close)
		if err != nil {
			return kind, nil, "", err
		}
	}
	if name != "" {
		object, ok := value.(map[string]any)
		if !ok {
			return kind, nil, tail, fmt.Errorf("%s tool arguments must be a JSON object", kind)
		}
		return kind, []wireToolCall{{name: name, args: object}}, tail, nil
	}
	if list, ok := value.([]any); kind == wireLFMJSON && ok && len(list) > 0 {
		calls := make([]wireToolCall, 0, len(list))
		for _, item := range list {
			call, err := jsonCallObject(item)
			if err != nil {
				return kind, nil, tail, err
			}
			calls = append(calls, call)
		}
		return kind, calls, tail, nil
	}
	call, err := jsonCallObject(value)
	return kind, []wireToolCall{call}, tail, err
}

func requireWireClose(raw, close string) (string, error) {
	tail, ok := strings.CutPrefix(strings.TrimSpace(raw), close)
	if !ok {
		return raw, fmt.Errorf("missing %s after tool call", close)
	}
	return tail, nil
}
