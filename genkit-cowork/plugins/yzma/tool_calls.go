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
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"regexp"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/google/uuid"
	"github.com/hybridgroup/yzma/pkg/message"
	"github.com/xeipuuv/gojsonschema"
)

// templateMessage preserves typed Genkit arguments, call IDs, and tool outputs.
// Yzma's message.Tool stores arguments as strings, which would corrupt their
// types when replaying calls into a Jinja template.
type templateMessage struct {
	role string
	data map[string]any // Jinja and Genkit tool schemas require dynamic JSON values.
}

func (m templateMessage) GetRole() string { return m.role }
func (m templateMessage) GetContent() map[string]any {
	out := make(map[string]any, len(m.data))
	for key, value := range m.data {
		out[key] = value
	}
	return out
}

func generationMessages(input *ai.ModelRequest, enabled bool) ([]message.Message, error) {
	if !enabled {
		if input != nil && (len(input.Tools) > 0 || input.ToolChoice == ai.ToolChoiceRequired) {
			return nil, fmt.Errorf("yzma: enable ModelDefinition.ToolCalling for a tool-capable chat model")
		}
		chat, err := textMessages(input)
		out := make([]message.Message, len(chat))
		for i := range chat {
			out[i] = chat[i]
		}
		return out, err
	}
	if input == nil || len(input.Messages) == 0 {
		return nil, fmt.Errorf("yzma: at least one message is required")
	}
	if input.ToolChoice != "" && input.ToolChoice != ai.ToolChoiceAuto &&
		input.ToolChoice != ai.ToolChoiceNone && input.ToolChoice != ai.ToolChoiceRequired {
		return nil, fmt.Errorf("yzma: unsupported tool choice %q", input.ToolChoice)
	}
	if input.ToolChoice == ai.ToolChoiceRequired && len(input.Tools) == 0 {
		return nil, fmt.Errorf("yzma: required tool choice needs at least one tool")
	}
	// Reuse text-only request validation without discarding tool history.
	validation := *input
	validation.Messages = []*ai.Message{ai.NewUserTextMessage("validate")}
	validation.Tools, validation.ToolChoice = nil, ""
	if _, err := textMessages(&validation); err != nil {
		return nil, err
	}
	var out []message.Message
	calls, answered := map[string]string{}, map[string]bool{}
	for _, msg := range input.Messages {
		if msg == nil {
			return nil, fmt.Errorf("yzma: nil message")
		}
		if msg.Role == ai.RoleTool {
			for _, part := range msg.Content {
				if part == nil || !part.IsToolResponse() || part.ToolResponse == nil {
					return nil, fmt.Errorf("yzma: tool messages must contain tool responses")
				}
				response := part.ToolResponse
				if response.Ref == "" || calls[response.Ref] != response.Name || answered[response.Ref] {
					return nil, fmt.Errorf("yzma: unmatched or duplicate tool response %q", response.Ref)
				}
				body := response.Output
				if len(response.Content) > 0 {
					var text strings.Builder
					for _, extra := range response.Content {
						if !extra.IsText() {
							return nil, fmt.Errorf("yzma: tool response media is not supported")
						}
						text.WriteString(extra.Text)
					}
					body = map[string]any{"output": body, "content": text.String()}
				}
				out = append(out, templateMessage{"tool", map[string]any{
					"name": response.Name, "tool_call_id": response.Ref, "content": body,
				}})
				answered[response.Ref] = true
			}
			continue
		}
		if msg.Role != ai.RoleSystem && msg.Role != ai.RoleUser && msg.Role != ai.RoleModel {
			return nil, fmt.Errorf("yzma: unsupported message role %q", msg.Role)
		}
		var text strings.Builder
		var requests []map[string]any
		for _, part := range msg.Content {
			switch {
			case part == nil:
				return nil, fmt.Errorf("yzma: nil message part")
			case part.IsToolRequest() && part.ToolRequest != nil:
				request := part.ToolRequest
				if msg.Role != ai.RoleModel || request.Ref == "" || request.Name == "" || calls[request.Ref] != "" {
					return nil, fmt.Errorf("yzma: invalid or duplicate tool request %q", request.Ref)
				}
				args, err := jsonObject(request.Input)
				if err != nil {
					return nil, fmt.Errorf("tool request %q: %w", request.Ref, err)
				}
				requests = append(requests, map[string]any{
					"id": request.Ref, "type": "function",
					"function": map[string]any{"name": request.Name, "arguments": args},
				})
				calls[request.Ref] = request.Name
			case part.IsText():
				text.WriteString(part.Text)
			default:
				return nil, fmt.Errorf("yzma: unsupported message content")
			}
		}
		content := text.String()
		if strings.ContainsRune(content, '\x00') || (content == "" && len(requests) == 0) {
			return nil, fmt.Errorf("yzma: message must contain text or tool calls, without NUL bytes")
		}
		data := map[string]any{"content": content}
		if len(requests) > 0 {
			data["tool_calls"] = requests
		}
		out = append(out, templateMessage{roleMapping[msg.Role], data})
	}
	return out, nil
}

func jsonObject(value any) (map[string]any, error) {
	if value == nil {
		return map[string]any{}, nil
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var result map[string]any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&result); err != nil || result == nil {
		return nil, fmt.Errorf("tool arguments must be a JSON object")
	}
	return result, nil
}

// Yzma v1.33 skips Gemma calls with empty arguments. Normalize complete empty
// Gemma envelopes with a temporary argument so its parser also handles those.
var emptyGemmaCall = regexp.MustCompile(`<\|tool_call>\s*call:([a-zA-Z_][\w./-]*)\{\s*\}<tool_call\|>`)
var bareEmptyGemmaCall = regexp.MustCompile(`^\s*call:([a-zA-Z_][\w./-]*)\{\s*\}\s*$`)

func parsedToolParts(result TextGenerationResult, definitions []*ai.ToolDefinition, choice ai.ToolChoice) ([]*ai.Part, error) {
	wireCalls, text, err := parseWireToolCalls(result.Text)
	if err != nil {
		return nil, err
	}
	if len(wireCalls) > 0 {
		if result.Limited {
			return nil, fmt.Errorf("yzma: incomplete tool call; increase the output/context limit")
		}
		return validatedToolParts(text, wireCalls, definitions, choice)
	}
	if wireCalls != nil {
		return toolTextPart(text, choice)
	}

	raw, sentinel := result.Text, ""
	if emptyGemmaCall.MatchString(raw) || bareEmptyGemmaCall.MatchString(raw) {
		sentinel = "__yzma_empty_" + strings.ReplaceAll(uuid.NewString(), "-", "")
		raw = emptyGemmaCall.ReplaceAllString(raw, `<|tool_call>call:$1{`+sentinel+`:true}<tool_call|>`)
		raw = bareEmptyGemmaCall.ReplaceAllString(raw, `call:$1{`+sentinel+`:true}`)
	}
	calls := message.ParseToolCalls(raw)
	if len(calls) == 0 {
		if strings.Contains(raw, "<tool_call") || strings.Contains(raw, "<|tool_call") ||
			strings.Contains(raw, "<function=") || strings.Contains(raw, "<arg_key>") {
			return nil, fmt.Errorf("yzma: malformed or incomplete tool call")
		}
		return toolTextPart(message.StripMarkup(result.Text), choice)
	}
	// GPT calls are executable only when the lossless parser recognized them.
	if message.DetectFormat(raw) == message.FormatGPT {
		return nil, fmt.Errorf("yzma: unsupported GPT tool call syntax")
	}
	// Yzma repairs truncated JSON. Never execute a repaired, incomplete call.
	if result.Limited || !completeToolOutput(raw) {
		return nil, fmt.Errorf("yzma: incomplete tool call; increase the output/context limit")
	}
	for _, call := range calls {
		if sentinel != "" {
			delete(call.Function.Arguments, sentinel)
		}
		wireCalls = append(wireCalls, wireToolCall{name: call.Function.Name, untyped: call.Function.Arguments})
	}
	return validatedToolParts(message.StripMarkup(raw), wireCalls, definitions, choice)
}

func toolTextPart(text string, choice ai.ToolChoice) ([]*ai.Part, error) {
	if choice == ai.ToolChoiceRequired {
		return nil, fmt.Errorf("yzma: the model did not produce a required tool call")
	}
	return []*ai.Part{ai.NewTextPart(text)}, nil
}

func validatedToolParts(text string, calls []wireToolCall, definitions []*ai.ToolDefinition, choice ai.ToolChoice) ([]*ai.Part, error) {
	allowed := make(map[string]*ai.ToolDefinition, len(definitions))
	for _, tool := range definitions {
		allowed[tool.Name] = tool
	}
	var parts []*ai.Part
	if text != "" {
		parts = append(parts, ai.NewTextPart(text))
	}
	for _, call := range calls {
		tool := allowed[call.name]
		if tool == nil || choice == ai.ToolChoiceNone {
			return nil, fmt.Errorf("yzma: model requested unoffered tool %q", call.name)
		}
		args := call.args
		if args == nil {
			var err error
			args, err = untypedToolArguments(call.untyped, tool.InputSchema)
			if err != nil {
				return nil, fmt.Errorf("tool %q: %w", tool.Name, err)
			}
		}
		if err := validateToolArguments(args, tool.InputSchema); err != nil {
			return nil, fmt.Errorf("tool %q: %w", tool.Name, err)
		}
		if err := validateToolNumberTransport(args); err != nil {
			return nil, fmt.Errorf("tool %q: %w", tool.Name, err)
		}
		parts = append(parts, ai.NewToolRequestPart(&ai.ToolRequest{
			Name: tool.Name, Ref: uuid.NewString(), Input: args,
		}))
	}
	return parts, nil
}

// validateToolNumberTransport rejects numbers that Genkit v1.9's RunJSON would
// change when it decodes through float64 before normalizing the tool input.
func validateToolNumberTransport(value any) error {
	switch v := value.(type) {
	case map[string]any:
		for key, child := range v {
			if err := validateToolNumberTransport(child); err != nil {
				return fmt.Errorf("%s: %w", key, err)
			}
		}
	case []any:
		for i, child := range v {
			if err := validateToolNumberTransport(child); err != nil {
				return fmt.Errorf("[%d]: %w", i, err)
			}
		}
	case json.Number:
		f, err := v.Float64()
		if err == nil {
			original, ok := new(big.Rat).SetString(v.String())
			if ok {
				// Integers may be converted directly to int64 by Genkit. The
				// shortest decimal spelling of f alone cannot prove exactness.
				if !original.IsInt() || original.Cmp(new(big.Rat).SetFloat64(f)) == 0 {
					encoded, _ := json.Marshal(f) // Float64 succeeded, so f is finite.
					reencoded, ok := new(big.Rat).SetString(string(encoded))
					if ok && original.Cmp(reencoded) == 0 {
						return nil
					}
				}
			}
		}
		return fmt.Errorf("number %s would lose precision in Genkit tool dispatch", v)
	}
	return nil
}

// completeToolOutput checks structural completeness independently of the
// forgiving family parsers. Quoted braces and Gemma string delimiters are literal.
func completeToolOutput(raw string) bool {
	_, ok := scanToolStructure(raw, false)
	return ok
}

// scanToolStructure can also stop at the end of one argument object to keep
// literal tag text inside Gemma arguments out of the envelope scanner.
func scanToolStructure(raw string, oneObject bool) (int, bool) {
	var stack []byte
	quote := ""
	for i := 0; i < len(raw); {
		if quote != "" {
			if strings.HasPrefix(raw[i:], quote) {
				i += len(quote)
				quote = ""
			} else if raw[i] == '\\' && len(quote) == 1 {
				i += min(2, len(raw)-i)
			} else {
				i++
			}
			continue
		}
		if len(stack) > 0 {
			for _, delimiter := range []string{`<|"|>`, `<">`, `<|>`, `"`, `'`} {
				if strings.HasPrefix(raw[i:], delimiter) {
					quote = delimiter
					i += len(delimiter)
					break
				}
			}
			if quote != "" {
				continue
			}
		}
		switch raw[i] {
		case '{', '[':
			stack = append(stack, raw[i])
		case '}', ']':
			if len(stack) == 0 || (raw[i] == '}' && stack[len(stack)-1] != '{') ||
				(raw[i] == ']' && stack[len(stack)-1] != '[') {
				return 0, false
			}
			stack = stack[:len(stack)-1]
			if oneObject && len(stack) == 0 {
				return i + 1, true
			}
		}
		i++
	}
	return len(raw), !oneObject && len(stack) == 0 && quote == ""
}

// untypedToolArguments interprets non-JSON family syntax using the offered
// schema. JSON envelopes never take this path.
func untypedToolArguments(raw map[string]string, schema map[string]any) (map[string]any, error) {
	if schema == nil {
		schema = map[string]any{"type": "object", "additionalProperties": false}
	}
	root, err := dereferenceSchema(schema, schema)
	if err != nil {
		return nil, err
	}
	properties, _ := root["properties"].(map[string]any)
	args := make(map[string]any, len(raw))
	for key, value := range raw {
		property, _ := properties[key].(map[string]any)
		if property == nil {
			property, _ = root["additionalProperties"].(map[string]any)
		}
		property, err = dereferenceSchema(property, schema)
		if err != nil {
			return nil, err
		}
		kind, _ := property["type"].(string)
		switch kind {
		case "string":
			args[key] = value
		case "integer", "number", "boolean", "array", "object", "null":
			decoder := json.NewDecoder(bytes.NewBufferString(value))
			decoder.UseNumber()
			var decoded any
			if !json.Valid([]byte(value)) {
				return nil, fmt.Errorf("argument %q is not valid JSON for type %s", key, kind)
			}
			if err := decoder.Decode(&decoded); err != nil {
				return nil, err
			}
			if kind == "integer" {
				if n, ok := decoded.(json.Number); ok {
					integer, err := n.Float64()
					if err != nil || math.Abs(integer) >= 1<<53 {
						return nil, fmt.Errorf("argument %q exceeds the parser's lossless integer range", key)
					}
				}
			}
			args[key] = decoded
		default:
			return nil, fmt.Errorf("argument %q requires an unambiguous schema type; got %v", key, property["type"])
		}
	}
	return args, nil
}

func validateToolArguments(args, schema map[string]any) error {
	if schema == nil {
		schema = map[string]any{"type": "object", "additionalProperties": false}
	}
	validation, err := gojsonschema.Validate(gojsonschema.NewGoLoader(schema), gojsonschema.NewGoLoader(args))
	if err != nil {
		return err
	}
	if !validation.Valid() {
		return fmt.Errorf("arguments do not match the tool schema: %v", validation.Errors())
	}
	return nil
}

func dereferenceSchema(schema, root map[string]any) (map[string]any, error) {
	seen := map[string]bool{}
	for schema != nil {
		ref, ok := schema["$ref"].(string)
		if !ok {
			return schema, nil
		}
		if !strings.HasPrefix(ref, "#/") || seen[ref] {
			return nil, fmt.Errorf("unsupported or cyclic schema reference %q", ref)
		}
		seen[ref] = true
		var value any = root
		for _, key := range strings.Split(ref[2:], "/") {
			key = strings.ReplaceAll(strings.ReplaceAll(key, "~1", "/"), "~0", "~")
			object, _ := value.(map[string]any)
			value = object[key]
		}
		schema, _ = value.(map[string]any)
		if schema == nil {
			return nil, fmt.Errorf("unresolved schema reference %q", ref)
		}
	}
	return nil, nil
}

// toolTemplateSchema expands local references for templates that inspect
// properties directly rather than implementing a JSON Schema resolver.
// Validation still uses the original, unmodified schema.
func toolTemplateSchema(schema, root map[string]any, depth int) (map[string]any, error) {
	if depth > 32 {
		return nil, fmt.Errorf("recursive or excessively nested tool schemas are not supported by chat templates")
	}
	resolved, err := dereferenceSchema(schema, root)
	if err != nil {
		return nil, err
	}
	out := make(map[string]any, len(resolved))
	for key, value := range resolved {
		switch key {
		case "$defs", "definitions":
			continue
		case "properties":
			properties, _ := value.(map[string]any)
			expanded := make(map[string]any, len(properties))
			for name, child := range properties {
				object, ok := child.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("property %q requires an object schema", name)
				}
				expanded[name], err = toolTemplateSchema(object, root, depth+1)
				if err != nil {
					return nil, err
				}
			}
			out[key] = expanded
		case "items", "additionalProperties":
			if object, ok := value.(map[string]any); ok {
				out[key], err = toolTemplateSchema(object, root, depth+1)
				if err != nil {
					return nil, err
				}
			} else {
				out[key] = value
			}
		default:
			out[key] = value
		}
	}
	return out, nil
}
