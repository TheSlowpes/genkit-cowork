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
	"maps"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/firebase/genkit/go/ai"
	"github.com/xeipuuv/gojsonschema"
)

type outputConstraint struct {
	grammar string
	schema  *gojsonschema.Schema
}

func prepareConstraint(input *ai.ModelRequest) (*outputConstraint, error) {
	if input == nil || input.Output == nil {
		return nil, nil
	}
	out := input.Output
	if !out.Constrained && out.Schema == nil && out.Format != ai.OutputFormatJSON && out.ContentType != "application/json" {
		return nil, nil
	}
	// Genkit's no-tools negotiation can clear Constrained and Schema, but it
	// retains the JSON format and tools. Reject that downgrade too.
	if len(input.Tools) != 0 || (input.ToolChoice != "" && input.ToolChoice != ai.ToolChoiceAuto && input.ToolChoice != ai.ToolChoiceNone) {
		return nil, fmt.Errorf("yzma: constrained JSON output cannot be combined with tools")
	}
	if !out.Constrained || len(out.Schema) == 0 {
		return nil, fmt.Errorf("yzma: JSON output requires native constraints and a schema")
	}
	if (out.Format != "" && out.Format != ai.OutputFormatJSON) ||
		(out.ContentType != "" && out.ContentType != "application/json") {
		return nil, fmt.Errorf("yzma: constrained output requires JSON format")
	}
	return compileConstraint(out.Schema)
}

func compileConstraint(schema map[string]any) (*outputConstraint, error) {
	fail := func(err error) (*outputConstraint, error) {
		return nil, fmt.Errorf("yzma: unsupported output schema: %w", err)
	}
	data, err := json.Marshal(schema)
	if err != nil {
		return fail(err)
	}
	if len(data) > 64*1024 {
		return fail(fmt.Errorf("schema exceeds 64 KiB"))
	}
	// Normalize containers from direct requests to the same shapes Genkit
	// uses for inferred schemas. The whitelist below runs before validation
	// library compilation so no refs or schema dialects can be loaded.
	var normalized map[string]any
	if err := json.Unmarshal(data, &normalized); err != nil {
		return fail(err)
	}
	if normalized["type"] != "object" {
		return fail(fmt.Errorf("root must be a closed object"))
	}
	var builder grammarBuilder
	rule, err := builder.node(normalized, 1)
	if err != nil {
		return fail(err)
	}
	validator, err := gojsonschema.NewSchema(gojsonschema.NewBytesLoader(data))
	if err != nil {
		return fail(err)
	}
	return &outputConstraint{
		grammar: "root ::= " + rule + "\n" + jsonGrammar + strings.Join(builder.rules, "\n") + "\n",
		schema:  validator,
	}, nil
}

func (c *outputConstraint) validate(result TextGenerationResult) error {
	if result.Limited {
		return fmt.Errorf("yzma: incomplete constrained output; increase the output or context limit")
	}
	// Validate the raw response before Genkit can extract JSON from a fence
	// or accept a valid prefix followed by other text.
	if !utf8.ValidString(result.Text) || !json.Valid([]byte(result.Text)) {
		return fmt.Errorf("yzma: constrained output must be one complete JSON value")
	}
	validated, err := c.schema.Validate(gojsonschema.NewStringLoader(result.Text))
	if err != nil {
		return fmt.Errorf("yzma: validate constrained output: %w", err)
	}
	if !validated.Valid() {
		return fmt.Errorf("yzma: constrained output does not match the schema: %v", validated.Errors())
	}
	return nil
}

const jsonGrammar = `ws ::= [ \t\n\r]?
jstring ::= "\"" ([^"\\\x00-\x1f] | "\\" ["\\/bfnrt])* "\""
jinteger ::= "-"? ("0" | [1-9] [0-9]{0,14})
jnumber ::= jinteger ("." [0-9]+)? ([eE] [+-]? [0-9]{1,2})?
jboolean ::= "true" | "false"
`

// grammarBuilder only emits rules for recognized constraints. In particular,
// it cannot substitute generic JSON for a schema it doesn't understand.
type grammarBuilder struct {
	nodes int
	rules []string
}

func (b *grammarBuilder) rule(body string) string {
	name := "node-" + strconv.Itoa(len(b.rules))
	b.rules = append(b.rules, name+" ::= "+body)
	return name
}

func (b *grammarBuilder) node(schema map[string]any, depth int) (string, error) {
	b.nodes++
	if b.nodes > 256 || depth > 24 {
		return "", fmt.Errorf("schema exceeds 256 nodes or 24 levels")
	}
	kind, _ := schema["type"].(string)
	for _, key := range slices.Sorted(maps.Keys(schema)) {
		switch {
		case key == "type":
		case key == "title" || key == "description":
			if _, ok := schema[key].(string); !ok {
				return "", fmt.Errorf("%s must be a string", key)
			}
		case kind == "object" && (key == "properties" || key == "required" || key == "additionalProperties"):
		case kind == "array" && key == "items":
		case kind == "string" && key == "enum":
		default:
			return "", fmt.Errorf("keyword %q is not supported for type %q", key, kind)
		}
	}
	switch kind {
	case "object":
		return b.object(schema, depth)
	case "array":
		item, err := b.node(schemaMap(schema["items"]), depth+1)
		if err != nil {
			return "", fmt.Errorf("items: %w", err)
		}
		return b.rule(`"[" ws (` + item + ` ws ("," ws ` + item + ` ws)*)? "]"`), nil
	case "string":
		if values, exists := schema["enum"]; exists {
			return b.stringEnum(values)
		}
		return "jstring", nil
	case "integer", "number", "boolean":
		return "j" + kind, nil
	case "null":
		return `"null"`, nil
	default:
		return "", fmt.Errorf("missing or unsupported type %q", kind)
	}
}

func (b *grammarBuilder) object(schema map[string]any, depth int) (string, error) {
	if schema["additionalProperties"] != false {
		return "", fmt.Errorf("objects require additionalProperties: false")
	}
	properties := schemaMap(schema["properties"])
	if properties == nil {
		return "", fmt.Errorf("objects require a properties map")
	}
	required := make(map[string]bool)
	if value, exists := schema["required"]; exists {
		names, ok := value.([]any)
		if !ok {
			return "", fmt.Errorf("required must be a list of property names")
		}
		for _, value := range names {
			name, ok := value.(string)
			if !ok || properties[name] == nil || required[name] {
				return "", fmt.Errorf("required must list unique declared property names")
			}
			required[name] = true
		}
	}

	// Generate sorted keys once each. For optional keys, track whether a
	// preceding key was emitted so skipping a key never leaves a stray comma.
	first, tail := `""`, `""`
	keys := slices.Sorted(maps.Keys(properties))
	for _, key := range slices.Backward(keys) {
		child, err := b.node(schemaMap(properties[key]), depth+1)
		if err != nil {
			return "", fmt.Errorf("property %q: %w", key, err)
		}
		field := jsonLiteral(key) + ` ws ":" ws ` + child + " ws"
		start := field + " " + tail
		after := `"," ws ` + start
		if !required[key] {
			start += " | " + first
			after += " | " + tail
		}
		first, tail = b.rule(start), b.rule(after)
	}
	return b.rule(`"{" ws ` + first + ` "}"`), nil
}

func (b *grammarBuilder) stringEnum(value any) (string, error) {
	values, ok := value.([]any)
	if !ok || len(values) == 0 {
		return "", fmt.Errorf("enum must be a non-empty list of strings")
	}
	seen := make(map[string]bool, len(values))
	literals := make([]string, 0, len(values))
	for _, value := range values {
		s, ok := value.(string)
		if !ok || seen[s] {
			return "", fmt.Errorf("enum values must be unique strings")
		}
		seen[s] = true
		literals = append(literals, jsonLiteral(s))
	}
	return b.rule(strings.Join(literals, " | ")), nil
}

func schemaMap(value any) map[string]any {
	schema, _ := value.(map[string]any)
	return schema
}

func jsonLiteral(s string) string {
	raw, _ := json.Marshal(s) // Strings always marshal successfully.
	return strconv.Quote(string(raw))
}
