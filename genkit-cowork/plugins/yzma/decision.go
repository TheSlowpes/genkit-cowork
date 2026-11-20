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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/hybridgroup/yzma/exp/decide"
	"github.com/invopop/jsonschema"
)

// ChoiceCriteria is a string type whose zero value describes the options.
type ChoiceCriteria[T ~string] interface {
	~string
	Criteria() map[T]string
}

// Choice asks the model to select one option. Confidence is Yzma's readout
// confidence, which is not necessarily the winning option's probability.
type Choice[T ChoiceCriteria[T]] struct {
	Choice        T             `json:"choice"`
	Probabilities map[T]float64 `json:"probabilities"`
	Confidence    float64       `json:"confidence"`
}

// JSONSchemaExtend adds the choice criteria to Genkit's inferred output schema.
func (Choice[T]) JSONSchemaExtend(s *jsonschema.Schema) {
	var t T
	criteria := make(map[string]string)
	for key, text := range t.Criteria() {
		criteria[string(key)] = text
	}
	s.Extras = map[string]any{decisionSchemaKey: questionSpec{Type: decide.TypeChoice, Criteria: criteria}}
}

// ScoreLevels describes the ordered levels of a rubric, lowest first.
// Levels is called on the zero value.
type ScoreLevels interface {
	Levels() []string
}

// Score asks for a rating on a rubric. Score is the mean level, indexed from
// zero. Probabilities and Legend both follow the rubric's order.
type Score[L ScoreLevels] struct {
	Score         float64   `json:"score"`
	Probabilities []float64 `json:"probabilities"`
	Confidence    float64   `json:"confidence"`
	Legend        []string  `json:"legend"`
}

// JSONSchemaExtend adds the rubric levels to Genkit's inferred output schema.
func (Score[L]) JSONSchemaExtend(s *jsonschema.Schema) {
	var levels L
	s.Extras = map[string]any{decisionSchemaKey: questionSpec{Type: decide.TypeScore, Levels: levels.Levels()}}
}

// Noul asks a yes/no question. Probability is the probability of yes.
type Noul struct {
	Probability float64 `json:"probability"`
}

// JSONSchemaExtend marks the field as a yes/no question in the output schema.
func (Noul) JSONSchemaExtend(s *jsonschema.Schema) {
	s.Extras = map[string]any{decisionSchemaKey: questionSpec{Type: decide.TypeNoul}}
}

const decisionSchemaKey = "x-yzma-decision"

type questionSpec struct {
	Type     decide.Type       `json:"type"`
	Criteria map[string]string `json:"criteria,omitempty"`
	Levels   []string          `json:"levels,omitempty"`
}

type decisionState struct {
	value        any // String or a json.RawMessage containing an object or array.
	instructions string
}

func readDecisionState(messages []*ai.Message) (decisionState, error) {
	var state decisionState
	for _, msg := range messages {
		if msg == nil || len(msg.Content) != 1 || msg.Content[0] == nil {
			return state, errors.New("decision models require exactly one part per message")
		}
		part := msg.Content[0]
		if strings.TrimSpace(part.Text) == "" || strings.ContainsRune(part.Text, '\x00') {
			return state, errors.New("decision messages require non-empty content without NUL bytes")
		}
		switch {
		case msg.Role == ai.RoleSystem && part.Kind == ai.PartText && state.value == nil && state.instructions == "":
			state.instructions = part.Text
		case msg.Role == ai.RoleUser && state.value == nil:
			switch part.Kind {
			case ai.PartText:
				state.value = part.Text
			case ai.PartData:
				raw := bytes.TrimSpace([]byte(part.Text))
				if !json.Valid(raw) || (raw[0] != '{' && raw[0] != '[') {
					return state, errors.New("decision data part must contain a JSON object or array")
				}
				state.value = json.RawMessage(raw)
			default:
				return state, errors.New("decision state must be text or a JSON data part")
			}
		default:
			return state, errors.New("decision models accept one user message after optional system instructions")
		}
	}
	if state.value == nil {
		return state, errors.New("decision models require a user message for state")
	}
	return state, nil
}

func decisionQuestions(out *ai.ModelOutputConfig, system string) ([]string, []decide.Question, error) {
	if out == nil || out.Format != ai.OutputFormatJSON ||
		(out.ContentType != "" && out.ContentType != "application/json") {
		return nil, nil, errors.New("decision models require a typed JSON output (use GenerateData)")
	}
	properties := schemaMap(out.Schema["properties"])
	if out.Schema["type"] != "object" || len(properties) == 0 {
		return nil, nil, errors.New("decision output must be a non-empty struct of Choice, Score, or Noul fields")
	}
	keys := slices.Sorted(maps.Keys(properties))
	qs := make([]decide.Question, len(keys))
	for i, key := range keys {
		q, err := decisionQuestion(schemaMap(properties[key]), system)
		if err != nil {
			return nil, nil, fmt.Errorf("decision field %q: %w", key, err)
		}
		qs[i] = q
	}
	return keys, qs, nil
}

func decisionQuestion(field map[string]any, system string) (decide.Question, error) {
	var q decide.Question
	specData, err := json.Marshal(field[decisionSchemaKey])
	if err != nil {
		return q, err
	}
	var spec questionSpec
	if err := json.Unmarshal(specData, &spec); err != nil {
		return q, err
	}
	text, _ := field["description"].(string)
	if strings.TrimSpace(text) == "" {
		return q, errors.New("add question instructions with a jsonschema_description tag")
	}
	if system != "" {
		text = system + "\n\n" + text
	}
	switch spec.Type {
	case decide.TypeChoice:
		if len(spec.Criteria) < 2 {
			return q, errors.New("choice needs at least two options")
		}
		options := make([]decide.Option, 0, len(spec.Criteria))
		for _, name := range slices.Sorted(maps.Keys(spec.Criteria)) {
			if strings.TrimSpace(name) == "" {
				return q, errors.New("choice option names must not be blank")
			}
			options = append(options, decide.Option{Name: name, Description: spec.Criteria[name]})
		}
		q = decide.ChoiceDesc(text, options...)
	case decide.TypeScore:
		for _, level := range spec.Levels {
			if strings.TrimSpace(level) == "" {
				return q, errors.New("score levels must not be blank")
			}
		}
		q = decide.Score(text, spec.Levels...)
	case decide.TypeNoul:
		q = decide.Noul(text, "", "")
	default:
		return q, errors.New("use Choice, Score, or Noul")
	}
	_, err = q.Names()
	return q, err
}

func (g *generator) generateDecision(ctx context.Context, input *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	if cb != nil {
		return nil, errors.New("yzma: decision models do not support streaming")
	}
	if len(input.Tools) != 0 || (input.ToolChoice != "" && input.ToolChoice != ai.ToolChoiceAuto) {
		return nil, errors.New("yzma: decision models do not support tools")
	}
	if len(input.Docs) != 0 {
		return nil, errors.New("yzma: decision models do not support attached documents; include evidence in the state")
	}
	if input.Config != nil {
		cfg, err := json.Marshal(input.Config)
		if err != nil || (string(cfg) != "{}" && string(cfg) != "null") {
			return nil, errors.New("yzma: decision models do not support per-request config")
		}
	}
	state, err := readDecisionState(input.Messages)
	if err != nil {
		return nil, err
	}
	keys, questions, err := decisionQuestions(input.Output, state.instructions)
	if err != nil {
		return nil, err
	}
	if err := g.acquire(ctx); err != nil {
		return nil, err
	}
	defer g.release()
	results, err := g.decision.DecideMany(ctx, state.value, questions)
	if err != nil {
		return nil, fmt.Errorf("decide with model %q: %w", g.model.Name, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(results) != len(keys) {
		return nil, fmt.Errorf("yzma: decision model returned %d answers for %d questions", len(results), len(keys))
	}
	answers := make(map[string]any, len(keys)) // The value type varies by question.
	var tokens int
	for i, key := range keys {
		answer, err := decisionAnswer(questions[i], results[i])
		if err != nil {
			return nil, fmt.Errorf("decision field %q: %w", key, err)
		}
		answers[key] = answer
		tokens += results[i].InputTokens
	}
	data, err := json.Marshal(answers)
	if err != nil {
		return nil, err
	}
	// Genkit v1.9 extracts Markdown fences even from JSON strings.
	text := strings.ReplaceAll(string(data), "`", `\u0060`)
	return &ai.ModelResponse{
		Message: ai.NewModelMessage(ai.NewTextPart(text)), Request: input,
		FinishReason: ai.FinishReasonStop,
		Usage:        &ai.GenerationUsage{InputTokens: tokens, TotalTokens: tokens},
	}, nil
}

func decisionAnswer(q decide.Question, result *decide.Result) (any, error) {
	names, err := q.Names()
	if err != nil {
		return nil, err
	}
	if result == nil || !slices.Equal(names, result.Options) || len(names) != len(result.Probabilities) ||
		!slices.Contains(names, result.Answer) || !validProbability(result.Confidence) {
		return nil, errors.New("invalid decision result")
	}
	var total, mean float64
	for i, p := range result.Probabilities {
		if !validProbability(p) {
			return nil, fmt.Errorf("invalid probability for %q", names[i])
		}
		total += p
		mean += float64(i) * p
	}
	if math.Abs(total-1) > 1e-6 {
		return nil, errors.New("decision probabilities do not sum to one")
	}
	switch q.Type {
	case decide.TypeChoice:
		p := make(map[string]float64, len(names))
		for i, name := range names {
			p[name] = result.Probabilities[i]
		}
		return choiceAnswer{Choice: result.Answer, Probabilities: p, Confidence: result.Confidence}, nil
	case decide.TypeScore:
		legend := make([]string, len(q.Options))
		for i, option := range q.Options {
			legend[i] = option.Description
		}
		return Score[ScoreLevels]{Score: mean, Probabilities: result.Probabilities, Confidence: result.Confidence, Legend: legend}, nil
	default:
		return Noul{Probability: result.Probability("true")}, nil
	}
}

type choiceAnswer struct {
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
}

func validProbability(p float64) bool { return p >= 0 && p <= 1 }
