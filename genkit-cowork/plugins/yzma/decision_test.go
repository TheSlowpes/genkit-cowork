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
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
	"github.com/hybridgroup/yzma/exp/decide"
	"github.com/hybridgroup/yzma/pkg/llama"
)

type ticketDepartment string

func (ticketDepartment) Criteria() map[ticketDepartment]string {
	return map[ticketDepartment]string{
		"billing": "Payments, invoices, and refunds", "technical": "Bugs and outages",
	}
}

type ticketFrustration int

func (ticketFrustration) Levels() []string {
	return []string{"Calm", "Concerned but civil", "Very angry"}
}

type ticketDecision struct {
	Department  Choice[ticketDepartment] `json:"team" jsonschema_description:"Which team should handle the ticket?"`
	Frustration Score[ticketFrustration] `json:"frustration" jsonschema_description:"How frustrated is the customer?"`
	Urgent      Noul                     `json:"urgent" jsonschema_description:"Does the customer explicitly request a same-day fix?"`
}

type testDecisionOperator struct {
	testModelOperator
	loadDecision func(context.Context, string, DecisionOptions, ContextOptions) (DecisionModel, error)
}

func (o *testDecisionOperator) LoadDecision(ctx context.Context, path string, opts DecisionOptions, params ContextOptions) (DecisionModel, error) {
	return o.loadDecision(ctx, path, opts, params)
}

type testDecisionModel struct {
	score func(context.Context, any, []decide.Question) ([]*decide.Result, error)
	close func(context.Context) error
}

func (m *testDecisionModel) DecideMany(ctx context.Context, state any, qs []decide.Question) ([]*decide.Result, error) {
	return m.score(ctx, state, qs)
}

func (m *testDecisionModel) Close(ctx context.Context) error { return m.close(ctx) }

type testDecisionVisionOperator struct {
	*testVisionOperator
	reader DecisionModel
}

func (o *testDecisionVisionOperator) LoadDecision(context.Context, string, DecisionOptions, ContextOptions) (DecisionModel, error) {
	return o.reader, nil
}

func TestDecisionLoadWaitsForVisionCompanion(t *testing.T) {
	projectorLoading, allowProjector := make(chan struct{}), make(chan struct{})
	op := &testDecisionVisionOperator{
		testVisionOperator: &testVisionOperator{
			testModelOperator: &testModelOperator{
				load: func(context.Context, ModelOptions) (llama.Model, error) { return 1, nil },
			},
			loadProjector: func(ctx context.Context, _ llama.Model, _ ModelDefinition) (ProjectorHandle, error) {
				close(projectorLoading)
				select {
				case <-allowProjector:
					return 1, nil
				case <-ctx.Done():
					return 0, ctx.Err()
				}
			},
			freeProjector: func(context.Context, ProjectorHandle) error { return nil },
		},
		reader: &testDecisionModel{close: func(context.Context) error { return nil }},
	}
	p, g := initDiscoveryTest(t, t.TempDir(), op)
	vision := testModelDefinition()
	vision.Projector = &ProjectorOptions{GGUFPath: "projector.gguf", Images: true}
	p.DefineModel(g, vision, nil)
	p.DefineModel(g, ModelDefinition{
		Name: "decision", Type: ModelTypeDecision, Loading: ModelOptions{GGUFPath: "decision.gguf"},
		Decision: DecisionOptions{Readout: DecisionReadoutGGUF},
	}, nil)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	visionLoad := make(chan error, 1)
	go func() { visionLoad <- p.Preload(ctx, vision.Name) }()
	select {
	case <-projectorLoading:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	wait, stop := context.WithTimeout(ctx, 30*time.Millisecond)
	defer stop()
	if err := p.Preload(wait, "decision"); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("decision preload while projector is loading = %v, want deadline exceeded", err)
	}
	close(allowProjector)
	if err := <-visionLoad; err != nil {
		t.Fatal(err)
	}
	if err := p.Preload(ctx, "decision"); err != nil {
		t.Fatal(err)
	}
}

func TestGenerateDataDecision(t *testing.T) {
	const system = "State is a support ticket."
	const state = `{"ticket":"Please refund the duplicate charge today."}`
	path := filepath.Join(t.TempDir(), "manual.gguf")
	loads, calls, closes := 0, 0, 0
	operator := &testDecisionOperator{
		loadDecision: func(_ context.Context, gotPath string, o DecisionOptions, _ ContextOptions) (DecisionModel, error) {
			loads++
			if gotPath != path || o.Readout != DecisionReadoutGGUF {
				t.Fatalf("load decision: path=%q options=%+v", gotPath, o)
			}
			return &testDecisionModel{
				score: func(_ context.Context, gotState any, qs []decide.Question) ([]*decide.Result, error) {
					calls++
					if diff := cmp.Diff(json.RawMessage(state), gotState); diff != "" {
						t.Errorf("state (-want +got):\n%s", diff)
					}
					want := []decide.Question{
						decide.Score(system+"\n\nHow frustrated is the customer?", ticketFrustration(0).Levels()...),
						decide.ChoiceDesc(system+"\n\nWhich team should handle the ticket?",
							decide.Option{Name: "billing", Description: "Payments, invoices, and refunds"},
							decide.Option{Name: "technical", Description: "Bugs and outages"}),
						decide.Noul(system+"\n\nDoes the customer explicitly request a same-day fix?", "", ""),
					}
					if diff := cmp.Diff(want, qs, cmp.AllowUnexported(decide.Question{})); diff != "" {
						t.Fatalf("questions (-want +got):\n%s", diff)
					}
					return []*decide.Result{
						{Options: []string{"0", "1", "2"}, Probabilities: []float64{0.125, 0.25, 0.625},
							Answer: "2", Confidence: 0.25, InputTokens: 10},
						{Options: []string{"billing", "technical"}, Probabilities: []float64{0.8, 0.2},
							Answer: "billing", Confidence: 0.6, InputTokens: 10},
						{Options: []string{"false", "true"}, Probabilities: []float64{0.1, 0.9},
							Answer: "true", Confidence: 0.8, InputTokens: 10},
					}, nil
				},
				close: func(context.Context) error { closes++; return nil },
			}, nil
		},
	}
	plugin, g := initDiscoveryTest(t, t.TempDir(), operator)
	plugin.DefineModel(g, ModelDefinition{
		Name: "ticket", Type: ModelTypeDecision, Loading: ModelOptions{GGUFPath: path},
		Decision: DecisionOptions{Readout: DecisionReadoutGGUF},
	}, nil)
	if loads != 0 {
		t.Fatal("registration loaded weights")
	}
	for range 2 {
		got, resp, err := genkit.GenerateData[ticketDecision](t.Context(), g, ai.WithModelName("yzma/ticket"),
			ai.WithSystem(system), ai.WithMessages(ai.NewUserMessage(ai.NewDataPart(state))))
		if err != nil {
			t.Fatal(err)
		}
		want := &ticketDecision{
			Department: Choice[ticketDepartment]{Choice: "billing", Confidence: 0.6,
				Probabilities: map[ticketDepartment]float64{"billing": 0.8, "technical": 0.2}},
			Frustration: Score[ticketFrustration]{Score: 1.5, Confidence: 0.25,
				Probabilities: []float64{0.125, 0.25, 0.625}, Legend: ticketFrustration(0).Levels()},
			Urgent: Noul{Probability: 0.9},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("decision (-want +got):\n%s", diff)
		}
		if resp.Usage.InputTokens != 30 || resp.Usage.OutputTokens != 0 {
			t.Errorf("usage = %+v, want 30 input and zero output tokens", resp.Usage)
		}
	}
	if err := plugin.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if loads != 1 || calls != 2 || closes != 1 {
		t.Errorf("loads, scoring calls, closes = %d, %d, %d; want 1, 2, 1", loads, calls, closes)
	}
}

func TestDecisionRejectsUnsupportedRequestsBeforeLoad(t *testing.T) {
	operator := &testDecisionOperator{loadDecision: func(context.Context, string, DecisionOptions, ContextOptions) (DecisionModel, error) {
		t.Fatal("unsupported request loaded the model")
		return nil, nil
	}}
	plugin, g := initDiscoveryTest(t, t.TempDir(), operator)
	plugin.DefineModel(g, ModelDefinition{
		Name: "decision", Type: ModelTypeDecision, Loading: ModelOptions{GGUFPath: "unused.gguf"},
		Decision: DecisionOptions{Readout: DecisionReadoutGGUF},
	}, nil)
	tool := genkit.DefineTool(g, "unused", "Unused tool", func(_ *ai.ToolContext, in string) (string, error) {
		t.Fatal("decision model executed a tool")
		return in, nil
	})
	tests := []struct {
		name string
		want string
		opts []ai.GenerateOption
	}{
		{"stream", "do not support streaming", []ai.GenerateOption{ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error {
			t.Fatal("decision model streamed a response")
			return nil
		})}},
		{"tool", "does not support tool use", []ai.GenerateOption{ai.WithTools(tool)}},
		{"media", "does not support media", []ai.GenerateOption{ai.WithMessages(ai.NewUserMessage(ai.NewMediaPart("image/png", "data:image/png;base64,AA==")))}},
		{"sampling", "do not support per-request config", []ai.GenerateOption{ai.WithConfig(GenerateContentConfig{Temperature: new(float64)})}},
		{"history", "accept one user message", []ai.GenerateOption{ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("older request")))}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := append([]ai.GenerateOption{
				ai.WithModelName("yzma/decision"), ai.WithPrompt("state"), ai.WithOutputType(ticketDecision{}),
			}, tt.opts...)
			if _, err := genkit.Generate(t.Context(), g, opts...); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Generate error = %v, want %q", err, tt.want)
			}
		})
	}
	t.Run("ordinary output", func(t *testing.T) {
		type notDecision struct {
			Urgent bool `json:"urgent" jsonschema_description:"Is this urgent?"`
		}
		_, _, err := genkit.GenerateData[notDecision](t.Context(), g, ai.WithModelName("yzma/decision"), ai.WithPrompt("state"))
		if err == nil || !strings.Contains(err.Error(), "use Choice, Score, or Noul") {
			t.Errorf("GenerateData error = %v, want a decision field type error", err)
		}
	})
}

func TestDiscoverDecisionCatalog(t *testing.T) {
	dir := t.TempDir()
	writeGGUFFixture(t, dir, "Embedded.gguf", map[string]ggufValue{
		"llama.decision.type": {kind: ggufString, text: "lev"},
		systemOneTemplateKey:  {kind: ggufString, text: "{{ state }}"},
	})
	writeGGUFFixture(t, dir, "Unsupported.gguf", map[string]ggufValue{
		"llama.decision.type": {kind: ggufString, text: "unknown"},
		systemOneTemplateKey:  {kind: ggufString, text: "{{ state }}"},
	})
	writeGGUFFixture(t, dir, "NoTemplate.gguf", map[string]ggufValue{
		"llama.decision.type": {kind: ggufString, text: "lev"},
	})
	for _, name := range []string{"Sidecar.Q2_K", "Invalid", "Ambiguous"} {
		writeGGUFFixture(t, dir, name+".gguf", nil)
		calibration := `{"temperature":1.03}`
		if name == "Invalid" {
			calibration = `{"temperature":-1}`
		}
		if err := os.WriteFile(filepath.Join(dir, name+".decider.json"), []byte(calibration), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "Ambiguous.jevk5.json"), []byte(`{"temperature":1}`), 0600); err != nil {
		t.Fatal(err)
	}
	operator := &testDecisionOperator{loadDecision: func(context.Context, string, DecisionOptions, ContextOptions) (DecisionModel, error) {
		t.Fatal("discovery loaded weights")
		return nil, nil
	}}
	plugin, g := initDiscoveryTest(t, dir, operator)
	if Model(g, "sidecar.q2_k") == nil {
		t.Fatal("sidecar model did not resolve before listing")
	}
	if diff := cmp.Diff([]string{"yzma/embedded", "yzma/sidecar.q2_k"}, discoveryNames(t.Context(), plugin)); diff != "" {
		t.Errorf("catalog (-want +got):\n%s", diff)
	}
	for _, d := range plugin.ListActions(t.Context()) {
		supports := d.Metadata["model"].(map[string]any)["supports"].(map[string]any)
		if supports["constrained"] != ai.ConstrainedSupportNoTools || supports["tools"] != false || supports["media"] != false {
			t.Errorf("%q supports = %v", d.Name, supports)
		}
	}
}

func TestDecisionAnswerRejectsMisalignedProbabilities(t *testing.T) {
	q := decide.Noul("Is this urgent?", "", "")
	r := &decide.Result{Answer: "true", Options: []string{"true", "false"}, Probabilities: []float64{0.9, 0.1}}
	if _, err := decisionAnswer(q, r); err == nil {
		t.Fatal("accepted readout probabilities that did not follow the question's option order")
	}
}

func TestNativeDecision(t *testing.T) {
	path := os.Getenv("YZMA_TEST_DECISION_MODEL")
	if path == "" || os.Getenv(libEnv) == "" {
		t.Skip("set YZMA_TEST_DECISION_MODEL and YZMA_LIB for native inference")
	}
	p := &Yzma{ModelsDir: filepath.Dir(path)}
	g := genkit.Init(t.Context(), genkit.WithPlugins(p))
	t.Cleanup(func() {
		if err := p.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Errorf("close decision model: %v", err)
		}
	})
	id := strings.TrimSuffix(strings.ToLower(filepath.Base(path)), ".gguf")
	got, resp, err := genkit.GenerateData[ticketDecision](t.Context(), g, ai.WithModelName("yzma/"+id),
		ai.WithPrompt("I was billed twice for the same order. Please refund the duplicate charge today."))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("answers: %s", resp.Text())
	if got.Department.Choice != "billing" || got.Department.Probabilities["billing"] <= 0.5 {
		t.Errorf("department = %+v, want billing more likely than technical", got.Department)
	}
	var total float64
	for _, p := range got.Frustration.Probabilities {
		total += p
	}
	if len(got.Frustration.Probabilities) != 3 || math.Abs(total-1) > 1e-6 {
		t.Errorf("frustration = %+v, want a distribution over all three levels", got.Frustration)
	}
	if got.Urgent.Probability <= 0.5 || got.Urgent.Probability > 1 {
		t.Errorf("urgency probability = %v, want (0.5, 1]", got.Urgent.Probability)
	}
}
