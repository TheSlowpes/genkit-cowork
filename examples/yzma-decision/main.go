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

// The yzma-decision example routes a support ticket with a local System One
// model. Its output struct declares the questions, so the app can inspect
// probabilities without parsing generated prose. To run it with the public
// Decider 0.8B GGUF and calibration sidecar, follow the decision example setup
// in genkit-cowork/plugins/yzma/README.md.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/yzma"
)

type department string

func (department) Criteria() map[department]string {
	return map[department]string{
		"billing":   "Payments, invoices, and refunds",
		"technical": "Bugs and outages",
		"other":     "Requests that are not about billing or technical problems",
	}
}

type frustration int

func (frustration) Levels() []string {
	return []string{"Calm", "Concerned but civil", "Very angry"}
}

type triage struct {
	Department yzma.Choice[department] `json:"department" jsonschema_description:"Which team should handle the ticket?"`
	Mood       yzma.Score[frustration] `json:"frustration" jsonschema_description:"How frustrated is the customer?"`
	Urgent     yzma.Noul               `json:"urgent" jsonschema_description:"Does the customer explicitly request a same-day fix?"`
}

func main() {
	dir := flag.String("models-dir", "", "directory containing the GGUF and its sidecar")
	id := flag.String("model", "decider-0.8b.q2_k", "unprefixed Yzma model ID")
	ticket := flag.String("ticket", "I was billed twice for the same order. Please refund the duplicate charge today.", "ticket to classify")
	flag.Parse()
	if err := run(context.Background(), *dir, *id, *ticket); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, dir, id, ticket string) (err error) {
	p := &yzma.Yzma{ModelsDir: dir}
	g := genkit.Init(ctx, genkit.WithPlugins(p))
	defer func() {
		if closeErr := p.Close(context.WithoutCancel(ctx)); err == nil {
			err = closeErr
		}
	}()
	model := yzma.Model(g, id)
	if model == nil {
		return fmt.Errorf("model %q not found; check the GGUF and matching calibration sidecar in %q", id, dir)
	}
	out, _, err := genkit.GenerateData[triage](ctx, g, ai.WithModel(model),
		ai.WithSystem("The state is a customer support ticket."), ai.WithPrompt(ticket))
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}
