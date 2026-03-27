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

package memory

import (
	"fmt"

	"github.com/firebase/genkit/go/ai"
)

// TokenEstimator estimates token usage for a persisted message.
type TokenEstimator interface {
	EstimateTokens(msg SessionMessage) int
}

// TokenEstimatorFunc adapts a function to TokenEstimator.
type TokenEstimatorFunc func(msg SessionMessage) int

// EstimateTokens implements TokenEstimator.
func (f TokenEstimatorFunc) EstimateTokens(msg SessionMessage) int {
	return f(msg)
}

type generationUsageTokenEstimator struct{}

func (generationUsageTokenEstimator) EstimateTokens(msg SessionMessage) int {
	usage, ok := generationUsageFromMessage(msg)
	if !ok {
		return 1
	}
	if usage.TotalTokens > 0 {
		return usage.TotalTokens
	}
	if usage.InputTokens+usage.OutputTokens > 0 {
		return usage.InputTokens + usage.OutputTokens
	}
	return 1
}

func generationUsageFromMessage(msg SessionMessage) (ai.GenerationUsage, bool) {
	meta := msg.Content.Metadata
	if meta == nil {
		return ai.GenerationUsage{}, false
	}
	raw, ok := meta["generationUsage"]
	if !ok || raw == nil {
		return ai.GenerationUsage{}, false
	}

	switch v := raw.(type) {
	case ai.GenerationUsage:
		return v, true
	case *ai.GenerationUsage:
		if v == nil {
			return ai.GenerationUsage{}, false
		}
		return *v, true
	case map[string]any:
		usage := ai.GenerationUsage{}
		if total, ok := toInt(v["totalTokens"]); ok {
			usage.TotalTokens = total
		}
		if in, ok := toInt(v["inputTokens"]); ok {
			usage.InputTokens = in
		}
		if out, ok := toInt(v["outputTokens"]); ok {
			usage.OutputTokens = out
		}
		if usage.TotalTokens == 0 && usage.InputTokens == 0 && usage.OutputTokens == 0 {
			return ai.GenerationUsage{}, false
		}
		return usage, true
	default:
		return ai.GenerationUsage{}, false
	}
}

func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	case float32:
		return int(n), true
	default:
		return 0, false
	}
}

func applyTokenBudget(msgs []SessionMessage, budget int, estimator TokenEstimator) ([]SessionMessage, error) {
	if budget <= 0 {
		return []SessionMessage{}, nil
	}
	if len(msgs) == 0 {
		return []SessionMessage{}, nil
	}
	if estimator == nil {
		estimator = generationUsageTokenEstimator{}
	}

	start := len(msgs)
	used := 0
	for i := len(msgs) - 1; i >= 0; i-- {
		cost := estimator.EstimateTokens(msgs[i])
		if cost <= 0 {
			cost = 1
		}
		if used+cost > budget {
			break
		}
		used += cost
		start = i
	}

	if start >= len(msgs) {
		return []SessionMessage{}, nil
	}

	pruned := copyMessages(msgs[start:])
	if len(pruned) == 0 && len(msgs) > 0 {
		return nil, fmt.Errorf("token budget pruning produced empty history unexpectedly")
	}

	return pruned, nil
}
