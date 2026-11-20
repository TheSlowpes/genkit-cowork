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
)

const (
	harmonyCallEnd    = "<|call|>"
	harmonyTurnEnd    = "<|return|>"
	harmonyMessageEnd = "<|end|>"
	harmonyMessage    = "<|message|>"
	harmonyJSONBody   = `\s*(?:<\|constrain\|>\s*json\s*)?<\|message\|>`
)

var (
	harmonyRecipient     = regexp.MustCompile(`^functions\.(` + wireToolName + `)` + harmonyJSONBody)
	harmonyRoleRecipient = regexp.MustCompile(`^functions\.(` + wireToolName + `)\s*<\|channel\|>commentary` + harmonyJSONBody)
	harmonyText          = regexp.MustCompile(`^<\|channel\|>(analysis|commentary|final)\s*<\|message\|>`)
	harmonyChannelTo     = regexp.MustCompile(`^<\|channel\|>commentary\s+to=`)
)

func (t *wireTurn) harmony(raw string) (string, error) {
	var header []string
	if tail, started := strings.CutPrefix(raw, "<|start|>"); started {
		var assistant bool
		raw, assistant = strings.CutPrefix(tail, "assistant")
		if !assistant {
			return "", nil // A simulated user, tool, or system turn.
		}
		if recipient, ok := strings.CutPrefix(strings.TrimLeft(raw, " \t"), "to="); ok {
			header = harmonyRoleRecipient.FindStringSubmatch(recipient)
			if header == nil {
				return "", fmt.Errorf("unsupported or malformed Harmony recipient")
			}
			raw = recipient
		}
	}
	if loc := harmonyChannelTo.FindStringIndex(raw); loc != nil {
		raw = raw[loc[1]:]
		header = harmonyRecipient.FindStringSubmatch(raw)
	}
	if header != nil {
		kind, calls, tail, err := jsonCallSpan(wireHarmony, raw[len(header[0]):], header[1], harmonyCallEnd)
		if err != nil {
			return "", err
		}
		return tail, t.add(kind, calls)
	}
	if frame := harmonyText.FindStringSubmatch(raw); frame != nil {
		body, tail := harmonyTextSpan(raw[len(frame[0]):])
		if frame[1] != "analysis" {
			t.visible(body)
		}
		if frame[1] == "final" {
			return "", nil
		}
		return tail, nil
	}
	if body, ok := strings.CutPrefix(raw, harmonyMessage); ok {
		text, _ := harmonyTextSpan(body)
		t.visible(text)
		return "", nil
	}
	return "", fmt.Errorf("unsupported or malformed Harmony header")
}

// A text frame is opaque to tool detection, even if its body quotes a call
// header. Only message and turn terminators can end it, outside quoted text.
func harmonyTextSpan(raw string) (body, tail string) {
	for i := 0; i < len(raw); {
		n := strings.IndexAny(raw[i:], `"<`)
		if n < 0 {
			break
		}
		i += n
		switch {
		case raw[i] == '"':
			if end, ok := quotedEnd(raw[i:]); ok {
				i += end
				continue
			}
		case strings.HasPrefix(raw[i:], harmonyTurnEnd):
			return raw[:i], ""
		case strings.HasPrefix(raw[i:], harmonyMessageEnd):
			return raw[:i], raw[i+len(harmonyMessageEnd):]
		}
		i++
	}
	return raw, ""
}
