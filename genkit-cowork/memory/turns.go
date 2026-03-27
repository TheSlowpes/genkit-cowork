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
	"time"
)

// TurnEvent records a typed event that happened during a turn.
type TurnEvent struct {
	Type      string         `json:"type"`
	Timestamp time.Time      `json:"timestamp"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// TurnRecord is an immutable ledger entry for one persisted turn.
type TurnRecord struct {
	TurnID               string      `json:"turnID"`
	Sequence             int64       `json:"sequence"`
	FlowName             string      `json:"flowName,omitempty"`
	LoopTurns            int         `json:"loopTurns,omitempty"`
	FinishReason         string      `json:"finishReason,omitempty"`
	StartedAt            time.Time   `json:"startedAt"`
	EndedAt              time.Time   `json:"endedAt"`
	FirstMessageSequence int64       `json:"firstMessageSequence"`
	LastMessageSequence  int64       `json:"lastMessageSequence"`
	MessageCount         int         `json:"messageCount"`
	Events               []TurnEvent `json:"events,omitempty"`
}

// ValidateSessionLedger validates append-only and sequencing invariants for a
// complete session ledger state.
//
// This validation is storage-agnostic and can be used by operators, tests,
// migrations, or diagnostic tooling before persisting or after loading state.
func ValidateSessionLedger(state SessionState) error {
	return validateAppendOnlyState(SessionState{}, state)
}

func validateAppendOnlyState(existing, incoming SessionState) error {
	if len(incoming.Messages) < len(existing.Messages) {
		return fmt.Errorf("append-only violation: messages truncated from %d to %d", len(existing.Messages), len(incoming.Messages))
	}
	if len(incoming.Turns) < len(existing.Turns) {
		return fmt.Errorf("append-only violation: turns truncated from %d to %d", len(existing.Turns), len(incoming.Turns))
	}
	if len(incoming.Snapshots) < len(existing.Snapshots) {
		return fmt.Errorf("append-only violation: snapshots truncated from %d to %d", len(existing.Snapshots), len(incoming.Snapshots))
	}

	for i := range existing.Messages {
		if existing.Messages[i].MessageID != incoming.Messages[i].MessageID {
			return fmt.Errorf("append-only violation: message %d changed from %q to %q", i, existing.Messages[i].MessageID, incoming.Messages[i].MessageID)
		}
		if existing.Messages[i].Sequence > 0 && incoming.Messages[i].Sequence == 0 {
			return fmt.Errorf("append-only violation: message %d sequence dropped from %d to 0", i, existing.Messages[i].Sequence)
		}
		if existing.Messages[i].Sequence > 0 && incoming.Messages[i].Sequence > 0 && existing.Messages[i].Sequence != incoming.Messages[i].Sequence {
			return fmt.Errorf("append-only violation: message %d sequence changed from %d to %d", i, existing.Messages[i].Sequence, incoming.Messages[i].Sequence)
		}
	}
	for i := range existing.Turns {
		if existing.Turns[i].TurnID != incoming.Turns[i].TurnID {
			return fmt.Errorf("append-only violation: turn %d changed from %q to %q", i, existing.Turns[i].TurnID, incoming.Turns[i].TurnID)
		}
		if existing.Turns[i].Sequence > 0 && incoming.Turns[i].Sequence == 0 {
			return fmt.Errorf("append-only violation: turn %d sequence dropped from %d to 0", i, existing.Turns[i].Sequence)
		}
		if existing.Turns[i].Sequence > 0 && incoming.Turns[i].Sequence > 0 && existing.Turns[i].Sequence != incoming.Turns[i].Sequence {
			return fmt.Errorf("append-only violation: turn %d sequence changed from %d to %d", i, existing.Turns[i].Sequence, incoming.Turns[i].Sequence)
		}
	}
	for i := range existing.Snapshots {
		if existing.Snapshots[i].SnapshotID != incoming.Snapshots[i].SnapshotID {
			return fmt.Errorf("append-only violation: snapshot %d changed from %q to %q", i, existing.Snapshots[i].SnapshotID, incoming.Snapshots[i].SnapshotID)
		}
		if existing.Snapshots[i].Sequence > 0 && incoming.Snapshots[i].Sequence == 0 {
			return fmt.Errorf("append-only violation: snapshot %d sequence dropped from %d to 0", i, existing.Snapshots[i].Sequence)
		}
		if existing.Snapshots[i].Sequence > 0 && incoming.Snapshots[i].Sequence > 0 && existing.Snapshots[i].Sequence != incoming.Snapshots[i].Sequence {
			return fmt.Errorf("append-only violation: snapshot %d sequence changed from %d to %d", i, existing.Snapshots[i].Sequence, incoming.Snapshots[i].Sequence)
		}
	}

	var lastMessageSequence int64
	messageSeqSeen := false
	for i := range incoming.Messages {
		seq := incoming.Messages[i].Sequence
		if seq == 0 {
			continue
		}
		if !messageSeqSeen {
			lastMessageSequence = seq
			messageSeqSeen = true
			continue
		}
		if seq <= lastMessageSequence {
			return fmt.Errorf("message sequence must be strictly increasing")
		}
		lastMessageSequence = seq
	}

	var lastTurnSequence int64
	turnSeqSeen := false
	for i := range incoming.Turns {
		seq := incoming.Turns[i].Sequence
		if seq == 0 {
			continue
		}
		if !turnSeqSeen {
			lastTurnSequence = seq
			turnSeqSeen = true
			continue
		}
		if seq <= lastTurnSequence {
			return fmt.Errorf("turn sequence must be strictly increasing")
		}
		lastTurnSequence = seq
	}

	var lastSnapshotSequence int64
	snapshotSeqSeen := false
	for i := range incoming.Snapshots {
		seq := incoming.Snapshots[i].Sequence
		if seq == 0 {
			continue
		}
		if !snapshotSeqSeen {
			lastSnapshotSequence = seq
			snapshotSeqSeen = true
			continue
		}
		if seq <= lastSnapshotSequence {
			return fmt.Errorf("snapshot sequence must be strictly increasing")
		}
		lastSnapshotSequence = seq
	}

	return nil
}
