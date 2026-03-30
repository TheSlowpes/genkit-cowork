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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"time"
)

// StateSnapshot captures immutable checkpoint metadata for one whole-state write.
//
// Canonical replay state lives in SessionState persisted by SessionOperator.
// Snapshots provide ordered checkpoint metadata and an integrity checksum over
// the canonical state.
type StateSnapshot struct {
	SnapshotID    string    `json:"snapshotID"`
	Sequence      int64     `json:"sequence"`
	CapturedAt    time.Time `json:"capturedAt"`
	TenantID      string    `json:"tenantID"`
	SessionID     string    `json:"sessionID"`
	MessageCount  int       `json:"messageCount"`
	TurnSequence  int64     `json:"turnSequence"`
	StateVersion  int       `json:"stateVersion"`
	StateChecksum string    `json:"stateChecksum"`
}

func stateChecksum(state SessionState) string {
	type checksumState struct {
		TenantID      string           `json:"tenantID"`
		Messages      []SessionMessage `json:"messages"`
		Turns         []TurnRecord     `json:"turns"`
		MessageAssets []SessionAsset   `json:"assets"`
	}

	b, err := json.Marshal(checksumState{
		TenantID:      state.TenantID,
		Messages:      state.Messages,
		Turns:         state.Turns,
		MessageAssets: state.Assets,
	})
	if err != nil {
		return ""
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
