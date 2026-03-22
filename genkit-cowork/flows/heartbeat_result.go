// Copyright 2025 Google LLC
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

package flows

import (
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
)

// HeartbeatResultKind classifies heartbeat execution outcomes.
type HeartbeatResultKind string

const (
	// HeartbeatAck indicates an "all clear" heartbeat result.
	HeartbeatAck HeartbeatResultKind = "ack"
	// HeartbeatAlert indicates actionable or anomalous heartbeat output.
	HeartbeatAlert HeartbeatResultKind = "alert"
	// HeartbeatSkipped indicates the heartbeat was intentionally not run.
	HeartbeatSkipped HeartbeatResultKind = "skipped"
	// HeartbeatError indicates the heartbeat run failed.
	HeartbeatError HeartbeatResultKind = "error"
)

// SkipReason explains why a heartbeat run was skipped.
type SkipReason string

const (
	// SkipReasonBusy indicates a previous heartbeat run is still in progress.
	SkipReasonBusy SkipReason = "busy"
	// SkipReasonOutsideHours indicates the current time is outside ActiveHours.
	SkipReasonOutsideHours SkipReason = "outside_hours"
	// SkipReasonErrors indicates repeated or policy-based error backoff.
	SkipReasonErrors SkipReason = "errors"
)

const heartbeatOKToken = "HEARTBEAT_OK"

// HeartbeatOutput is the structured result returned by heartbeat flow runs.
type HeartbeatOutput struct {
	Kind      HeartbeatResultKind `json:"kind"`
	SessionID string              `json:"sessionID,omitempty"`
	RunAt     time.Time           `json:"runAt"`

	Response        *ai.Message `json:"response,omitempty"`
	RawContent      string      `json:"rawContent,omitempty"`
	DeliveryContent string      `json:"deliveryContent,omitempty"`
	ShouldDeliver   bool        `json:"shouldDeliver"`
	SkipReason      SkipReason  `json:"skipReason,omitempty"`
	Err             error       `json:"-"`
	ErrMessage      string      `json:"error,omitempty"`
	Turns           int         `json:"turns,omitempty"`
}

func parseHeartbeatResponse(raw string, ackMaxChars int) (kind HeartbeatResultKind, stripped string) {
	trimmed := strings.TrimSpace(raw)

	atStart := strings.HasPrefix(trimmed, heartbeatOKToken)
	atEnd := strings.HasSuffix(trimmed, heartbeatOKToken)

	if !atStart && !atEnd {
		return HeartbeatAlert, trimmed
	}

	remaining := trimmed
	if atStart {
		remaining = strings.TrimSpace(trimmed[len(heartbeatOKToken):])
	} else {
		remaining = strings.TrimSpace(trimmed[:len(trimmed)-len(heartbeatOKToken)])
	}

	if len(remaining) <= ackMaxChars {
		return HeartbeatAck, remaining
	}

	return HeartbeatAlert, remaining
}

func evaluateHeartbeatResult(
	sessionID string,
	runAt time.Time,
	rawContent string,
	turns int,
	cfg *HeartbeatConfig,
	response *ai.Message,
) HeartbeatOutput {
	kind, stripped := parseHeartbeatResponse(rawContent, cfg.resolvedAckMaxChars())

	result := HeartbeatOutput{
		Kind:            kind,
		SessionID:       sessionID,
		RunAt:           runAt,
		Response:        response,
		RawContent:      rawContent,
		DeliveryContent: stripped,
		Turns:           turns,
	}

	result.ShouldDeliver = shouldDeliver(kind, cfg.Delivery)
	return result
}

func shouldDeliver(kind HeartbeatResultKind, delivery HeartbeatDelivery) bool {
	switch kind {
	case HeartbeatAck:
		return delivery.ShowOk
	case HeartbeatAlert:
		return delivery.ShowAlerts
	default:
		return false
	}
}

func skippedResult(sessionID string, reason SkipReason) HeartbeatOutput {
	return HeartbeatOutput{
		Kind:       HeartbeatSkipped,
		SessionID:  sessionID,
		RunAt:      time.Now(),
		SkipReason: reason,
	}
}

func errorResult(sessionID string, runAt time.Time, err error) HeartbeatOutput {
	return HeartbeatOutput{
		Kind:       HeartbeatError,
		SessionID:  sessionID,
		RunAt:      runAt,
		Err:        err,
		ErrMessage: err.Error(),
	}
}
