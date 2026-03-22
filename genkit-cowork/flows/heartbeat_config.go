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
	"fmt"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
)

// HeartbeatTarget controls where heartbeat replies are directed.
type HeartbeatTarget string

const (
	// HeartbeatTargetNone disables reply delivery for heartbeat runs.
	HeartbeatTargetNone HeartbeatTarget = "none"
	// HeartbeatTargetLast sends replies to the most recent destination context.
	HeartbeatTargetLast HeartbeatTarget = "last"
)

// ActiveHours defines a daily active window in HH:MM local time.
type ActiveHours struct {
	Start    string `json:"start"`              // e.g. "09:00"
	End      string `json:"end"`                // e.g. "17:00"
	Timezone string `json:"timezone,omitempty"` // IANA tz; empty = host local
}

func (a ActiveHours) location() *time.Location {
	if a.Timezone == "" {
		return time.Local
	}
	loc, err := time.LoadLocation(a.Timezone)
	if err != nil {
		return time.Local
	}
	return loc
}

func parseHHMM(s string) (h int, m int, err error) {
	if s == "24:00" {
		return 24, 0, nil
	}
	_, err = fmt.Sscanf(s, "%d:%d", &h, &m)
	if err != nil {
		return 0, 0, fmt.Errorf("heartbeat: invalid time %q (want HH:MM)", s)
	}
	return h, m, nil
}

// Contains reports whether t falls within the configured active window.
// A nil receiver or invalid configuration fails open and returns true.
func (a *ActiveHours) Contains(t time.Time) bool {
	if a == nil {
		return true
	}
	loc := a.location()
	local := t.In(loc)

	sh, sm, err := parseHHMM(a.Start)
	if err != nil {
		return true
	}

	eh, em, err := parseHHMM(a.End)
	if err != nil {
		return true
	}

	startMins := sh*60 + sm
	endMins := eh*60 + em

	nowMins := local.Hour()*60 + local.Minute()
	return nowMins >= startMins && nowMins < endMins
}

// HeartbeatDelivery controls when heartbeat results should be forwarded.
type HeartbeatDelivery struct {
	ShowOk       bool `json:"showOk"`
	ShowAlerts   bool `json:"showAlerts"`
	UseIndicator bool `json:"useIndicator"`
}

// DefaultHeartbeatDelivery returns the default heartbeat delivery policy.
func DefaultHeartbeatDelivery() HeartbeatDelivery {
	return HeartbeatDelivery{
		ShowOk:       false,
		ShowAlerts:   true,
		UseIndicator: true,
	}
}

// HeartbeatConfig configures scheduler behavior, message routing, and loop
// options for heartbeat runs.
type HeartbeatConfig struct {
	Interval    time.Duration `json:"interval"`
	ActiveHours *ActiveHours  `json:"activeHours,omitempty"`

	AgentConfig *AgentLoopConfig     `json:"agentConfig,omitempty"`
	Prompt      ai.PromptFn          `json:"-"`
	SessionID   string               `json:"sessionID,omitempty"`
	TenantID    string               `json:"tenantID,omitempty"`
	AckMaxChars int                  `json:"ackMaxChars,omitempty"`
	Target      HeartbeatTarget      `json:"target,omitempty"`
	To          memory.MessageOrigin `json:"to,omitempty"`
	Delivery    HeartbeatDelivery    `json:"delivery"`
}

func (c *HeartbeatConfig) resolvedAckMaxChars() int {
	if c.AckMaxChars > 0 {
		return c.AckMaxChars
	}
	return 300
}
