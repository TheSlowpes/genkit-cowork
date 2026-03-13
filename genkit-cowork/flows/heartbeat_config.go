package flows

import (
	"context"
	"fmt"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
)

type HearbeatTarget string

const (
	HearbeatTargetNone HearbeatTarget = "none"
	HearbeatTargetLast HearbeatTarget = "last"
)

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

type HeartBeatDelivery struct {
	ShowOk       bool `json:"showOk"`
	ShowAlerts   bool `json:"showAlerts"`
	UseIndicator bool `json:"useIndicator"`
}

func DefaultHeartBeatDelivery() HeartBeatDelivery {
	return HeartBeatDelivery{
		ShowOk:       false,
		ShowAlerts:   true,
		UseIndicator: true,
	}
}

type HeartbeatConfig struct {
	Interval    time.Duration `json:"interval"`
	ActiveHours *ActiveHours  `json:"activeHours,omitempty"`

	AgentConfig *AgentLoopConfig     `json:"agentConfig,omitempty"`
	Prompt      ai.PromptFn          `json:"prompt,omitempty"`
	SessionID   string               `json:"sessionID,omitempty"`
	AckMaxChars int                  `json:"ackMaxChars,omitempty"`
	Target      HearbeatTarget       `json:"target,omitempty"`
	To          memory.MessageOrigin `json:"to,omitempty"`
	Delivery    HeartBeatDelivery    `json:"delivery"`
}

const DefaultPrompt = "Read HEARTBEAT.md if it exists. Follow it strictly. Do not infer or repeat old tasks from prior chats. If nothing needs attention, reply HEARTBEAT_OK"

func (c *HeartbeatConfig) resolvedPrompt(ctx context.Context, a any) (string, error) {
	if c.Prompt != nil {
		prompt, err := c.Prompt(ctx, a)
		if err != nil {
			return "", fmt.Errorf("resolving prompt")
		}
		return prompt, nil
	}
	return DefaultPrompt, nil
}

func (c *HeartbeatConfig) resolvedAckMaxChars() int {
	if c.AckMaxChars > 0 {
		return c.AckMaxChars
	}
	return 300
}
