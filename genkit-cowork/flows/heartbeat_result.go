package flows

import (
	"strings"
	"time"
)

type HeartbeatResultKind string

const (
	HeartbeatAck     HeartbeatResultKind = "ack"
	HeartbeatAlert   HeartbeatResultKind = "alert"
	HeartbeatSkipped HeartbeatResultKind = "skipped"
	HeartbeatError   HeartbeatResultKind = "error"
)

type SkipReason string

const (
	SkipReasonBusy         SkipReason = "busy"
	SkipReasonOutsideHours SkipReason = "outside_hours"
)

const heartbeatOKToken = "HEARTBEAT_OK"

type HeartbeatOutput struct {
	Kind      HeartbeatResultKind `json:"kind"`
	SessionID string              `json:"sessionID,omitempty"`
	RunAt     time.Time           `json:"runAt"`

	RawContent      string     `json:"rawContent,omitempty"`
	DeliveryContent string     `json:"deliveryContent,omitempty"`
	ShouldDeliver   bool       `json:"shouldDeliver"`
	SkipReason      SkipReason `json:"skipReason,omitempty"`
	Err             error      `json:"error,omitempty"`
	Turns           int        `json:"turns,omitempty"`
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

func evaluateHearbeatResult(
	sessionID string,
	runAt time.Time,
	rawContent string,
	turns int,
	cfg *HeartbeatConfig,
) HeartbeatOutput {
	kind, stripped := parseHeartbeatResponse(rawContent, cfg.AckMaxChars)

	result := HeartbeatOutput{
		Kind:            kind,
		SessionID:       sessionID,
		RunAt:           runAt,
		RawContent:      rawContent,
		DeliveryContent: stripped,
		Turns:           turns,
	}

	result.ShouldDeliver = shouldDeliver(kind, cfg.Delivery)
	return result
}

func shouldDeliver(kind HeartbeatResultKind, delivery HeartBeatDelivery) bool {
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
		Kind:      HeartbeatError,
		SessionID: sessionID,
		RunAt:     runAt,
		Err:       err,
	}
}
