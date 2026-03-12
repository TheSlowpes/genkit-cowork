package flows

import (
	"context"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

type HeartbeatInput struct{}

type heartbeatOptions struct {
	operator AgentLoopOperator
}

type HeartbeatOptions func(*heartbeatOptions)

func NewHeartbeatFlow(
	g *genkit.Genkit,
	store *memory.Session,
	opts ...HeartbeatOptions,
) core.Flow[*HeartbeatInput, *HeartbeatOutput, struct{}] {
	options := &heartbeatOptions{
		operator: &defaultAgentLoopOperator{g: g},
	}
	for _, opt := range opts {
		opt(options)
	}
	return genkit.DefineFlow(
		g,
		"heartbeat",
	)
}
