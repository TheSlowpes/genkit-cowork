package flows

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/firebase/genkit/go/genkit"
)

type Heartbeat struct {
	cfg      *HeartbeatConfig
	flow     *core.Flow[*HeartbeatInput, *HeartbeatOutput, struct{}]
	onResult func(*HeartbeatOutput)

	running atomic.Bool
	stopCh  chan struct{}
	once    sync.Once
}

type HeartbeatInput struct {
	SessionID   string           `json:"sessionID"`
	AgentConfig *AgentLoopConfig `json:"agentConfig,omitempty"`
	TenantID    string           `json:"tenantID,omitempty"`
}

type heartbeatOptions struct {
	bus           *EventBus
	defaultConfig *AgentLoopConfig
	onResult      func(*HeartbeatOutput)
	operator      AgentLoopOperator
}

type HeartbeatOptions func(*heartbeatOptions)

func WithHeartbeatEventBus(bus *EventBus) HeartbeatOptions {
	return func(opts *heartbeatOptions) {
		opts.bus = bus
	}
}

func WithHeartbeatOnResult(onResult func(*HeartbeatOutput)) HeartbeatOptions {
	return func(opts *heartbeatOptions) {
		opts.onResult = onResult
	}
}

func WithHeartbeatLoopOperator(loopOperator AgentLoopOperator) HeartbeatOptions {
	return func(opts *heartbeatOptions) {
		opts.operator = loopOperator
	}
}

func WithCustomHeartbeatAgentConfig(config AgentLoopConfig) HeartbeatOptions {
	return func(opts *heartbeatOptions) {
		opts.defaultConfig = &config
	}
}

func NewHeartbeatFlow(
	g *genkit.Genkit,
	store *memory.Session,
	opts ...HeartbeatOptions,
) *core.Flow[*HeartbeatInput, *HeartbeatOutput, struct{}] {
	options := &heartbeatOptions{
		operator: &defaultAgentLoopOperator{g: g},
	}
	for _, opt := range opts {
		opt(options)
	}
	return genkit.DefineFlow(
		g,
		"heartbeat",
		func(ctx context.Context, input *HeartbeatInput) (*HeartbeatOutput, error) {
			sess, err := session.Load(ctx, store, input.SessionID)
			if err != nil {
				sess, err = session.New(ctx,
					session.WithID[memory.SessionState](input.SessionID),
					session.WithStore(store),
					session.WithInitialState(memory.SessionState{
						TenantID: input.TenantID,
					}),
				)
				if err != nil {
					return nil, fmt.Errorf("create new session: %w", err)
				}
			}

			ctx = session.NewContext(ctx, sess)

			var history []*ai.Message
			for _, msg := range sess.State().Messages {
				history = append(history, &msg.Content)
			}

			resolvedConfig := mergeAgentConfig(options.defaultConfig, input.AgentConfig)
			_ = resolvedConfig
			return nil, nil
		},
	)
}

func (h *Heartbeat) Start(ctx context.Context) {
	if h.cfg.Interval <= 0 {
		return
	}

	go func() {
		ticker := time.NewTicker(h.cfg.Interval)
		defer ticker.Stop()

		for {
			select {
			case <-h.stopCh:
				return
			case <-ctx.Done():
				return
			case t := <-ticker.C:
				h.Run(ctx, t)
			}
		}
	}()
}

func (h *Heartbeat) Wake(ctx context.Context) {
	go h.Run(ctx, time.Now())
}

func (h *Heartbeat) Stop() {
	h.once.Do(func() {
		close(h.stopCh)
	})
}

func (h *Heartbeat) tryRun(ctx context.Context, tickTime time.Time) HeartbeatOutput {

}

func (h *Heartbeat) Run(ctx context.Context, tickTime time.Time) {
	sessionID := h.sessionID()

	if !h.cfg.ActiveHours.Contains(tickTime) {
		skipResult := skippedResult(sessionID, SkipReasonOutsideHours)
		h.onResult(&skipResult)
		return
	}

	if !h.running.CompareAndSwap(false, true) {
		skipResult := skippedResult(sessionID, SkipReasonBusy)
		h.onResult(&skipResult)
		return
	}
	defer h.running.Store(false)

	result := h.tryRun(ctx, tickTime)
	h.onResult(&result)
}

func (h *Heartbeat) sessionID() string {
	if h.cfg.SessionID != "" {
		return h.cfg.SessionID
	}
	return "heartbeat"
}
