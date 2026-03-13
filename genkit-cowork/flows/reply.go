package flows

import (
	"context"
	"fmt"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

type ChannelHandler interface {
	Setup(ctx context.Context, tenantID string) error
	SendReply(ctx context.Context, input *SendReplyInput) error
	Acknowledge(ctx context.Context, input *AcknowledgeInput) error
}

type sendReplyOptions struct {
	replyInThread bool
}

type SendReplyOption func(*sendReplyOptions)

func WithReplyInThread() SendReplyOption {
	return func(opts *sendReplyOptions) {
		opts.replyInThread = true
	}
}

type Sender struct {
	TenantID    string  `json:"tenantID"`
	DisplayName string  `json:"displayName"`
	Username    *string `json:"username,omitempty"`
}

type Destination struct {
	ChatID    string  `json:"chatID"`
	MessageID *string `json:"messageID,omitempty"`
	ThreadID  *string `json:"threadID,omitempty"`
}

type SendReplyInput struct {
	SessionID   string               `json:"sessionID"`
	Sender      Sender               `json:"sender"`
	Content     *ai.Message          `json:"content"`
	Channel     memory.MessageOrigin `json:"channel"`
	Target      HeartbeatTarget      `json:"target,omitempty"`
	Destination Destination          `json:"destination"`
	channelMeta map[string]any       `json:"-"`
}

type SendReplyOutput struct {
	SessionID string               `json:"sessionID"`
	Channel   memory.MessageOrigin `json:"channel"`
	Delivered bool                 `json:"delivered"`
	Skipped   bool                 `json:"skipped"`
	Reason    string               `json:"reason,omitempty"`
}

type AcknowledgeInput struct {
	SessionID   string               `json:"sessionID"`
	Sender      Sender               `json:"sender"`
	Channel     memory.MessageOrigin `json:"channel"`
	Destination Destination          `json:"destination"`
	channelMeta map[string]any       `json:"-"`
}

func NewSendReplyFlow(
	g *genkit.Genkit,
	senders map[memory.MessageOrigin]ChannelHandler,
	opts ...SendReplyOption,
) *core.Flow[*SendReplyInput, *SendReplyOutput, struct{}] {
	options := &sendReplyOptions{}
	for _, opt := range opts {
		opt(options)
	}

	_ = options // will be used when replyInThread is wired into flow logic

	return genkit.DefineFlow(
		g,
		"sendReply",
		func(ctx context.Context, input *SendReplyInput) (*SendReplyOutput, error) {
			output := &SendReplyOutput{
				SessionID: input.SessionID,
				Channel:   input.Channel,
			}

			handler, ok := senders[input.Channel]
			if !ok {
				output.Skipped = true
				output.Reason = fmt.Sprintf("no sender registered for channel %q", input.Channel)
				return output, nil
			}

			if input.Target == HeartbeatTargetNone || input.Target == "" {
				output.Skipped = true
				output.Reason = "target is none"
				return output, nil
			}

			if err := handler.SendReply(ctx, input); err != nil {
				return nil, fmt.Errorf("send reply via %q: %w", input.Channel, err)
			}

			output.Delivered = true
			return output, nil
		},
	)
}

func SetupSenders(
	ctx context.Context,
	tenantID string,
	senders map[memory.MessageOrigin]ChannelHandler,
) error {
	for channel, handler := range senders {
		if err := handler.Setup(ctx, tenantID); err != nil {
			return fmt.Errorf("setup sender for channel %q: %w", channel, err)
		}
	}
	return nil
}
