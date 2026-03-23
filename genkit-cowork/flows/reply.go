// Copyright 2025 Kevin Lopes
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
	"context"
	"fmt"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

// ChannelHandler adapts channel-specific setup and reply delivery.
type ChannelHandler interface {
	Setup(ctx context.Context, tenantID string) error
	SendReply(ctx context.Context, input *SendReplyInput) error
	Acknowledge(ctx context.Context, input *AcknowledgeInput) error
}

type sendReplyOptions struct {
	replyInThread bool
}

// SendReplyOption configures sendReply flow behavior.
type SendReplyOption func(*sendReplyOptions)

// WithReplyInThread enables threaded reply behavior for channel handlers that
// support thread routing.
func WithReplyInThread() SendReplyOption {
	return func(opts *sendReplyOptions) {
		opts.replyInThread = true
	}
}

// Sender describes the identity used when sending a reply.
type Sender struct {
	TenantID    string  `json:"tenantID"`
	DisplayName string  `json:"displayName"`
	Username    *string `json:"username,omitempty"`
}

// Destination identifies the channel destination for a reply.
type Destination struct {
	ChatID    string  `json:"chatID"`
	MessageID *string `json:"messageID,omitempty"`
	ThreadID  *string `json:"threadID,omitempty"`
}

// SendReplyInput is the input payload for sendReply flow executions.
type SendReplyInput struct {
	SessionID   string               `json:"sessionID"`
	Sender      Sender               `json:"sender"`
	Content     *ai.Message          `json:"content"`
	Channel     memory.MessageOrigin `json:"channel"`
	Target      HeartbeatTarget      `json:"target,omitempty"`
	Destination Destination          `json:"destination"`
	channelMeta map[string]any       `json:"-"`
}

// SendReplyOutput reports whether a reply was delivered or skipped.
type SendReplyOutput struct {
	SessionID string               `json:"sessionID"`
	Channel   memory.MessageOrigin `json:"channel"`
	Delivered bool                 `json:"delivered"`
	Skipped   bool                 `json:"skipped"`
	Reason    string               `json:"reason,omitempty"`
}

// AcknowledgeInput describes a non-message acknowledgement to a channel.
type AcknowledgeInput struct {
	SessionID   string               `json:"sessionID"`
	Sender      Sender               `json:"sender"`
	Channel     memory.MessageOrigin `json:"channel"`
	Destination Destination          `json:"destination"`
	channelMeta map[string]any       `json:"-"`
}

// NewSendReplyFlow creates a flow that routes model replies to channel
// handlers based on message origin.
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

// SetupSenders initializes all channel handlers for a tenant.
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
