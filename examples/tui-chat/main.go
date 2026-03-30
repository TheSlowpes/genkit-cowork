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

// Package main is a minimal terminal-based chat application that demonstrates
// how to wire together the genkit-cowork pillars: tools, flows, memory, and
// the reply-channel abstraction.
//
// It registers the TUI itself as a ChannelHandler for memory.UIMessage, so
// every agent reply is routed through the SendReply flow and printed to
// stdout by the TUI handler.
//
// Usage:
//
//	GEMINI_API_KEY=<your-key> go run ./examples/tui-chat
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/flows"
	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/tools"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

const (
	model     = "googleai/gemini-2.5-flash"
	sessionID = "tui-chat"
	tenantID  = "local"
	maxTurns  = 20
	workDir   = "."
	indexDir  = "./.genkit-memory/index"
	stateDir  = "./.genkit-memory/sessions"
)

// tuiChannelHandler implements flows.ChannelHandler for the terminal UI.
// Setup is a no-op; SendReply writes the agent response to stdout.
type tuiChannelHandler struct{}

func (h *tuiChannelHandler) Setup(_ context.Context, _ string) error { return nil }

func (h *tuiChannelHandler) SendReply(_ context.Context, input *flows.SendReplyInput) error {
	if input.Content == nil {
		return nil
	}
	var sb strings.Builder
	for _, part := range input.Content.Content {
		if part.IsText() {
			sb.WriteString(part.Text)
		}
	}
	text := strings.TrimSpace(sb.String())
	if text != "" {
		fmt.Printf("\n🤖 Agent: %s\n\n", text)
	}
	return nil
}

func (h *tuiChannelHandler) Acknowledge(_ context.Context, _ *flows.AcknowledgeInput) error {
	return nil
}

func main() {
	ctx := context.Background()

	// 1. Initialize Genkit with the Google AI plugin.
	g := genkit.Init(ctx,
		genkit.WithDefaultModel(model),
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
	)

	// 2. Create a tenant-scoped session store with vector indexing.
	embedder := genkit.LookupEmbedder(g, "googleai/gemini-embedding-001")
	vecBackend, err := memory.NewLocalVecBackend(g, "tui-memory", memory.LocalVecConfig{
		Embedder: embedder,
		IndexDir: indexDir,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "init vector backend: %v\n", err)
		os.Exit(1)
	}
	fileBackend := memory.NewFileSessionOperator(stateDir, tenantID)
	vectorOperator := memory.NewVectorOperator(fileBackend, vecBackend, stateDir)
	store := memory.NewSession(
		memory.WithCustomSessionOperator(vectorOperator),
		memory.WithTenantID(tenantID),
	)

	// 3. Register the core tools.
	cwd, err := os.Getwd()
	if err != nil {
		cwd = workDir
	}
	tools.NewBashTool(g, cwd)
	tools.NewReadTool(g, cwd)
	tools.NewEditTool(g, cwd)
	tools.NewWriteTool(g, cwd)
	tools.NewSearchSessionMemoryTool(g, vectorOperator)
	tools.NewSearchTenantMemoryTool(g, vectorOperator)

	// 4. Build the agent config shared between the message and heartbeat flows.
	agentCfg := flows.AgentLoopConfig{
		Model:    model,
		Tools:    []string{"bash", "read", "edit", "write", "search-session-memory", "search-tenant-memory"},
		MaxTurns: maxTurns,
		SystemPrompt: flows.BuildSystemPrompt(flows.SystemPromptOptions{
			CustomPrompt: "You are a helpful assistant running in a terminal. " +
				"You have access to bash, read, edit, write, and memory search tools. " +
				"Be concise.",
		}),
	}

	// 5. Set up the TUI channel handler and SendReply flow.
	tuiHandler := &tuiChannelHandler{}
	senders := map[memory.MessageOrigin]flows.ChannelHandler{
		memory.UIMessage: tuiHandler,
	}
	if err := flows.SetupSenders(ctx, tenantID, senders); err != nil {
		fmt.Fprintf(os.Stderr, "setup senders: %v\n", err)
		os.Exit(1)
	}
	replyFlow := flows.NewSendReplyFlow(g, senders)

	// 6. Wire up the HandleMessage flow.
	msgFlow := flows.NewHandleMessageFlow(g, store,
		flows.WithCustomAgentConfig(agentCfg),
	)

	// 7. Wire up the Heartbeat flow; it delivers alerts to the TUI channel.
	heartbeat := flows.NewHeartbeat(g, store, flows.HeartbeatConfig{
		Interval:  2 * time.Minute,
		SessionID: sessionID,
		TenantID:  tenantID,
		AgentConfig: &flows.AgentLoopConfig{
			Model:    model,
			MaxTurns: 5,
			SystemPrompt: flows.BuildSystemPrompt(flows.SystemPromptOptions{
				CustomPrompt: "Review the conversation and report any outstanding tasks or issues. " +
					"If everything is fine, respond with HEARTBEAT_OK.",
			}),
		},
		Target:   flows.HeartbeatTargetLast,
		To:       memory.UIMessage,
		Delivery: flows.DefaultHeartbeatDelivery(),
	}, flows.WithHeartbeatOnResult(func(output *flows.HeartbeatOutput) {
		if !output.ShouldDeliver || output.Response == nil {
			return
		}
		_, _ = replyFlow.Run(ctx, &flows.SendReplyInput{
			SessionID: output.SessionID,
			Sender:    flows.Sender{TenantID: tenantID, DisplayName: "Agent (heartbeat)"},
			Content:   output.Response,
			Channel:   memory.UIMessage,
			Target:    flows.HeartbeatTargetLast,
			Destination: flows.Destination{
				ChatID: sessionID,
			},
		})
	}))
	heartbeat.Start(ctx)
	defer heartbeat.Stop()

	// 8. Run the TUI read-eval-print loop.
	fmt.Println("=== genkit-cowork TUI Chat ===")
	fmt.Printf("Model: %s  |  Type your message and press Enter. Ctrl-C to quit.\n\n", model)

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		output, err := msgFlow.Run(ctx, &flows.HandleMessageInput{
			SessionID: sessionID,
			TenantID:  tenantID,
			Origin:    memory.UIMessage,
			Content:   *ai.NewUserTextMessage(line),
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			continue
		}

		// Deliver the response through the SendReply flow so the TUI handler
		// handles formatting, and to demonstrate the full reply-channel path.
		_, _ = replyFlow.Run(ctx, &flows.SendReplyInput{
			SessionID: output.SessionID,
			Sender:    flows.Sender{TenantID: tenantID, DisplayName: "Agent"},
			Content:   output.Response,
			Channel:   memory.UIMessage,
			Target:    flows.HeartbeatTargetLast,
			Destination: flows.Destination{
				ChatID: sessionID,
			},
		})
	}
}
