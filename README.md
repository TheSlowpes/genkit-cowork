# genkit-cowork

A coworking framework for [Firebase Genkit](https://github.com/firebase/genkit) (Go) that gives AI agents pluggable capabilities for autonomous and interactive work.

```
go get github.com/TheSlowpes/genkit-cowork
```

## Quick Start

### 1. Create a Genkit instance and session store

```go
package main

import (
    "context"

    "github.com/TheSlowpes/genkit-cowork/genkit-cowork/flows"
    "github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
    "github.com/firebase/genkit/go/genkit"
)

func main() {
    ctx := context.Background()
    g, _ := genkit.Init(ctx, genkit.WithDefaultModel("googleai/gemini-2.0-flash"))
    store := memory.NewSession()

    // ...
}
```

### 2. Register tools

Each tool is a standalone `ai.Tool` that works with any Genkit instance:

```go
import "github.com/TheSlowpes/genkit-cowork/genkit-cowork/tools"

bashTool := tools.NewBashTool(g, "/working/dir",
    tools.WithCommandPrefix("#!/bin/bash\nset -e\n"),
)
readTool  := tools.NewReadTool(g, "/working/dir")
editTool  := tools.NewEditTool(g, "/working/dir")
writeTool := tools.NewWriteTool(g, "/working/dir")
```

### 3. Set up message handling

`HandleMessageFlow` is a session-backed chat flow. It loads or creates session state, runs the agent loop, and persists the conversation:

```go
messageFlow := flows.NewHandleMessageFlow(g, store,
    flows.WithDefaultConfig(flows.AgentLoopConfig{
        Model:    "googleai/gemini-2.0-flash",
        Tools:    []string{"bash", "read", "edit", "write"},
        MaxTurns: 25,
    }),
)

output, err := messageFlow.Run(ctx, &flows.HandleMessageInput{
    SessionID: "session-1",
    TenantID:  "tenant-1",
    Origin:    memory.UIMessage,
    Content:   ai.Message{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("Hello")}},
})
```

### 4. Set up heartbeat monitoring

`Heartbeat` runs the agent loop on a schedule against existing session state, classifying results as `ack`, `alert`, `skipped`, or `error`:

```go
heartbeat := flows.NewHeartbeat(g, store, flows.HeartbeatConfig{
    Interval:  5 * time.Minute,
    SessionID: "heartbeat-session",
    TenantID:  "tenant-1",
    ActiveHours: &flows.ActiveHours{
        Start:    "09:00",
        End:      "17:00",
        Timezone: "America/New_York",
    },
    Delivery: flows.DefaultHeartbeatDelivery(),
    Target:   flows.HeartbeatTargetLast,
    To:       memory.WhatsAppMessage,
    AgentConfig: &flows.AgentLoopConfig{
        Model: "googleai/gemini-2.0-flash",
    },
}, flows.WithHeartbeatOnResult(func(output *flows.HeartbeatOutput) {
    if output.ShouldDeliver {
        // Forward to sendReply flow
    }
}))

heartbeat.Start(ctx)
defer heartbeat.Stop()
```

### 5. Set up reply delivery

`SendReplyFlow` routes agent output to external channels via the `ChannelHandler` interface:

```go
// Implement ChannelHandler for your channels
type WhatsAppHandler struct { /* ... */ }

func (h *WhatsAppHandler) Setup(ctx context.Context, tenantID string) error { /* ... */ }
func (h *WhatsAppHandler) SendReply(ctx context.Context, input *flows.SendReplyInput) error { /* ... */ }
func (h *WhatsAppHandler) Acknowledge(ctx context.Context, input *flows.AcknowledgeInput) error { /* ... */ }

// Register handlers by channel
senders := map[memory.MessageOrigin]flows.ChannelHandler{
    memory.WhatsAppMessage: &WhatsAppHandler{},
    memory.ZoomMessage:     &ZoomHandler{},
}

// Run per-tenant channel setup (webhooks, token verification, etc.)
if err := flows.SetupSenders(ctx, "tenant-1", senders); err != nil {
    log.Fatal(err)
}

// Create the flow
replyFlow := flows.NewSendReplyFlow(g, senders)
```

### 6. Wire heartbeat to reply delivery

```go
heartbeat := flows.NewHeartbeat(g, store, cfg,
    flows.WithHeartbeatOnResult(func(output *flows.HeartbeatOutput) {
        if !output.ShouldDeliver {
            return
        }
        replyFlow.Run(ctx, &flows.SendReplyInput{
            SessionID: output.SessionID,
            Sender:    flows.Sender{TenantID: cfg.TenantID, DisplayName: "Agent"},
            Content:   output.Response,
            Channel:   cfg.To,
            Target:    cfg.Target,
            Destination: flows.Destination{
                ChatID: "resolved-chat-id",
            },
        })
    }),
)
```

### 7. Add observability with EventBus

The `EventBus` provides typed lifecycle events for agent, turn, message, and tool execution stages:

```go
bus := flows.NewEventBus()

flows.Subscribe[flows.ToolExecutionContext](bus, flows.ToolExecutionEnd,
    func(ctx context.Context, event *flows.Event[flows.ToolExecutionContext]) error {
        log.Printf("tool %s completed in %s", event.Data.ToolName, event.Data.Duration)
        return nil
    },
)

// Pass the bus to flows
messageFlow := flows.NewHandleMessageFlow(g, store,
    flows.WithEventBus(bus),
)
heartbeat := flows.NewHeartbeat(g, store, cfg,
    flows.WithHeartbeatEventBus(bus),
)
```

## Architecture

The framework is built around four pillars:

```
┌─────────────────────────────────────────────────┐
│                 genkit-cowork                    │
│                                                  │
│   ┌───────────┐  ┌───────────┐                   │
│   │   Flows   │  │   Tools   │                   │
│   │           │  │           │                   │
│   │  agent    │  │  bash     │                   │
│   │  loop     │  │  read     │                   │
│   │  message  │  │  edit     │                   │
│   │  heartbeat│  │  write    │                   │
│   │  reply    │  │           │                   │
│   └───────────┘  └───────────┘                   │
│                                                  │
│   ┌───────────┐  ┌───────────┐                   │
│   │  Memory   │  │  Skills   │                   │
│   │           │  │           │                   │
│   │  sessions │  │  discover │                   │
│   │  retrieval│  │  list     │                   │
│   │  recall   │  │  resolve  │                   │
│   └───────────┘  └───────────┘                   │
│                                                  │
│          ┌─────────────────┐                     │
│          │  Genkit Runtime  │                     │
│          └─────────────────┘                     │
└─────────────────────────────────────────────────┘
```

Each pillar can be adopted independently. Use the full framework or pick individual pieces:

- **Tools only** — register `NewBashTool`, `NewReadTool`, `NewEditTool`, `NewWriteTool` with any Genkit instance.
- **Flows only** — use the agent loop, message handling, heartbeat, or reply flows.
- **Skills only** — register the `Skills` plugin to discover and serve domain knowledge.
- **Mix and match** — combine pillars based on your use case.

## Flows

| Flow | Registration | Purpose |
|------|-------------|---------|
| `agentLoop` | `genkit.NewFlow` (internal) | Core model/tool turn loop |
| `handleMessage` | `genkit.DefineFlow` | Session-backed chat |
| `heartbeat` | `genkit.DefineFlow` | Scheduled background monitoring |
| `sendReply` | `genkit.DefineFlow` | Channel-routed reply delivery |

## Tools

| Tool | Constructor | Description |
|------|------------|-------------|
| Bash | `NewBashTool(g, cwd, ...opts)` | Shell command execution with timeout, env, spawn hooks |
| Read | `NewReadTool(g, cwd, ...opts)` | File/image reading with pagination, truncation, auto-resize |
| Edit | `NewEditTool(g, cwd, ...opts)` | Find-and-replace with exact/fuzzy matching, unified diff |
| Write | `NewWriteTool(g, cwd, ...opts)` | File creation with auto-mkdir, operator interface |

## Skills

Skills are domain-specific knowledge modules discovered from a directory of `SKILL.md` files. The skills system is implemented as a Genkit plugin.

### Registration

```go
import "github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/skills"

g, _ := genkit.Init(ctx,
    genkit.WithDefaultModel("googleai/gemini-2.0-flash"),
    genkit.WithPlugins(&skills.Skills{SkillsDir: "./skills"}),
)
```

### Skill Format

Each skill lives in a subdirectory and must contain a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: my-skill
description: What the skill does
license: MIT
metadata:
  key: value
---
Skill content in Markdown...
```

### Tools

The plugin provides two tools for agent use:

| Tool | Name | Description |
|------|------|-------------|
| `ListSkillsTool(g)` | `list-skills` | Returns discovered skills with optional name filter |
| `ResolveSkillTool(g)` | `resolve-skill` | Loads the full Markdown body and metadata for a skill by name |

### Discovery

On `Init`, the plugin scans top-level subdirectories of `SkillsDir` for `SKILL.md` files, parses frontmatter, validates required fields (`name`, `description`), and catalogs all files in the skill directory. Invalid skills are silently skipped. Skill body content is loaded lazily only when `resolve-skill` is called.

## Design Patterns

- **Functional options** — all constructors accept variadic option functions for clean, extensible configuration.
- **Operator interfaces** — `BashOperator`, `ReadOperator`, `EditOperator`, `WriteOperator`, `SessionOperator`, `AgentLoopOperator` abstract I/O behind interfaces for testability and sandboxing.
- **Hook system** — lifecycle hooks (spawn hooks, event bus) allow interception and mutation before operations run.

## Package Layout

```
genkit-cowork/
├── flows/              # Flow definitions
│   ├── agent_loop.go         # Core model/tool execution loop
│   ├── message.go            # Session-backed message handling
│   ├── heartbeat.go          # Scheduled heartbeat runner
│   ├── heartbeat_config.go   # Heartbeat configuration types
│   ├── heartbeat_result.go   # Result parsing and classification
│   ├── reply.go              # Channel-routed reply delivery
│   ├── event.go              # EventBus and typed lifecycle events
│   └── event_context.go      # Event payload types
├── tools/              # Tool definitions
│   ├── bash.go               # Shell command execution
│   ├── read.go               # File/image reading
│   ├── edit.go               # Find-and-replace editing
│   ├── write.go              # File creation with auto-mkdir
│   ├── edit-diff.go          # Text normalization, fuzzy matching, diffs
│   ├── diff.go               # LCS-based line diff algorithm
│   ├── truncate.go           # Output truncation utilities
│   ├── path.go               # Path resolution utilities
│   └── constants.go          # Output limits
├── plugins/            # Genkit plugins
│   └── skills/               # Skill discovery and serving
│       ├── skills.go          # Plugin struct, Init, tool registration
│       ├── skill_parser.go    # SKILL.md frontmatter parsing
│       └── skill_scanner.go   # Directory scanning, skill discovery
├── media/              # Image detection and processing
│   └── mime.go               # MIME detection, image resizing
├── memory/             # Session persistence
│   └── sessions.go           # Session store, message origins
└── utils/              # Shared utilities
    └── shell.go              # Shell environment management
```
