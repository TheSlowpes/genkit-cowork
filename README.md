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
│   │  file     │  │  list     │                   │
│   │  vector   │  │  resolve  │                   │
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
- **Memory only** — use `NewSession` with in-memory, file-backed, or vector-augmented operators.
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

## Memory

Memory is implemented through a `Session` store plus pluggable `SessionOperator` backends.

### Core types

| Type | Constructor / API | Description |
|------|-------------------|-------------|
| Session store | `NewSession(...opts)` | Implements `session.Store[SessionState]` for Genkit flows |
| Persistence mode | `WithPersistenceMode(mode, n)` | Load behavior: `All`, `SlidingWindow`, `TailEndsPruning`, `TokenBudget` |
| Media asset store | `WithMediaAssetStore(store)` | Normalizes media data URI parts into persisted files and tracks `SessionAsset` metadata |
| Tenant binding | `WithTenantID(id)` / `ForTenant(id)` | Scopes `Get`/`Save` operations to a tenant in a `session.Store`-compatible API |
| Ledger validation | `ValidateSessionLedger(state)` | Validates append-only sequencing and immutable-prefix constraints for a session ledger |
| Replay window | `MessagesForTurn(state, turn)` | Reconstructs the exact message slice represented by a persisted turn sequence range |
| In-memory backend | default (`defaultSessionOperator`) | Process-local map-based state storage |
| File backend | `NewFileSessionOperator(rootDir)` | Durable JSON state at `rootDir/{sessionID}/state.json` |
| Vector wrapper | `NewVectorOperator(base, backend, rootDir)` | Wraps a base operator and indexes new messages for semantic retrieval |
| Local vector backend | `NewLocalVecBackend(g, name, cfg)` | `localvec`-based implementation of `VectorBackend` |
| File records backend | `NewFileRecordOperator(rootDir)` | Durable tenant-global file and chunk records under `rootDir/{tenantID}/files` |
| File ingest service | `NewFileIngestService(files, blobs, extractor, indexer)` | Stores tenant files, extracts text from supported MIME types, chunks, and indexes for recall |
| File blob store | `NewFileBlobDiskStore(rootDir)` | Stores raw tenant file bytes under `rootDir/{tenantID}/files/raw` |
| Insight backend | `NewFileInsightOperator(rootDir)` | Durable tenant-scoped consolidation run ledger and immutable insight records |
| Preference backend | `NewFilePreferenceOperator(rootDir)` | Durable tenant-scoped explicit and implicit preference records |
| Consolidation service | `NewConsolidationService(...)` | Runs tenant consolidation, derives insights (LLM deriver), persists run/insights, and optionally indexes insights |

### File-backed sessions

`NewFileSessionOperator` provides durable session state with per-session locks, atomic writes, append-only validation, and tenant consistency checks.

```go
fileOp := memory.NewFileSessionOperator("./data/sessions")
store := memory.NewSession(
    memory.WithCustomSessionOperator(fileOp),
    memory.WithPersistenceMode(memory.SlidingWindow, 100),
)
```

### Vector-augmented retrieval

`VectorOperator` composes on top of a base `SessionOperator`. It indexes new messages by `messageID` and supports semantic lookup with `Search`.

```go
fileOp := memory.NewFileSessionOperator("./data/sessions")

vecBackend, _ := memory.NewLocalVecBackend(g, "session-memory", memory.LocalVecConfig{
    Embedder: embedder, // any ai.Embedder
})

vecOp := memory.NewVectorOperator(fileOp, vecBackend, "./data/sessions", "tenant-1")

store := memory.NewSession(
    memory.WithCustomSessionOperator(vecOp),
    memory.WithTenantID("tenant-1"),
)

results, err := vecOp.Search(ctx, "tenant-1", "session-1", "customer asked about invoice", 5)
_ = results
_ = err
```

### Tenant-global file memory (text + structured)

`FileIngestService` adds cross-session tenant recall for uploaded files. It currently supports text and structured formats: `text/plain`, `text/markdown`, `application/json`, `text/csv`, and `text/html`.

```go
fileRecords := memory.NewFileRecordOperator("./data/memory")
blobStore := memory.NewFileBlobDiskStore("./data/memory")
indexer := memory.NewVectorFileIndexer(vecBackend)

ingest := memory.NewFileIngestService(fileRecords, blobStore, nil, indexer)

ingested, err := ingest.Ingest(ctx, memory.FileIngestInput{
    TenantID:  "tenant-1",
    SessionID: "session-a",
    FileName:  "./docs/policy.md",
    Data:      []byte("# Policy\nInvoices close monthly."),
})
_ = ingested
_ = err

chunks, err := ingest.SearchTenantFiles(ctx, memory.FileChunkSearchInput{
    TenantID: "tenant-1",
    Query:    "invoice policy",
    TopK:     5,
})
_ = chunks
_ = err
```

### Consolidation (nightly enrichment)

`ConsolidationService` derives immutable tenant insights from sessions and tenant-global files. The default LLM deriver expects a model already registered in the same Genkit instance.

```go
insightStore := memory.NewFileInsightOperator("./data/memory")
deriver := memory.NewLLMInsightDeriver(g, "googleai/gemini-2.0-flash")
insightIndexer := memory.NewVectorInsightIndexer(vecBackend)

consolidation := memory.NewConsolidationService(
    sessionOp,
    fileRecords,
    insightStore,
    memory.NewFilePreferenceOperator("./data/memory"),
    deriver,
    insightIndexer,
    memory.ConsolidationConfig{
        Model:                         "googleai/gemini-2.0-flash",
        PromptVersion:                 "v1",
        PreferencePromotionConfidence: 0.8,
    },
)

run, err := consolidation.RunTenant(ctx, "tenant-1")
_ = run
_ = err

results, err := consolidation.SearchTenantInsights(ctx, "tenant-1", "invoice policy", 5)
_ = results
_ = err
```

### Tenant preferences

Preferences are first-class tenant records and can be explicit (user provided) or implicit (promoted from consolidation preference candidates with confidence/provenance).

```go
prefs := memory.NewFilePreferenceOperator("./data/memory")

saved, err := prefs.SavePreference(ctx, "tenant-1", memory.PreferenceRecord{
    Key:        "response_style",
    Value:      "concise",
    Source:     memory.PreferenceSourceExplicit,
    Confidence: 1,
})
_ = saved
_ = err

all, err := prefs.ListPreferences(ctx, "tenant-1", memory.PreferenceFilter{Status: memory.PreferenceStatusActive})
_ = all
_ = err
```

### Stored message model

Each `SessionMessage` stores `MessageID`, `Origin`, `Kind`, `Content`, and `Timestamp`.

- `Kind` is auto-derived when missing: tool-role messages become `instrumental`, others default to `episodic`.
- Additional kinds (`semantic`, `procedural`) are available for higher-level memory workflows.
- `Session.Save` auto-fills missing `MessageID` and `Timestamp`.

### Examples

- `examples/pgvector/main.go` shows the most basic pgvector wiring by wrapping the Genkit PostgreSQL plugin as a `memory.VectorBackend` and plugging it into `memory.NewVectorOperator`.

## Skills

Skills are domain-specific knowledge modules discovered from a directory of `SKILL.md` files. The skills system is implemented as a Genkit plugin.

### Registration

```go
import "github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/skills"

g, _ := genkit.Init(ctx,
    genkit.WithDefaultModel("googleai/gemini-2.0-flash"),
    genkit.WithPlugins(&skills.Skills{
        SkillsDir:     "./skills", // optional; falls back to default search paths
        AllowedSkills: []string{"my-skill"}, // optional whitelist; all skills exposed when empty
    }),
)
```

When `SkillsDir` is not set, the plugin searches for the first existing directory among: `./skills`, `./SKILLS`, `./.agent/skills`, `./agent/skills`, `./docs/skills`. If none are found, the plugin starts with an empty skill set and does not panic.

After `Init`, register the tool with a Genkit instance:

```go
s := &skills.Skills{SkillsDir: "./skills"}
// after genkit.Init with the plugin...
skillTool := s.SkillTool(g)
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

The plugin provides a single tool for agent use:

| Tool | Name | Description |
|------|------|-------------|
| `SkillTool(g)` | `resolve-skill` | Lists all available skills in the tool description; accepts a skill name and returns the full SKILL.md body and metadata |

### Discovery

On `Init`, the plugin resolves the skills directory (checking `defaultSkillsDirs` when `SkillsDir` is unset), then scans top-level subdirectories for `SKILL.md` files, parses frontmatter, validates required fields (`name`, `description`), and catalogs all files in the skill directory (including one level of subdirectories). Invalid skills are silently skipped. Skill body content is loaded lazily only when `resolve-skill` is called.

When `AllowedSkills` is non-empty, only skills whose names appear in that list are exposed by `SkillTool` and `ListSkills`.

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
│   ├── consolidation.go      # Optional consolidation flow wrapper
│   ├── reply.go              # Channel-routed reply delivery
│   ├── event.go              # EventBus and typed lifecycle events
│   └── event_context.go      # Event payload types
├── tools/              # Tool definitions
│   ├── bash.go               # Shell command execution
│   ├── read.go               # File/image reading
│   ├── edit.go               # Find-and-replace editing
│   ├── write.go              # File creation with auto-mkdir
│   ├── edit_diff.go          # Text normalization, fuzzy matching, diffs
│   ├── diff.go               # LCS-based line diff algorithm
│   ├── truncate.go           # Output truncation utilities
│   ├── path.go               # Path resolution utilities
│   ├── constants.go          # Output limits
│   ├── memory_retrieval.go   # Tenant/session memory retrieval tools
│   ├── file_memory.go        # Tenant-global file memory tools
│   └── insight_retrieval.go  # Tenant insight retrieval tool
├── plugins/            # Genkit plugins
│   └── skills/               # Skill discovery and serving
│       ├── skills.go          # Plugin struct, Init, tool registration
│       ├── skill_parser.go    # SKILL.md frontmatter parsing
│       └── skill_scanner.go   # Directory scanning, skill discovery
├── media/              # MIME detection and processing
│   ├── mime.go               # Image MIME detection, image resizing
│   └── text_extract.go       # Text/structured MIME extraction (txt/md/json/csv/html)
├── memory/             # Session persistence and retrieval
│   ├── sessions.go           # Session store, message models, persistence modes
│   ├── turns.go              # Turn ledger records and ledger validation
│   ├── snapshots.go          # State snapshot records and checksum support
│   ├── assets.go             # Session asset model and media asset store interface
│   ├── file_assets.go        # Filesystem media asset store implementation
│   ├── file_sessions.go      # File-backed SessionOperator (JSON + atomic write)
│   ├── files.go              # Tenant-global file + chunk records and interfaces
│   ├── file_files.go         # File-backed FileOperator for file/chunk metadata
│   ├── file_blob_store.go    # Raw tenant file blob store
│   ├── file_ingest.go        # MIME-aware file ingest, chunking, indexing, recall service
│   ├── file_recall.go        # Tenant file recall helper API
│   ├── insights.go           # Insight records + consolidation run ledger operators
│   ├── consolidation.go      # Tenant consolidation service and LLM deriver
│   ├── preferences.go        # Tenant preference records and preference operators
│   ├── vector_sessions.go    # VectorOperator wrapper + semantic search
│   └── vector_backend.go     # Vector backend interface + localvec backend
└── utils/              # Shared utilities
    └── shell.go              # Shell environment management
```
