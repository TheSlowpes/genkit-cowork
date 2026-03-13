# genkit-cowork

A coworking framework for [Firebase Genkit](https://github.com/firebase/genkit) (Go) that gives AI agents pluggable capabilities for autonomous and interactive work. The framework is built around four pillars — **Flows**, **Tools**, **Memory**, and **Skills** — each of which can be registered with a Genkit application independently or as a complete suite.

## Architecture

### Pillars

```
┌─────────────────────────────────────────────────┐
│                 genkit-cowork                    │
│                                                  │
│   ┌───────────┐  ┌───────────┐                   │
│   │   Flows   │  │   Tools   │                   │
│   │           │  │           │                   │
│   │  hooks    │  │  bash     │                   │
│   │  schedule │  │  read     │                   │
│   │  loop     │  │  edit     │                   │
│   │  chat     │  │  write    │                   │
│   └───────────┘  └───────────┘                   │
│                                                  │
│   ┌───────────┐  ┌───────────┐                   │
│   │  Memory   │  │  Skills   │                   │
│   │           │  │           │                   │
│   │  sessions │  │  org      │                   │
│   │  retrieval│  │  industry │                   │
│   │  recall   │  │  domain   │                   │
│   └───────────┘  └───────────┘                   │
│                                                  │
│          ┌─────────────────┐                     │
│          │  Genkit Runtime  │                     │
│          └─────────────────┘                     │
└─────────────────────────────────────────────────┘
```

### 1. Flows

Flows define how agents execute work. The current implementation in `genkit-cowork/flows` includes:

**Agent Loop (`agentLoop`)** — A model-driven turn loop that generates model responses, executes tool requests, appends tool responses, and continues until no more tool requests are returned. Supports max-turn limits, tool interrupt handling, and resume inputs (`toolResponses`, `toolRestarts`).

**Message Flow (`handleMessage`)** — Session-backed chat-style flow that loads or creates session state, appends incoming message content, runs the agent loop, and persists newly generated messages.

**Heartbeat Flow (`heartbeat`)** — Periodic/background flow that runs the agent loop against current session state and classifies results as `ack`, `alert`, `skipped`, or `error` based on heartbeat response parsing.

**Flow Events (`EventBus`)** — Typed event lifecycle with synchronous subscribers for `agent`, `turn`, `message`, and `tool-execution` stages. `tool-execution-start` handlers can mutate tool input before execution.

### 2. Tools

Tools define what agents can do. Each tool is registered with a Genkit instance and follows the functional options pattern for configuration and the operator/strategy interface pattern for testability and sandboxing.

**Bash** — Execute shell commands with configurable timeout, working directory, environment, and spawn hooks. Custom operators can be injected to sandbox or mock execution. See `tools/bash.go`.

**Read** — Read file contents (text and images) with line-offset pagination and output truncation. Supports image MIME detection and auto-resizing. See `tools/read.go`.

**Edit** — Find-and-replace editing with exact and fuzzy text matching, uniqueness validation, and unified diff output. See `tools/edit.go`.

**Write** — Create new files. Planned.

Tools are created via constructor functions that return `ai.Tool`, which integrates directly with the Genkit tool system:

```go
bashTool := tools.NewBashTool(g, cwd, tools.WithCommandPrefix("..."))
readTool := tools.NewReadTool(g, cwd, tools.WithCustomReadOperator(op))
editTool := tools.NewEditTool(g, cwd)
```

### 3. Memory

Memory defines what agents remember across and within sessions. The memory system uses a combination of RAG (Retrieval-Augmented Generation) for semantic search and markdown files for human-readable, editable state.

**Sessions** — Markdown-based persistence of conversation and session state. Sessions capture the interaction history and working context so an agent can resume where it left off or hand off to another agent.

**Retrieval** — RAG-powered search over knowledge bases and accumulated context. Retrieval enables agents to find relevant information from large bodies of prior work, documentation, or domain knowledge using vector similarity.

**Recall** — Structured markdown files that store key decisions, facts, patterns, and lessons learned. Recall provides a persistent, inspectable record that agents can reference for consistency and that humans can review and edit directly.

The combination of RAG and markdown ensures that memory is both semantically searchable by agents and transparently readable by humans.

### 4. Skills

Skills define what agents know. The skill system injects additional domain-specific context into agent capabilities, augmenting their behavior with specialized competence.

Skills are pluggable modules that provide structured knowledge for particular areas of work. The framework prioritizes skills that assist in workplace scenarios — organizational context, industry-specific knowledge, and domain expertise.

Skills are loaded and composed alongside the other pillars, allowing an agent to be configured with exactly the competencies required for its role.

## Pluggability

The framework is designed so that each pillar can be registered with a Genkit application independently. A consumer can adopt the full framework or pick individual pieces:

**Full framework** — Register all four pillars to get a fully capable coworking agent.

**Individual tools** — Register only the tools you need. Each tool constructor (`NewBashTool`, `NewReadTool`, etc.) returns a standalone `ai.Tool` that works with any Genkit instance.

**Individual pillars** — Mix and match flows, tools, memory, and skills based on the use case. An application might use only the tools and memory pillars without flows or skills, or use chat flows with a custom tool set.

All constructors accept functional options for configuration and operator/strategy interfaces for swapping implementations, making each component independently testable and customizable.

## Codebase

### Package Layout

```
genkit-cowork/
├── flows/           # Flow definitions (agent loop, message, heartbeat, events)
│   ├── agent_loop.go      # Core model/tool execution loop
│   ├── message.go         # Session-backed message handling flow
│   ├── heartbeat.go       # Scheduled/background heartbeat flow wrapper
│   ├── event.go           # Event bus and typed lifecycle events
│   ├── event_context.go   # Event payload/context types
│   ├── heartbeat_config.go # Heartbeat scheduling/delivery config
│   └── heartbeat_result.go # Heartbeat result parsing and classification
├── tools/           # Tool definitions (bash, read, edit, write)
│   ├── bash.go      # Bash command execution tool
│   ├── read.go      # File/image reading tool
│   ├── edit.go      # Find-and-replace editing tool
│   ├── edit-diff.go  # Text normalization, fuzzy matching, diff formatting
│   ├── diff.go       # LCS-based line diff algorithm
│   ├── truncate.go  # Output truncation utilities
│   ├── path.go      # Path resolution utilities
│   └── constants.go # Output truncation limits
├── media/           # Image detection and processing
│   └── mime.go      # MIME type detection, image resizing
└── utils/           # Shared utilities
    └── shell.go     # Shell environment management
```

### Design Patterns

**Functional Options** — All tool constructors accept variadic option functions (`BashToolOption`, `ReadToolOption`) that mutate a private options struct. This keeps the API clean while allowing extensive configuration.

**Operator Interfaces** — `BashOperator`, `ReadOperator`, and `EditOperator` abstract the actual I/O operations behind interfaces. Default implementations are provided, but consumers can inject custom operators for sandboxing, testing, or alternative execution environments. All operator methods that perform I/O accept `context.Context` for cooperative cancellation.

**Hook System** — Hooks (e.g., `BashSpawnHook`) allow callers to intercept and modify execution context before an operation runs. This pattern will extend to flows and other pillars.

### Current State

| Component | Status |
|---|---|
| `flows/agent_loop.go` — model/tool turn loop, interrupts, resume support | Implemented |
| `flows/message.go` — session-backed message flow over agent loop | Implemented |
| `flows/heartbeat.go` — periodic/background heartbeat flow | Implemented |
| `flows/event.go` + `flows/event_context.go` — typed flow lifecycle events | Implemented |
| `flows/heartbeat_config.go` + `flows/heartbeat_result.go` — heartbeat config and result classification | Implemented |
| `tools/bash.go` — command execution with spawn hooks | Implemented |
| `tools/read.go` — text file reading with offset/limit, line-number prefixing, truncation | Implemented |
| `tools/read.go` — image reading with auto-resize (JPEG, PNG, GIF, WebP) | Implemented |
| `tools/truncate.go` — output truncation (line + byte limits) | Implemented |
| `tools/path.go` — path resolution (cwd-relative, ~ expansion, OS-agnostic) | Implemented |
| `media/mime.go` — MIME detection and image auto-resize (CatmullRom scaling) | Implemented |
| `utils/shell.go` — shell environment | Implemented |
| `tools/edit.go` — find-and-replace with fuzzy matching, BOM/line-ending preservation | Implemented |
| `tools/edit-diff.go` — text normalization, fuzzy matching, unified diff generation | Implemented |
| `tools/diff.go` — LCS-based line diff algorithm | Implemented |
| Tools: Write | Planned |
| Memory | Planned |
| Skills | Planned |

## Development

### Conventions

- Go module path: `github.com/TheSlowpes/genkit-cowork`
- Primary dependency: `github.com/firebase/genkit/go v1.4.0`
- Use functional options for all public constructors
- Define operator interfaces for any I/O or side-effecting operations
- Keep tool handler logic in the `tools` package; supporting utilities in `media` and `utils`
- All operator interface methods that perform I/O accept `context.Context` as their first parameter for cooperative cancellation
- `main.go` is gitignored and used for local testing only
