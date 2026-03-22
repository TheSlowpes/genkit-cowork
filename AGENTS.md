# genkit-cowork

A coworking framework for [Firebase Genkit](https://github.com/firebase/genkit) (Go) that gives AI agents pluggable capabilities for autonomous and interactive work. The framework is built around four pillars — **Flows**, **Tools**, **Memory**, and **Skills** — each of which can be registered with a Genkit application independently or as a complete suite.

## Agent Instructions

As an AI eveloper working on this codebase, you MUST adhere to the operational protocols:

- **Registry-First Thinking:** Genkit is built round a central registry. Always use `genkit.DefineX`(e.g. `DefineFlow`, `DefineTool`) for any component that should be discoverble by tools or the Dev UI.
- **Context is mandatory:** Every Genkit function and tool execution requires a `context.Context`. Pass it through faithfully; never use `contex.Background()` inside n implementation unless it is the entry point.
- **Schema Strictness:** When defining input/output types for Flows or Tools, use clear, JSON-tagged Go structs. The model uses these tags and types to understand the interface.
- **Search Before Implementing:** Many common utilities exist in the genkit repository, Check there before implementing new JSON parsing or string manipulation logic.
- **Idiomatic Concurrency:**Prefer Go's native concurrency (goroutines/channels) but be mindful of the `context` lifecycle within long-running background tasks.

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

#### Defining a Flow

Use `genkit.DefineFlow` for orchestration tasks. Use typed structs or input and output to ensure schema generation.

```go
type OrderInput struct {
	ID int `json:"id"`
}

type OrderOutput struct {
	Status string `json:"status"`
}

// Define the flow
var GetOrderStatusFlow = genkit.DefineFlow(g, "getOrderStatus",
	func(ctx context.Context, input *OrderInput) (*OrderOutput, error) {
		// Use genkit.Run to create a trace span for a specific step
		status, err := genkit.Run(ctx, "lookup-db", func() (string, error) {
			return "Shipped", nil 
		})
		if err != nil {
			return nil, err
		}
		return &OrderOutput{Status: status}, nil
	},
)
```

In the genkit-cowork framework, most flows will use the agentLoop flow as its core.

```go
agentLoop := NewAgentLoop(
				g,
				WithEventBus(options.bus),
				WithCustomGenerateOptions(options.baseOpts...),
				WithCustomAgentLoopOperator(options.loopOperator),
			)

loopOutput, err := agentLoop.Run(ctx, loopInput)
if err != nil {
	return nil, fmt.Errorf("agent loop: %w", err)
}
```


### 2. Tools

Tools define what agents can do. Each tool is registered with a Genkit instance and follows the functional options pattern for configuration and the operator/strategy interface pattern for testability and sandboxing.

**Bash** — Execute shell commands with configurable timeout, working directory, environment, and spawn hooks. Custom operators can be injected to sandbox or mock execution. See `tools/bash.go`.

**Read** — Read file contents (text and images) with line-offset pagination and output truncation. Supports image MIME detection and auto-resizing. See `tools/read.go`.

**Edit** — Find-and-replace editing with exact and fuzzy text matching, uniqueness validation, and unified diff output. See `tools/edit.go`.

**Write** — Create or overwrite files with automatic parent directory creation. Custom operators can be injected for sandboxing or testing. See `tools/write.go`.

#### Defining a Tool

Tools are used by models. The description is crucial as the model uses it to decide when to call the tool.

```go
type WeatherInput struct {
	Location string `json:"location"`
}

var WeatherTool = genkit.DefineTool(g, "getWeather", "fetches current weather for a location",
	func(ctx *ai.ToolContext, input *WeatherInput) (string, error) {
		// Implementation logic
		return "Sunny", nil
	},
)
```

Tools are created via constructor functions that return `ai.Tool`, which integrates directly with the Genkit tool system:

```go
bashTool := tools.NewBashTool(g, cwd, tools.WithCommandPrefix("..."))
readTool := tools.NewReadTool(g, cwd, tools.WithCustomReadOperator(op))
editTool := tools.NewEditTool(g, cwd)
writeTool := tools.NewWriteTool(g, cwd)
```

### 3. Memory

Memory defines what agents remember across and within sessions. The memory system uses a combination of RAG (Retrieval-Augmented Generation) for semantic search and markdown files for human-readable, editable state.

**Sessions** — Markdown-based persistence of conversation and session state. Sessions capture the interaction history and working context so an agent can resume where it left off or hand off to another agent.

**Retrieval** — RAG-powered search over knowledge bases and accumulated context. Retrieval enables agents to find relevant information from large bodies of prior work, documentation, or domain knowledge using vector similarity.

**Recall** — Structured markdown files that store key decisions, facts, patterns, and lessons learned. Recall provides a persistent, inspectable record that agents can reference for consistency and that humans can review and edit directly.

The combination of RAG and markdown ensures that memory is both semantically searchable by agents and transparently readable by humans.

### 4. Skills

Skills define what agents know. The skill system is implemented as a Genkit plugin (`plugins/skills/`) that discovers, parses, and serves domain-specific knowledge from `SKILL.md` files.

**Plugin Registration** — The `Skills` struct implements `api.Plugin` and is registered via `genkit.WithPlugins()`. On `Init`, it resolves `SkillsDir` (trying `./skills`, `./SKILLS`, `./.agent/skills`, `./agent/skills`, `./docs/skills` when unset) and scans it for subdirectories containing `SKILL.md` files with YAML frontmatter (`name`, `description`, optional `license` and `metadata`). When `AllowedSkills` is non-empty, only the named skills are exposed.

**Discovery** — `discoverSkills` walks top-level subdirectories, parses each `SKILL.md` via `parseSkillMetadata`, validates required fields, and catalogs all files in the skill directory (including one level of subdirectories). Invalid skills are silently skipped.

**Tools** — The plugin provides a single tool:
- `SkillTool(g)` — Lists all available skills (name + description) in the tool description and accepts a skill name to load the full Markdown body and metadata. Tool name: `resolve-skill`.

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
├── flows/           # Flow definitions (agent loop, message, heartbeat, reply, events)
│   ├── agent_loop.go      # Core model/tool execution loop
│   ├── message.go         # Session-backed message handling flow
│   ├── heartbeat.go       # Scheduled/background heartbeat flow wrapper
│   ├── reply.go           # Channel-routed reply delivery flow
│   ├── event.go           # Event bus and typed lifecycle events
│   ├── event_context.go   # Event payload/context types
│   ├── heartbeat_config.go # Heartbeat scheduling/delivery config
│   └── heartbeat_result.go # Heartbeat result parsing and classification
├── tools/           # Tool definitions (bash, read, edit, write)
│   ├── bash.go      # Bash command execution tool
│   ├── read.go      # File/image reading tool
│   ├── edit.go      # Find-and-replace editing tool
│   ├── write.go     # File creation with auto-mkdir
│   ├── edit-diff.go  # Text normalization, fuzzy matching, diff formatting
│   ├── diff.go       # LCS-based line diff algorithm
│   ├── truncate.go  # Output truncation utilities
│   ├── path.go      # Path resolution utilities
│   └── constants.go # Output truncation limits
├── plugins/         # Genkit plugins
│   └── skills/      # Skill discovery and serving
│       ├── skills.go        # Plugin struct, Init, tool registration
│       ├── skill_parser.go  # SKILL.md frontmatter parsing
│       └── skill_scanner.go # Directory scanning, skill discovery
├── media/           # Image detection and processing
│   └── mime.go      # MIME type detection, image resizing
├── memory/          # Session persistence
│   └── sessions.go  # Session store, message origins
└── utils/           # Shared utilities
    └── shell.go     # Shell environment management
```

### Design Patterns

**Functional Options** — All tool constructors accept variadic option functions (`BashToolOption`, `ReadToolOption`) that mutate a private options struct. This keeps the API clean while allowing extensive configuration.

**Operator Interfaces** — `BashOperator`, `ReadOperator`, `EditOperator`, and `WriteOperator` abstract the actual I/O operations behind interfaces. Default implementations are provided, but consumers can inject custom operators for sandboxing, testing, or alternative execution environments. All operator methods that perform I/O accept `context.Context` for cooperative cancellation.

**Hook System** — Hooks (e.g., `BashSpawnHook`) allow callers to intercept and modify execution context before an operation runs. This pattern will extend to flows and other pillars.

**Constructors:**
    `DefineX` (e.g., `DefineFlow`, `DefineTool`): Creates and registers a component with the registry. Use this for components that should be discoverable by the reflection API (and thus the Dev UI).
    `NewX` (e.g., `NewTool`): Creates a component without registering it. Use this for internal components, dynamic creation, or testing.

### Code Quality & Linting

- **Run Linting**: Always run `go vet ./...` from the `go/` directory for all Go code changes.
- **Format Code**: Run `bin/fmt` (which runs `go fmt`) to ensure code is formatted correctly.
- **Pass All Tests**: Ensure all unit tests pass (`go test ./...`).
- **Production Ready**: The objective is to produce production-grade code.
- **Shift Left**: Employ a "shift left" strategy—catch errors early.
- **Strict Typing**: Go is statically typed. Do not use `interface{}` (or `any`) unless absolutely necessary and documented.
- **No Warning Suppression**: Avoid ignoring linter warnings unless there is a compelling, documented reason.
- Group imports: standard library first, then third-party, then internal. `goimports` handles this automatically.

### Detailed Coding Guidelines

#### Target Environment

- **Go Version:** Target Go 1.25 or newer.
- **Environment:** Use `go mod` for dependency management.

#### Typing and Style

- **Syntax:**
    - Use standard Go formatting (`gofmt`).
    - Use idioms like `if err != nil` for error handling.
    - Prefer short variable names for short scopes (e.g. `i` for index, `ctx` for context).
- **Concurrency:** Use channels and goroutines for concurrency. Avoid shared mutable state where possible.
- **Comments:**
    - Use proper punctuation.
    - Start function comments with the function name
    - Use `// TODO (issue-id): fix this later.` format for stubs.
- Ensure that `go vet` passes without errors.

#### Documentation

- **Format:** Write comprehensive GoDoc-style comments for exported packages, types, and functions.
- **Content:**
    - **Explain Concepts:** Explain the terminology and concepts used in the code to someone unfamiliar with the code.
    - **Visuals:** Prefer using diagrams if helpful to explain complex flows.
- **Required Sections:**
    - **Overview:** Description of what the package/function does.
    - **Examples:** Provide examples for complex APIs (using `Example` functions in `_test.go` files is best practice).
- **External Packages:** Use the `go doc` command to understand type and function definitions when working with external packages.
- **References:** Please use the descriptions from genkit.dev and github.com/genkit-ai/docsite as the source of truth for the API and concepts.
- Keep examples in documentation and comments simple.
- Add links to relevant documentation on the Web or elsewhere in the relevant places in comments.
- Always update package comments and function comments when updating code.
- Scan documentation for every package you edit and keep it up-to-date.

#### Implementation

- Always add unit tests to improve coverage. Use Genkit primitives and helper functions instead of mocking types.
- Always add/update samples to demonstrate the usage of the API or functionality.
- Use default input values for flows and actions to make them easier to use.
- In the samples, explain the whys, hows, and whats of the sample in the package comment so the user learns more about the feature being demonstrated. Also explain how to test the sample.
- Avoid mentioning sample specific stuff in core framework or plugin code.
- Always check for missing dependencies in go.mod for each sample and add them if we're using them.
- When a plugin is updated or changes, please also update relevant documentation and samples.
- Avoid boilerplate comments in the code. Comments should tell why, not what.
- Always update the README.md (if exists) to match the updated code changes.
- Make sure to not leave any dead code or unused imports.

#### Formatting

- **Tool:** Format code using `go fmt`.
- **Line Length:** Go doesn't have a strict line length limit, but keep it reasonable (e.g. 80-100 characters).
- **Strings:** Wrap long lines and strings appropriately.

#### Testing

- **Framework:** Use the standard `testing` package.
- **Assertions**: Use plain `if/else` blocks following the `want`/`got` pattern.
    - Example:
```go
    if got := func(); got != want {
      t.Errorf("func() = %v, want %v", got, want)
    }
```

- For complex object comparisons (structs, slices, maps), use `github.com/google/go-cmp/cmp` (and `cmpopts` if needed).

```go
    if diff := cmp.Diff(want, got); diff != "" {
      t.Errorf("mismatch (-want +got):\n%s", diff)
    }
```

- **Scope**: Write comprehensive unit tests following the fail-fast approach.
- **Execution**: Run via `go test ./...`.
- **Porting**: Maintain 1:1 logic parity accurately if porting tests. Do not invent behavior.
- **Fixes**: Fix underlying code issues rather than special-casing tests.
- **Modernize**: Consider using `modernize` to update code to modern Go idioms (e.g., `slices`, `maps`) when fixing underlying issues.
- **Genkit Testing**:
  - **Test Actions Directly**: Use `flow.Run(ctx, input)` or `tool.Run(ctx, input)` to test the logic of your Genkit components.
  - **Verify Schemas**: Ensure that the `Action` returned by `DefineX` has the expected input and output schemas.
  - **Mock Models**: When testing Flows that call models, use a mock model implementation to ensure deterministic tests.
  - **Trace Inspection**: For complex flows, use tests to verify that `genkit.Run` steps are being executed as expected.

#### Logging

- **Library**: Use `log/slog` (available in Go 1.21+) or the internal logger.
- **Format**: Use structured logging keys and values.

#### Licensing

Include the Apache 2.0 license header at the top of each file (update year as needed):

```go
// Copyright [year] Google LLC
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
```

#### Git commit message guidelines

- Please draft a plain-text commit message after you're done with changes.
- Please do not include absolute file paths as links in commit messages.
- Since lines starting with `#` are treated as comments, please use a simpler format for headings.
- Add a rationale paragraph explaining the why and the what before listing all the changes.
- Please use conventional commits for the format.
- For scope, please refer to release-please configuration if available.
- Keep it short and simple.

### Current State

| Component | Status |
|---|---|
| `flows/agent_loop.go` — model/tool turn loop, interrupts, resume support | Implemented |
| `flows/message.go` — session-backed message flow over agent loop | Implemented |
| `flows/heartbeat.go` — periodic/background heartbeat flow | Implemented |
| `flows/reply.go` — channel-routed reply delivery flow | Implemented |
| `flows/event.go` + `flows/event_context.go` — typed flow lifecycle events | Implemented |
| `flows/heartbeat_config.go` + `flows/heartbeat_result.go` — heartbeat config and result classification | Implemented |
| `tools/bash.go` — command execution with spawn hooks | Implemented |
| `tools/read.go` — text file reading with offset/limit, line-number prefixing, truncation | Implemented |
| `tools/read.go` — image reading with auto-resize (JPEG, PNG, GIF, WebP) | Implemented |
| `tools/write.go` — file creation with auto-mkdir, operator interface | Implemented |
| `tools/truncate.go` — output truncation (line + byte limits) | Implemented |
| `tools/path.go` — path resolution (cwd-relative, ~ expansion, OS-agnostic) | Implemented |
| `tools/edit.go` — find-and-replace with fuzzy matching, BOM/line-ending preservation | Implemented |
| `tools/edit-diff.go` — text normalization, fuzzy matching, unified diff generation | Implemented |
| `tools/diff.go` — LCS-based line diff algorithm | Implemented |
| `media/mime.go` — MIME detection and image auto-resize (CatmullRom scaling) | Implemented |
| `utils/shell.go` — shell environment | Implemented |
| `memory/sessions.go` — session store with persistence modes, operator interface | Implemented |
| `plugins/skills/` — skill discovery, parsing, single `resolve-skill` tool with built-in listing | Implemented |
| Memory: Retrieval (RAG) | Planned |
| Memory: Recall (structured markdown) | Planned |

## Development

### Conventions

- Go module path: `github.com/TheSlowpes/genkit-cowork`
- Primary dependency: `github.com/firebase/genkit/go v1.5.0`
- Use functional options for all public constructors
- Define operator interfaces for any I/O or side-effecting operations
- Keep tool handler logic in the `tools` package; supporting utilities in `media` and `utils`
- All operator interface methods that perform I/O accept `context.Context` as their first parameter for cooperative cancellation
