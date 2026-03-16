# Flows — Experience Layer (Concept + Current State)

*This document captures both the intended design of Flows and the behavior currently implemented in `genkit-cowork/flows`.*

---

## 1. What is a Flow?

A Flow is the primary unit of observable agent behavior. It represents a complete, stateful agent experience. From the moment a user or system
initiates an interaction, through every model call and tool execution, to the point the model signals it is done.

- carries state across steps,
- provides clear stage boundaries,
- exposes interception points for operators,
- and makes autonomous work inspectable and controllable.

>**Design intent** If somethis is happening that a developer, operator, or observer should be able to see, influence, or react to; it happens
> inside a flow

In practical terms, Flows provide:

- **Structured execution** — clear beginning, progress points, and completion.
- **Scoped state** — agent, turn, message, and tool execution concerns stay separated.
- **Composable behavior** — higher-level experiences are built from reusable flow units.
- **Operational hooks** — logging, policy checks, mutation, and side effects attach to lifecycle stages.

---

## 2. Implemented Flow Types

The project currently implements four concrete flow layers:

1. **Agent loop (`agentLoop`)**
   - Core think/act loop that calls the model, executes tool requests, and continues until the model stops requesting tools.
2. **Message handling (`handleMessage`)**
   - Session-aware wrapper around `agentLoop` for user/chat style turns.
   - Loads or creates a session, appends incoming messages, runs the loop, and persists new history.
3. **Heartbeat (`heartbeat`)**
   - Periodic/background execution path that runs the loop against existing session context.
   - Supports schedule interval, active-hour windows, and delivery filtering (`ack`, `alert`, `skipped`, `error`).
   - Preserves the full `*ai.Message` response (including media and documents) for downstream delivery.
4. **Send reply (`sendReply`)**
   - Channel-routed reply delivery flow that dispatches agent output to external messaging channels.
   - Routes via a `ChannelHandler` registry keyed by `MessageOrigin` (e.g., WhatsApp, Zoom, email).
   - Supports multi-tenant sender identity and channel-specific destination routing.

---

## 3. Agent Loop Contract

At the core of every Flow is an agent loop. This is the cycle the model moves through as it reasons, call tools, receives results, and
decides whether to continue or stop.

agentLoop stages, in order:

1. Emit `agent-start`.
2. For each turn, emit `turn-start`.
3. Call model generation with history, configured tools, and model/system options.
4. Emit `message-start` and `message-end` for the model response.
5. If no tool requests are present, emit `turn-end`, then finish.
6. If tool requests are present, execute each tool call, append a tool message, emit `turn-end`, and continue.
7. Emit `agent-end` on completion (or before returning on generation failure).

Additional behavior:

- Optional `MaxTurns` guard returns an error when exceeded.
- Resume support via `toolResponses` and `toolRestarts` on the first turn.
- Interrupt support: if a tool returns an interrupt error, the loop returns `FinishReasonInterrupted` and surfaces interrupt parts.

---

## 4. Event and Context Model

Flow observability uses `EventBus` with typed event payloads.

Defined event types:

- `agent-start`, `agent-end`
- `turn-start`, `turn-end`
- `message-start`, `message-update`, `message-end`
- `tool-execution-start`, `tool-execution-update`, `tool-execution-end`

Important implementation notes:

- `message-update` is defined but not currently emitted by `agentLoop` (no token streaming events yet).
- `tool-execution-update` is currently emitted for tool interrupts.
- Event handlers run synchronously in registration order.
- `tool-execution-start` can mutate tool input through returned event data (`event.Data.Input`), and that mutated input is used for execution.
- Most event emission call sites currently ignore handler errors; events are primarily observational in the current implementation.

---

## 5. Session-Coupled Flows

Both `handleMessage` and `heartbeat` persist loop output into session state:

- Load existing session by `SessionID`, or create one with `TenantID`.
- Build model history from stored session messages.
- Run `agentLoop`.
- Persist only newly generated messages back to the session.

Origin mapping in persisted messages:

- User role uses inbound origin.
- Model role maps to `model` origin.
- Tool role maps to `tool` origin.
- Heartbeat writes user-equivalent entries with `heartbeat` origin.

---

## 6. Heartbeat Runtime Behavior

`Heartbeat` is a managed runner around the `heartbeat` flow:

- `Start(ctx)` creates a ticker using configured `Interval`.
- `Run(ctx, tickTime)` enforces:
  - active-hours filtering (`outside_hours` skip),
  - single-flight guard (`busy` skip),
  - execution and result callback.
- `Wake(ctx)` triggers immediate asynchronous run.
- `Stop()` closes the internal stop channel once.

Heartbeat output classification:

- `HEARTBEAT_OK` token => `ack` (if stripped content length is within `AckMaxChars`).
- Otherwise => `alert`.
- Runtime skip/error paths => `skipped` / `error`.

`HeartbeatOutput` preserves the full `Response *ai.Message` from the agent loop. This ensures downstream consumers (such as the send reply flow) have access to the complete model response, including any media parts or structured content, not just the extracted text.

---

## 7. Send Reply Flow

`sendReply` is a channel-routed delivery flow that dispatches agent output to external messaging channels (WhatsApp, Zoom, email, etc.).

### Routing Model

Handlers are registered via a `map[memory.MessageOrigin]ChannelHandler` keyed by message origin. When the flow receives a `SendReplyInput`, it looks up the handler for the input's `Channel`. If no handler is registered, the reply is skipped with a reason.

### ChannelHandler Interface

Each channel implements three methods:

- `Setup(ctx, tenantID)` — One-time initialization for a tenant (e.g., authenticate, validate credentials).
- `SendReply(ctx, input)` — Deliver a reply to the channel. The input carries the full `*ai.Message` content, sender identity, and destination routing.
- `Acknowledge(ctx, input)` — Send a lightweight acknowledgment (e.g., a typing indicator or read receipt). Not invoked by the flow directly; available for caller orchestration.

### Skip Conditions

The flow skips delivery (returns `Skipped: true`) when:

- No handler is registered for the input channel.
- The `Target` is `HeartbeatTargetNone` or empty.

### Input/Output Structure

`SendReplyInput` carries:

- `Sender` — Multi-tenant identity with `TenantID`, `DisplayName`, and optional `Username`.
- `Content` — The full `*ai.Message` (not just text), preserving media and structured parts.
- `Channel` — The `MessageOrigin` used for handler routing.
- `Target` — The heartbeat target classification that triggered this reply.
- `Destination` — Channel-specific routing with `ChatID`, optional `MessageID`, and optional `ThreadID`.

`SendReplyOutput` reports whether the message was `Delivered`, `Skipped`, and an optional `Reason`.

### SetupSenders Helper

`SetupSenders(ctx, tenantID, senders)` iterates all registered handlers and calls `Setup` on each. Fails fast on the first error.

### Options

- `WithReplyInThread()` — Signals that replies should be threaded where the channel supports it. Wired into options but not yet consumed by flow logic.

---

## 8. Current Boundaries and Intended Direction

The current flow system is functional and covers the core execution, session management, background heartbeat, and channel delivery paths:

- Event-driven hooks exist through `EventBus`, but there is no separate static-vs-runtime hook registry abstraction yet.
- Direct sub-flow orchestration and explicit parent/child scoped-state composition are not implemented as a first-class flow API.
- Streaming message lifecycle (`message-update`) is planned by type definition but not yet active in the loop.
- The `sendReply` flow handles outbound delivery routing, but orchestration of heartbeat-to-reply wiring is left to the caller.
- `WithReplyInThread` is accepted as an option but not yet consumed by the flow handler logic.

Intended direction:

- Keep the model-driven loop as the core execution primitive.
- Expand lifecycle coverage to include streaming updates and richer turn/message telemetry.
- Formalize flow composition (sub-flows and scoped handoff contracts) as a first-class API.
- Evolve hooks toward clearer policy/middleware ergonomics while preserving synchronous determinism where needed.
- Wire `replyInThread` into channel handler dispatch and extend `Destination` routing as channel-specific needs emerge.
