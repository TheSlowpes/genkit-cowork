# Flows — Experience Layer (Concept + Current State)

*This document captures both the intended design of Flows and the behavior currently implemented in `genkit-cowork/flows`.*

---

## 1. Design Intent

A Flow is the primary unit of observable agent behavior. It is not just a function call; it is a lifecycle that:

- carries state across steps,
- provides clear stage boundaries,
- exposes interception points for operators,
- and makes autonomous work inspectable and controllable.

The intent is that anything important in agent execution should happen inside a Flow boundary, where it can be observed, constrained, and extended.

In practical terms, Flows provide:

- **Structured execution** — clear beginning, progress points, and completion.
- **Scoped state** — agent, turn, message, and tool execution concerns stay separated.
- **Composable behavior** — higher-level experiences are built from reusable flow units.
- **Operational hooks** — logging, policy checks, mutation, and side effects attach to lifecycle stages.

---

## 2. Implemented Flow Types

The project currently implements three concrete flow layers:

1. **Agent loop (`agentLoop`)**
   - Core think/act loop that calls the model, executes tool requests, and continues until the model stops requesting tools.
2. **Message handling (`handleMessage`)**
   - Session-aware wrapper around `agentLoop` for user/chat style turns.
   - Loads or creates a session, appends incoming messages, runs the loop, and persists new history.
3. **Heartbeat (`heartbeat`)**
   - Periodic/background execution path that runs the loop against existing session context.
   - Supports schedule interval, active-hour windows, and delivery filtering (`ack`, `alert`, `skipped`, `error`).

---

## 3. Agent Loop Contract

`agentLoop` is model-driven and turn-based:

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

---

## 7. Current Boundaries and Intended Direction

The current flow system is functional but intentionally narrow:

- Event-driven hooks exist through `EventBus`, but there is no separate static-vs-runtime hook registry abstraction yet.
- Direct sub-flow orchestration and explicit parent/child scoped-state composition are not implemented as a first-class flow API.
- Streaming message lifecycle (`message-update`) is planned by type definition but not yet active in the loop.

Intended direction:

- Keep the model-driven loop as the core execution primitive.
- Expand lifecycle coverage to include streaming updates and richer turn/message telemetry.
- Formalize flow composition (sub-flows and scoped handoff contracts) as a first-class API.
- Evolve hooks toward clearer policy/middleware ergonomics while preserving synchronous determinism where needed.
