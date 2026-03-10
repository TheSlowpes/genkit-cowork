# Flows — The Experience Layer

*Genkit Application — Conceptual Documentation*

---

## 1. What is a Flow?

A Flow is the primary unit of observable behaviour in the application. It represents a complete, stateful agent experience — from the moment a user or system initiates an interaction, through every model call and tool execution, to the point the model signals it is done.

Flows are not just function wrappers. They are structured lifecycles: self-contained arenas in which an agent loop runs, state accumulates, and hooks can observe or reshape what is happening at every meaningful moment.

> **Design intent:** If something is happening that a developer, operator, or observer should be able to see, influence, or react to — it happens inside a Flow.

---

## 2. The Agent Loop

At the core of every Flow is an agent loop. This is the cycle the model moves through as it reasons, calls tools, receives results, and decides whether to continue or stop.

The loop is model-driven. The model itself determines when it is done — not an external counter or a hard-coded predicate. When the model has no further tool calls to make, it produces a final response and the loop concludes naturally.

### Loop stages, in order

1. A turn begins.
2. The model receives the current state (history, context, prior tool results).
3. The model produces a message, optionally including one or more tool calls.
4. Each tool call is executed and its result is recorded.
5. If the model issued tool calls, the loop continues to the next turn.
6. If the model issued no tool calls, the loop ends and the Flow concludes.

> **Key principle:** The loop has no fixed iteration count. It runs for as many turns as the model needs. Hooks are the mechanism for shaping that process — not for replacing the model's judgment about when to stop.

---

## 3. Lifecycle Stages and State Scopes

Every meaningful transition in the agent loop is a named lifecycle stage. Each stage has a corresponding state scope — a structured slice of data that belongs to that stage and that stage alone.

The scopes nest hierarchically, mirroring the natural containment of the loop:

### Agent scope

Exists for the full duration of a Flow. Contains everything that must survive across turns: the full conversation history, accumulated tool results, session context, and flow-level metadata such as start time and step count.

- `agent-start` — emitted once when the Flow is first entered.
- `agent-end` — emitted once when the Flow fully completes or fails.

### Turn scope

Exists for a single pass through the loop. Contains the input the model received for this turn and everything produced during it — messages and tool calls — before those results are folded back into agent scope for the next turn.

- `turn-start` — emitted at the beginning of each loop iteration.
- `turn-end` — emitted when all messages and tool calls in a turn have resolved.

### Message scope

Exists for a single model output. A message begins when the model starts generating and ends when it is complete. For streaming flows, message state accumulates incrementally.

- `message-start` — emitted when the model begins generating output.
- `message-update` — emitted as the model streams tokens or partial content.
- `message-end` — emitted when the model's output is finalised.

### Tool execution scope

Exists for a single tool call. Tool scope is nested inside message scope — a message may contain multiple tool calls, each with its own isolated scope. Tool scope holds the call input, intermediate progress, and the final result.

- `tool-execution-start` — emitted when a tool call begins.
- `tool-execution-update` — emitted for tools that report incremental progress.
- `tool-execution-end` — emitted when the tool call resolves with a result or error.

> **Scope rule:** Data at a narrower scope never outlives its parent scope. Turn data is not visible outside the turn. Tool data is not visible outside its parent message. The agent scope is the only data that persists for the full lifetime of a Flow.

---

## 4. Hooks — The Middleware Layer

Hooks are the mechanism by which a developer intervenes in the agent loop. Every lifecycle stage — every start, update, and end event listed above — is a hookable point.

Hooks follow a middleware pattern. When a lifecycle stage fires, the registered hooks for that stage execute in order. Each hook receives the current state for that scope, can read it, and can modify it before returning. The loop does not proceed until the hook has finished.

### What hooks can do

- Read any field in the current scope's state.
- Modify fields in the current scope's state before the next stage runs.
- Halt the loop entirely by returning an error or terminal signal.
- Emit side effects: logging, metrics, external notifications.
- Inject data into state that subsequent hooks or the model will see.

### Hook registration

Hooks can be registered in two ways, and both may be in use simultaneously:

#### Static registration

Hooks declared as part of the Flow definition. These are the default behaviours that run every time this Flow executes, regardless of how it was invoked. Examples include standard logging, safety checks, or mandatory context injection.

#### Runtime registration

Hooks injected at the moment a Flow is invoked. These override or extend the static defaults for that specific invocation. This is the mechanism for per-request customisation — for example, attaching a tracing hook for a specific debug session, or inserting a user-specific context hook without modifying the Flow definition itself.

> **Execution order:** Static hooks run first, then runtime overrides. Within each group, hooks execute in registration order. Every hook is blocking — the loop waits for each hook to complete before moving to the next stage.

---

## 5. Flow Composability

Flows are composable. A Flow may invoke another Flow as a sub-flow, creating a parent-child relationship. This allows complex experiences to be built from smaller, reusable, independently-defined units.

### State scoping in sub-flows

When a parent Flow invokes a sub-flow, the sub-flow receives a scoped view of the parent's agent state — not the full state, and not an empty slate. The sub-flow can read the slice of parent state made available to it, and can write into its own state, but it cannot reach outside its scope to modify parent state directly.

When the sub-flow completes, its result is returned to the parent, which decides how to incorporate it. The parent's agent scope is updated at that point, not before.

> **Isolation guarantee:** A sub-flow cannot corrupt parent state mid-execution. The handoff is explicit and happens only at sub-flow completion.

### Event-driven composition

Flows also interact through events. Every lifecycle stage transition emits an event, and other Flows (or other parts of the system) may subscribe to those events. This allows loosely coupled coordination: a Flow can react to something another Flow did without being directly invoked by it.

This is distinct from direct sub-flow invocation. Event-driven composition is appropriate when the relationship between Flows is observational — one Flow cares about what another produced, but does not need to control it or wait for it to finish before proceeding.

---

## 6. Summary

Flows are the experience layer of the application. They define the observable shape of what the agent does, structured around a model-driven loop with a precise lifecycle. The conceptual contract in brief:

- A Flow is a stateful, lifecycle-managed agent loop.
- The model drives the loop — it signals completion via tool-calling behaviour.
- Every transition in the loop is a named stage that emits an event.
- State is scoped to its stage: agent, turn, message, and tool execution.
- Hooks intercept stages in a blocking middleware chain, with static and runtime registration.
- Flows compose via direct sub-flow invocation (scoped state) or events (loose coupling).

> **Next concept to specify:** Skills — the discrete capabilities the agent loop can invoke during tool execution.
