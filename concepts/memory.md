# Memory - Time and the Accumulation of History

## 1. Design Intent

Memory is the record of everything that has happened to the tenant over time. It is not a cache, and it is not a prompt buffer. It is the authoritative, append-only ledger of
experience — every turn, every tool result, every file provided, every insight consolidated — structured so that nothing is ever lost and everything is always recoverable.

If Flows are what the agent *does*, Memory is what the agent *has been through*. It is the temporal dimension of the system.

> **Design intent:** Memory does not decide what the agent sees right now — that is the job of the Context Engine. Memory's job is to ensure that everything which happened is preserved faithfully, owned unambiguously, and available to be recalled.

---

## 2. Tenant Ownership

Memory is modelled on the principle that data belongs to the tenant who created it, strictly and completely. No memory — no session, no turn, no file, no insight — ever crosses a tenant boundary. Tenants are isolated from each other at every layer of the memory system.

This follows the same conceptual foundation as the AT Protocol: the tenant is the owner, and ownership is not a permission setting but a structural guarantee. The system is architected so that cross-tenant access is not a thing that can be misconfigured — it is simply not a path that exists.

### What tenant ownership means in practice

- Every piece of stored data carries a tenant identifier as a first-class attribute, not an annotation.
- All reads and writes are scoped to a tenant at the storage layer, not enforced only at the application layer.
- Consolidation, vectorisation, and indexing jobs operate strictly within a single tenant's data.
- There is no operator-level view that aggregates or compares data across tenants.

---

## 3. Sessions — The Primary Unit of Memory

A session accumulates the full, unabridged history of everything that occurred within it:

- Every turn: the model input, the model output, and all tool calls and their results.
- The lifecycle metadata for each turn: timestamps, step counts, flow identifiers.
- The complete agent-scope state snapshot at the end of each iteration.

### Whole-state checkpoints at every iteration

At the conclusion of each Flow iteration, the entire agent-scope state is written to the session record — not a diff, not a summary, the whole state. This is intentional. It means that any point in the session's history can be reloaded exactly as it was, enabling retries, replays, and debugging without reconstruction or inference.

Snapshot records are metadata checkpoints over that canonical state, including sequence metadata and integrity checksums. They are not standalone embedded copies of session state; replay state is sourced from the append-only session record.

> **Immutability guarantee:** Session history is append-only. Once a turn is written, it is never modified or deleted. The record of what happened is permanent. This is not a technical limitation — it is a design commitment. The past is not editable.

---

## 4. Pruning — Loading a Manageable Window

Storing the whole history at every iteration does not mean the whole history is loaded into the agent's working context every time. On session load, the history can be pruned to a window appropriate for the current Flow run. Three strategies are supported:

### Sliding window

Load only the last N turns from the session. This is the simplest strategy and works well for sessions where recency is what matters most — the conversation has been building continuously and the tail is the most relevant part.

### Tail ends — first N and last N turns

Load the first N turns of the session and the last N turns, skipping the middle. This is the strategy for long-lived sessions where both the origin of the relationship and the most recent activity are important, but the bulk of intermediate history can be omitted from the immediate context. The omitted middle is not lost — it remains in the session record and in the vector store, available for recall.

### Token budget

Load as much recent history as fits within a configured token budget. This strategy preserves recency while adapting to variable message sizes, and is useful when strict model context limits matter more than a fixed message count.

> **Pruning does not delete.** Every strategy is a read-time windowing decision. The full session history in the store is never altered by pruning.

---

## 5. The Vector Store — Long-Range Recall

Every turn written to the session is also vectorised and indexed in a vector database, scoped to the tenant. This runs in parallel with session storage and is not on the critical path of a Flow execution.

The vector store serves a different purpose from session state. Session state is the ordered, complete record — the ledger. The vector store is the recall surface — the mechanism for finding relevant history that falls outside the pruned window.

When the Context Engine (which will be covered in its own document) assembles context for a Flow run, it can query the vector store to retrieve semantically relevant turns from anywhere in the tenant's history, not just the recent window. This means that even in a long-lived session where the middle has been pruned, an old conversation about a specific topic can be surfaced when that topic becomes relevant again.

---

## 6. Files — Tenant-Global, Vectorised

Files provided by the tenant — documents, references, uploads of any kind — are stored and indexed at the tenant level, not the session level. A file uploaded in one session is part of the tenant's memory and is available to every subsequent session.

On ingestion, each file is:

1. Stored in the tenant's file store.

2. Chunked and vectorised, with chunks indexed in the tenant's vector store alongside conversation history.

3. Tagged with provenance metadata: when it was provided, in which session, and by what means.

---

## 7. Consolidation — Building Cross-Cutting Insight

On a scheduled basis — nightly by default — a consolidation job runs across each tenant's full memory: their complete session history and their full file library.

The job's purpose is to surface connections that span individual sessions and files. A preference mentioned in one session and a document provided in another may share a theme that neither surface makes explicit on its own. Consolidation finds these links and writes them as derived insights, also stored in the tenant's vector store.

### What consolidation produces

- Linked references between related turns across different sessions.
- Linked references between conversation history and file content that address the same topic.
- Extracted and reinforced tenant preferences — patterns in how the tenant works, what they respond to, what methods they favour — surfaced from accumulated history and stored as explicit preference records.

### What consolidation does not do

- It does not modify any original record. Sessions, turns, and files remain immutable.
- It does not cross tenant boundaries. Each tenant's consolidation job sees only that tenant's data.
- It does not make decisions about what the agent will see — that remains the Context Engine's responsibility.

> **Consolidation is an enrichment layer.** It adds derived structure on top of the immutable record, making the full depth of a tenant's history more navigable and more useful without altering the history itself.

---

## 8. Tenant Preferences

Preferences are a first-class record type in the memory system, distinct from session history and files. They represent what the system has learned about how a specific tenant works: their preferred methods, their communication style, their domain-specific terminology, their treatment preferences.


Preferences are seeded in two ways:

- **Explicitly** — the tenant states a preference directly, which is recorded immediately.

- **Implicitly** — the consolidation job extracts recurring patterns from history and promotes them to preference records, subject to a confidence threshold.

Like all memory records, preferences are tenant-owned. 

---

## 9. Summary

Memory is the temporal layer of the system — the accumulation of everything a tenant has done, provided, and expressed over time. Its conceptual contract:

- Memory is tenant-owned, structurally isolated, and never shared across tenant boundaries.
- Sessions are the primary container. They accumulate full whole-state snapshots at every iteration, immutably and append-only.
- Sessions may be short (a single Flow run) or long-lived (spanning many Flow runs over time).
- On load, history is windowed by one of three pruning strategies: recent N turns, token budget, or tail ends.
- The full history — including pruned turns — is vectorised and available for long-range recall.
- Files are tenant-global, vectorised on ingestion, and immutable.
- A nightly consolidation job links cross-cutting insights across sessions and files without modifying any original record.
- Tenant preferences are a first-class record type, populated explicitly or extracted by consolidation.
