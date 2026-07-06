# Plan: Branch-scoped worker state

Status: implemented.
Scope: `crates/core`, `crates/napi`, `packages/sdk`, docs. No backward compatibility required.

---

## 1. Background (read this if you have no context)

Substructure is an event-sourced agent-session engine. A **session** is an aggregate
(`crates/core/src/runtime/session/`) whose state is folded from an append-only event
log (`SessionState::apply` in `session/state.rs`). The engine makes no agent decisions
itself: whenever anything happens (a user message arrives, an effect settles), it emits
a **worker decision request** carrying the trigger, the transcript, in-flight effects,
and the worker's own state. A **worker** (customer code, any language, plain JSON over
HTTP or an embedded callback) answers with a **submit**: actions to take (`call.llm`,
`call.tool`, `done`, …), a transcript, and its state.

Two session structures matter here:

- **The message tree.** Messages are nodes with parent pointers
  (`SessionState.nodes: Vec<Node>`, `session/events.rs`). A single pointer
  `SessionState.head_id` marks the active branch's leaf. The *only* writer is
  `reconcile_transcript` (`session/command.rs`): known-id messages advance a parent
  cursor, unknown/id-less messages are appended under it. Appending after a known
  prefix that already has other children **forks** the tree; `head_id` follows every
  append (`apply` for `NewMessage`). So branching exists and works today.

- **Worker state.** An opaque JSON value (`WorkerState`, `runtime/worker/state.rs`)
  the engine stores but never interprets. Today it is a **single mutable cell**:
  `SessionState.worker_state`, overwritten by every `WorkerDecisionCompleted` and
  `WorkerStateUpdated` event, delivered back on every decision request.

## 2. The problem

The transcript **branches**; worker state is **linear**. Workers keep conversational
context in state — agent memory, rolling summaries, parked approval calls, and (by the
convention this plan introduces) the system prompt. When the tree forks back to an
earlier point, the state still contains everything learned on the abandoned branch:

- Fork at message 10 of 50 → the new branch prompts with memory from messages 11–50,
  a timeline that never happened on this branch.
- A summary covering messages 1–40 governs a branch that only contains 1–10.
- There is no per-branch record of which instructions/memory version governed which
  response.

Secondary problem, solved here purely by convention: harnesses contort themselves to
keep a system-prompt message in the tree with a stable node id (find-existing-node,
reuse-id, strip-and-reprepend dances), and that pattern silently freezes instructions
at turn 1. The fix: **the tree is pure conversation**. Instructions are worker config,
prepended to the LLM request at build time (the exact prompt is already durably
recorded in every `LlmCallRequested` event); their *history* is recorded in
`state.system`, which this plan makes branch-aware.

## 3. Design

### Core idea

Replace the single mutable cell with **versions anchored to tree positions**:

- Every state write becomes a `StateVersion { state, anchor }` where `anchor` is the
  node id that was the active head when the version was written (`None` when the tree
  was empty).
- The **current state is derived, never stored**: the newest version whose anchor lies
  on the active path (root → `head_id`). An anchor of `None` matches every path.

One rule, all the required semantics:

| Situation | Outcome |
|---|---|
| Linear session | Anchors are always on the path → identical to today's behavior. |
| Fork at node W | Versions anchored after W fall off the path → the branch resolves to state **as-of W**. Uncontaminated by construction, zero copying. |
| Worker wants to carry state across a fork | It submits state on the forking decision; the version anchors to the new head. Both semantics expressible, engine chooses neither. |
| Two branches alternately extended | Each resolves its own versions. Per-branch isolation. |
| "Branch at the system prompt" | Change `state.system` → new version anchored at head; same conversation, new instructions, old branch's version still resolvable. |

### Write rules (all mechanical, no policy)

1. **Dedup (PUT semantics).** A submitted state equal to the currently-resolved state
   writes nothing. Workers echo state on every decision; echoes must be free.

   *Equality semantics:* plain structural equality on parsed values —
   `WorkerState` derives `PartialEq`, which delegates to `serde_json::Value::eq`.
   No canonicalization, hashing, or byte comparison; never re-serialize to compare.
   Consequences:
   - Wire formatting (whitespace, object key order, escaping) never matters —
     erased by parsing; `Value::Object` equality is key-set based.
   - Arrays are ordered: `[1,2] != [2,1]`.
   - `null` is not absence: `{"a": null} != {}`.
   - Numbers compare by parsed variant: integer `1` ≠ float `1.0`; strings compare
     by exact codepoints (no Unicode normalization).

   The edges are safe because the failure modes are asymmetric and equality only
   affects log leanness, never correctness: a false *difference* (e.g. `1` vs `1.0`)
   writes one redundant version whose resolution is indistinguishable; a false
   *equality* is impossible by construction — if two values are `Value`-equal,
   the stored one is indistinguishable from the submitted one for every reader.
2. **Same-anchor compaction.** Two versions with the same anchor are indistinguishable
   to resolution (identical path-membership; newest-first scan always picks the later),
   so the fold keeps only the newest per anchor. Bound: **at most one live version per
   tree node**, plus one unanchored. The full change history remains in the event log.
3. **Omitted state means "keep".** `SubmitWorkerDecision.state` becomes
   `Option<WorkerState>`: `None` → no opinion, no write. (Today an omitted wire field
   deserializes to JSON `null` and silently clears state — a footgun for no-SDK
   workers.) Explicitly clearing is still expressible: submit `{}` or `null` as a
   present value.

### Where state lives (end state)

1. **Durable truth** — `WorkerStateUpdated { state, anchor }` events in the session's
   log, interleaved with everything else. The only state-carrying event.
   (`WorkerDecisionCompleted` loses its `state` field; it is purely a completion
   marker.)
2. **Fold** — `SessionState.state_versions: Vec<StateVersion>`, compacted per anchor.
   The old `SessionState.worker_state` field is deleted.
3. **Delivery** — `DerivedState.worker_state` is populated from resolution, so the
   worker-facing wire (decision requests via queue/push) is byte-identical to today.

### Explicitly out of scope (compose later, block nothing)

- `HeadMoved` (repoint head to an existing leaf without appending).
- `window_after` delivery bounding (engine ships only the transcript tail).
- Exposing `state_versions` through a client read route (the event log already has it).
- State-as-of reconstruction tooling for forks (derivable from the log).

---

## 4. Implementation

### 4.1 `crates/core/src/runtime/session/events.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionCompleted {
    pub decision_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStateUpdated {
    pub state: WorkerState,
    /// The active head when this version was written; `None` if the tree was empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}
```

### 4.2 `crates/core/src/runtime/session/state.rs`

Types and fields:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVersion {
    pub state: WorkerState,
    pub anchor: Option<String>,
}
```

- On `SessionState`: delete `pub worker_state: WorkerState`; add
  `#[serde(default)] pub state_versions: Vec<StateVersion>` (default keeps snapshot
  deserialization total; old snapshots are not otherwise supported per no-back-compat).
- `DerivedState.worker_state: WorkerState` stays — now filled by resolution.

Resolution helpers on `SessionState`:

```rust
/// All node ids (messages and controls) on the root→leaf chain.
/// `path_to` returns only messages; anchors may be any node kind.
fn path_ids<'a>(&'a self, leaf: &'a str) -> std::collections::HashSet<&'a str> {
    let by_id: HashMap<&str, &Node> = self.nodes.iter().map(|n| (n.id(), n)).collect();
    let mut ids = std::collections::HashSet::new();
    let mut cursor = Some(leaf);
    while let Some(id) = cursor {
        let Some(node) = by_id.get(id) else { break };
        if !ids.insert(id) {
            break; // malformed parent cycle guard, mirrors path_to
        }
        cursor = node.parent_id();
    }
    ids
}

/// Newest version whose anchor is on the path to `leaf`; unanchored matches any path.
pub fn resolve_state_for(&self, leaf: Option<&str>) -> WorkerState {
    let on_path = leaf.map(|l| self.path_ids(l)).unwrap_or_default();
    self.state_versions
        .iter()
        .rev()
        .find(|v| match v.anchor.as_deref() {
            None => true,
            Some(a) => on_path.contains(a),
        })
        .map(|v| v.state.clone())
        .unwrap_or_default()
}

pub fn resolved_worker_state(&self) -> WorkerState {
    self.resolve_state_for(self.head_id.as_deref())
}
```

`apply` changes:

```rust
EventPayload::WorkerDecisionCompleted(p) => {
    if let Some(wd) = self.worker_decisions.get_mut(&p.decision_id) {
        wd.tracking.complete();
    }
    // no state write — WorkerStateUpdated is the only state-carrying event
}
EventPayload::WorkerStateUpdated(p) => {
    // same-anchor compaction: the older version can never win resolution
    self.state_versions.retain(|v| v.anchor != p.anchor);
    self.state_versions.push(StateVersion {
        state: p.state.clone(),
        anchor: p.anchor.clone(),
    });
}
```

(`retain` preserves relative order of survivors and `push` appends the newest, so the
newest-first scan order is exactly event order. Correctness of compaction: an older
same-anchor version matches precisely the same paths as the newer one and is scanned
later, so it is unreachable.)

`derived_state()`: `worker_state: self.resolved_worker_state()`.

### 4.3 `crates/core/src/runtime/session/command.rs`

Command shape:

```rust
SubmitWorkerDecision {
    decision_id: String,
    transcript: Vec<Message>,
    actions: Vec<WorkerAction>,
    state: Option<WorkerState>,   // None = keep current
},
```

Handler (`CommandPayload::SubmitWorkerDecision` arm). Keep the existing idempotency
guard, dispatched-effect resolution, action dispatch, and queued-decision promotion
unchanged. Insert state handling between reconcile and actions:

```rust
let mut events: Vec<EventPayload> = vec![EventPayload::WorkerDecisionCompleted(
    WorkerDecisionCompleted { decision_id },
)];
let reconcile = self.reconcile_transcript(transcript);

// The head after this batch applies: last appended node, else the current head.
let post_head = reconcile
    .iter()
    .rev()
    .find_map(|e| match e {
        EventPayload::NewMessage(m) => Some(m.message.id.clone()),
        _ => None,
    })
    .or_else(|| self.head_id.clone());

// The known-tree leaf of the post-reconcile path. New nodes carry no anchors, so
// resolving at the last known prefix node is equivalent to resolving at post_head.
// The first appended message's parent_id IS that prefix leaf (reconcile_transcript
// threads it); None means the fork starts at the root.
let prefix_leaf = reconcile
    .iter()
    .find_map(|e| match e {
        EventPayload::NewMessage(m) => Some(m.parent_id.clone()),
        _ => None,
    })
    .unwrap_or_else(|| self.head_id.clone());

events.extend(reconcile);

if let Some(state) = state {
    if state != self.resolve_state_for(prefix_leaf.as_deref()) {
        events.push(EventPayload::WorkerStateUpdated(WorkerStateUpdated {
            state,
            anchor: post_head,
        }));
    }
}
// ... action dispatch as today ...
```

Ordering rationale: the state version must anchor to the *post-reconcile* head so a
later fork from inside this turn's appends resolves correctly; it must precede action
dispatch so a `done` in the same batch completes the turn with the new state already
recorded.

`CompleteToolCall` / `FailToolCall` arms (the out-of-band `settleEffect` path that may
carry `worker_state: Option<WorkerState>` — see `runtime/mod.rs::settle_effect`):
replace the unconditional `WorkerStateUpdated` push with:

```rust
if let Some(ws) = worker_state {
    if ws != self.resolved_worker_state() {
        events.push(EventPayload::WorkerStateUpdated(WorkerStateUpdated {
            state: ws,
            anchor: self.head_id.clone(),
        }));
    }
}
```

### 4.4 Plumbing (mechanical)

- `runtime/worker/queue.rs` — `SubmitDecision.state: Option<WorkerState>`.
- `runtime/mod.rs::submit_decision` — passes `input.state` through unchanged (now
  `Option`).
- `transport/worker_http/types.rs` — `SubmitRequest.state:
  Option<WorkerState>` with `#[serde(default)]`; `routes.rs` passes through.
- `crates/napi/src/lib.rs` (~line 181) — `state: submit.state.map(Into::into)`
  instead of `unwrap_or_default()`.
- `runtime/worker/handler.rs`, `push.rs`, wake system, SSE — no changes; they consume
  `DerivedState.worker_state`, which is now resolved.

### 4.5 `packages/sdk`

- `src/types.ts` — `SubmitRequest.state` becomes optional (`state?: unknown`); the
  `worker.decision.completed` event type drops `state`; the `worker.state.updated`
  event type gains `anchor?: string`.
- `src/worker.ts` (`runDecision`) — replace
  `state: out.state !== undefined ? out.state : req.state` with `state: out.state`.
  Undefined now correctly means "keep" at the engine, which is exactly what an agent
  returning no state opinion intends. Agents that do return state are deduped
  engine-side.

### 4.6 Tests

`session/state.rs` unit tests:
- resolution: linear (latest wins), fork (as-of the fork point), unanchored fallback,
  empty versions → default `WorkerState` (JSON null).
- compaction: same-anchor replace-in-place; resolution equivalence before/after.

`session/command.rs` tests (use the existing `dispatch`/`create_session` helpers):
- echo dedup: submit with state equal to resolved → no `WorkerStateUpdated` in batch;
  include a key-order-shuffled object (still deduped) and a `{"a": null}` vs `{}`
  pair (written — null is not absence).
- change: submit with new state → one `WorkerStateUpdated` anchored at the
  post-reconcile head (assert anchor equals the last appended node id).
- fork: submit a transcript diverging after a known prefix plus a new state → anchor
  is the new leaf; then submit a second fork from the same prefix *without* state and
  assert the delivered/derived state resolves to the pre-fork version.
- omitted state (`None`) → no state event, resolution unchanged.
- `settleEffect` path: `CompleteToolCall` with changed/unchanged `worker_state`.
- update existing tests constructing `SubmitWorkerDecision` (`state: X` →
  `state: Some(X)` or `None`; assertions on `WorkerDecisionCompleted.state` deleted).

Worker projection (`worker/handler.rs` has test coverage via derived events): assert a
decision requested after a fork carries the as-of state.

### 4.7 Docs + changelog

- `docs/02-concepts.md` — replace the worker-state section: versions, anchors,
  resolution rule; "conversational truth lives in state and forks with the branch;
  physical truth (what actually ran) lives in the event log and never forks."
- `docs/06-patterns.md` — new patterns:
  - **System prompt**: instructions are worker config; prepend to `call.llm` messages
    at build time; record in `state.system`; never store a system message in the tree.
    Changing `state.system` is branching at the system prompt.
  - **Memory**: keep agent memory in `state.memory`; forks are uncontaminated
    automatically.
  - **Compaction**: `state.summary = { text, through: node_id }`; prompt =
    `[system, summary] ++ messages after through`; the tree keeps full history.
- `docs/07-protocol.md` — submit semantics: state is PUT-with-dedup; omitted = keep;
  present `null`/`{}` = explicit clear; dedup equality is structural JSON equality
  (key order irrelevant, arrays ordered, `null` ≠ absent key).
- `CHANGELOG.md` — entry describing: branch-scoped worker state (versions anchored to
  tree nodes, resolution by active path), `state` optional on submit (omitted = keep),
  `WorkerDecisionCompleted` no longer carries state, `WorkerStateUpdated` gains
  `anchor`.

### 4.8 Suggested sequence

1. Events + fold + resolution + state.rs tests (additive, nothing consumes it yet).
2. Command handler + plumbing (queue, mod, worker_http, napi) + command tests.
3. SDK types/worker + docs + CHANGELOG.

`cargo test -p substructure-core` after each step; SDK typecheck/tests after step 3.

---

## 5. Worked example (sanity check for the implementer)

```
turn 1:  user u1 → worker submits state {system: "v1"}         version A anchor u1
turn 2:  … a1, u2, a2 appended; worker echoes state            no writes (dedup)
turn 3:  memory tool fires → state {system: "v1", memory: M}   version B anchor a3
         (a3 = head after this submit's reconcile)
fork:    worker submits transcript [u1, X] (X id-less, new)    X appended, parent u1,
                                                               head → X
resolve: path is u1→X; B's anchor a3 is off-path; A's anchor
         u1 is on-path → state = {system: "v1"}                as-of semantics
carry:   had the fork submit also sent {system: "v1", memory: M},
         that would be version C anchor X → explicit carry-over
```
