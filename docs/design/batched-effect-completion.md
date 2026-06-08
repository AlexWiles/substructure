# Design: Batched effect completion + parallel sub-agents

> **Status: implemented.** Phases 1–4 landed (engine batch-drain + `effects.complete`,
> wire/SDK types, simplified `tools`/`subAgents`, tests). The drain is centralized
> in `handle_wake` on fully-applied state and fires via the existing
> `wake_at() == now` path; gating uses a per-effect `delivered` flag with
> `result.is_some()` as the terminal signal. Parallel worker-tool execution
> (phase 5) remains deferred — see Open questions.

## Summary

Today a turn's effects (tool calls and sub-agent delegations) are completed
**one at a time**: the engine emits a worker decision per completion, and the
worker is responsible for not advancing the loop until all of them are in. The
`tools` middleware does this (`pendingToolCalls`); `subAgents` does not — so
parallel and mixed delegations call the LLM prematurely.

This moves the gating into the **engine**, where the authoritative effect state
already lives. The engine runs all of a turn's effects, waits until every one
reaches a terminal state, then delivers **one batched decision** carrying all
results in order. The worker folds them and makes exactly one follow-up LLM
call. Worker-side gating (`pendingToolCalls`, `subAgentTracker`) is deleted.

Scope of this work:

- **In:** engine-owned batched completion for any mix of tool calls and
  sub-agents; parallel sub-agent execution (already separate sessions — this
  makes their *completion* correct).
- **Out (follow-up):** parallel execution of worker-handled **tools** (blocked
  by the single worker-state blob; see Open questions).

## Behavior goal

When the model, in one assistant turn, requests **N effects** (any mix of tool
calls and sub-agent delegations):

1. Run all N as durable steps (sub-agents concurrently; worker tools remain
   sequential for now).
2. Wait until **every** effect reaches a terminal state — `completed` or
   `failed` (retries exhausted).
3. Append every result to the transcript exactly once, paired to its
   `tool_call_id`, in the order the calls were issued.
4. Make **exactly one** follow-up LLM call with all N results present.

Invariants: no premature call (none fire while any effect is pending); no
duplicate call (exactly one per round); no lost results (a failure yields an
error result, never a hang); valid transcript (each result follows the
assistant message carrying its `tool_call`); type-agnostic (identical for tools,
sub-agents, or a mix); and **independent of worker middleware composition order**
— the worker no longer tracks pending state.

## Permutation matrix

A "batch" is the full set of effects produced by one `llm.response`. All cases
gate on *all effects terminal → one `effects.complete` → one `call.llm`*.

| Turn contains | Execution | Completion |
|---|---|---|
| 1 tool | sequential | batch of 1 → one call.llm |
| N tools | sequential (today) | wait for all N → one call.llm |
| 1 sub-agent | own session | child completes → one call.llm |
| N sub-agents | N parallel sessions | wait for all N → one call.llm |
| tool + sub-agent | tool seq, sub-agent parallel | wait for **both** → one call.llm |
| K tools + M sub-agents | tools seq, sub-agents parallel | wait for all K+M → one call.llm |
| includes a client/deferred tool | completes out-of-band via `submitToolCallResult` | batch waits for it too |
| an effect fails (exhausted) | — | error result in the batch; batch still completes |
| an effect times out | engine deadline → `failed` | terminal → batch completes with an error result |
| an effect is `retry_scheduled` | re-run | not terminal → batch keeps waiting |
| 0 effects (no tool_calls) | — | not a batch → `llmLoop` emits `done` |

The mixed and N-of-a-kind rows are exactly what is broken today.

## Current architecture (grounded)

- One worker decision can return many actions; the engine turns each into a
  `ToolCallRequested` / `SubAgentRequested` event in one pass
  (`command.rs` `SubmitWorkerDecision` loop).
- Worker decisions are **globally serialized per session**:
  `has_pending_worker_decision()` is true if *any* decision is pending, so the
  rest queue as `DecisionRequestQueued` (`state.rs`). This protects the single
  opaque `worker_state` blob from concurrent mutation.
- **Sub-agents** run as independent child sessions → already concurrent. Their
  completions arrive at the parent as serialized `sub_agent.turn.complete`
  decisions.
- **Worker-handled tools** run via `tool.execute` worker decisions → serialized
  → sequential.
- On each completion the engine emits a per-effect decision trigger
  (`tool.result` at `command.rs:591/637`, `sub_agent.turn.complete` at `:729`).
  Effect state lives in `SessionState.tool_calls` / `.sub_agent_calls`, each with
  `EffectTracking.status ∈ {pending, completed, failed, retry_scheduled}`.

The engine already has everything needed to gate; it just doesn't.

## The "effect batch"

Because worker decisions serialize and no new effects are created until the
current ones resolve, **all outstanding effects belong to the current turn**.
The batch is therefore implicit: *all `tool_calls` + `sub_agent_calls` whose
status is not terminal*. No explicit batch id is required; an effect ordering
index is (for stable result order).

Terminal = `completed | failed`. Non-terminal = `pending | retry_scheduled`.

## Engine changes (`crates/core`)

### 1. Carry the tool-call identity on sub-agent spawns

So the engine can produce a homogeneous `ToolResult` for a finished sub-agent
without the worker's mapping:

- `WorkerAction::SpawnSubAgent` (`decision.rs`): add `tool_call_id: String`
  (and reuse `agent_id` as the result `name`).
- `RequestSubAgent` command + `SubAgentRequested` event + `SubAgentCallState`
  (`state.rs`): store `tool_call_id`.
- Add an effect **ordering index** to `ToolCallState` and `SubAgentCallState`
  (monotonic in creation order within the turn) for stable result ordering.

> Alternative considered: keep the `session → tool_call_id` map in the worker
> and deliver a heterogeneous batched trigger. Rejected — carrying it on the
> spawn lets the engine own ordering + result assembly and lets us delete the
> worker's `subAgentTracker`.

### 2. Gate completion behind batch drain

Replace per-effect decision emission with a batch-drain check. Introduce one
helper, e.g. `emit_effects_complete_if_drained() -> Option<EventPayload>`:

- Returns `None` while any `tool_calls`/`sub_agent_calls` entry is
  `pending | retry_scheduled`.
- When all are terminal, assembles `Vec<ToolResult>` ordered by the effect
  index (tool results from `ToolCallState`; sub-agent results from
  `SubAgentCallState.tool_call_id` + the child's turn data / error), and returns
  one `emit_decision_request(DecisionTrigger::EffectsComplete { results })`.

Call this helper from **every** path that can move an effect to terminal:

- `CompleteToolCall` / `FailToolCall` (`command.rs:~591/637`) — drop the direct
  `tool.result` emission; call the drain check instead.
- `CompleteSubAgentTurn` / `SubAgentError` (`:712/703`) — drop the direct
  `sub_agent.turn.complete` / `sub_agent.error` emission; call the drain check.
- The wake/timeout path (`:1008–1093`) — when a timed-out effect becomes
  `failed`, run the drain check so a timeout can complete the batch.

The internal events (`ToolCallCompleted`, `SubAgentTurnCompleted`, …) are
unchanged and still recorded — only the **worker-facing decision trigger**
changes. The drain state is recomputable from those events, so replay is safe.

### 3. New decision trigger

`decision.rs` `DecisionTrigger`:

```rust
#[serde(rename = "effects.complete")]
EffectsComplete { results: Vec<ToolResult> },
```

`ToolResult` already exists (`tool_call_id`, `name`, `content`, `is_error`).
This replaces per-effect `tool.result` / `sub_agent.turn.complete` delivery to
the worker. (`sub_agent.error` is folded in as an `is_error` result.)

## Wire / SDK type changes (`packages/sdk/src/types.ts`)

- `WorkerAction` `spawn.sub_agent`: add `tool_call_id: string`.
- `DecisionTrigger`: add
  `| { type: "effects.complete"; results: ToolResult[] }`.
- Remove (or keep deprecated) the now-unused `tool.result` /
  `sub_agent.turn.complete` / `sub_agent.error` trigger handling paths once the
  middleware below stops using them.

## SDK middleware changes (`packages/sdk/src/middleware.ts`)

The worker becomes stateless about gating.

- **`messageHistory` / `messageHistoryCurrentTurn`** — on `effects.complete`,
  push every `ToolResult` as a `role:"tool"` message, in order (extend
  `triggerToMessage` into a `triggerToMessages -> Message[]`, or special-case).
- **`llmLoop`** — treat `effects.complete` like `tool.result`: emit one
  `call.llm`. Remove the `tool.result` case once nothing emits it.
- **`tools`** — delete the `pendingToolCalls` slice and the
  "suppress `call.llm` until all results in" block. Keep: `llm.response` →
  `call.tool`; `tool.execute` → run `execute`; merge tool defs into `call.llm`.
- **`subAgents`** — delete `subAgentTracker`, the `sub_agent.turn.complete` /
  `sub_agent.error` handlers, and `appendToolResultOnce`. Keep: on
  `llm.response`, emit `spawn.sub_agent` (now with `tool_call_id`) +
  `send.message`; merge sub-agent tool defs into `call.llm`; filter `call.tool`
  for sub-agent names. It no longer touches results at all.

Net: both middlewares shrink substantially, and the order-dependence between
them (and with `messageHistory`) disappears — results arrive pre-assembled.

Adapters (`ToolLoopAgent`, `OpenAIAgent`) are unaffected; they compose the same
middlewares.

## Parallel sub-agents

Already parallel (independent sessions). This work only makes their *completion*
correct: the engine waits for all child sessions in the batch before delivering
`effects.complete`. No new execution mechanism is needed.

## Parallel tool execution (deferred)

Worker-handled tools stay sequential this phase. True concurrency is blocked by
the global worker-decision serialization, which exists to protect the single
`worker_state` blob — two `tool.execute` decisions running at once would race on
it. With batched completion, sequential tools still yield one `call.llm`, so the
behavior goal is met; only the latency optimization is deferred.

## Edge cases

- **Timeouts / hangs:** per-effect deadlines mark a stuck effect `failed`
  (terminal), which drains the batch with an error result — no permanent stall.
  The drain check must run from the timeout path (see Engine §2).
- **Failures:** an exhausted effect contributes an `is_error` result; the batch
  still completes.
- **Client/deferred tools:** ordinary batch members; the batch waits for
  `submitToolCallResult`.
- **Zero effects:** `llm.response` with no tool_calls → `done` (unchanged).
- **Nested sub-agents:** a child spawning its own children is its own batch in
  its own session; the parent sees only the child's turn result.
- **Cancellation:** cancelling the parent cancels in-flight children (existing
  behavior); the batch is abandoned with the session.

## Backward compatibility

This changes the worker decision protocol (new `effects.complete` trigger,
`spawn.sub_agent.tool_call_id`) — a breaking change for workers, acceptable
pre-1.0 with a minor bump. Persisted **events** are unaffected (the change is in
worker-facing triggers, recomputed from state). Risk: sessions paused
mid-batch across the upgrade — drain in-flight turns before deploying, or accept
that an in-flight batch may need one manual nudge.

## Testing

**Engine (Rust, `command.rs` tests):**
- N tools → exactly one `EffectsComplete` after the last completes; none before.
- N sub-agents → same.
- tool + sub-agent mixed → waits for both; results ordered by issue order.
- a failed/timed-out effect drains the batch with an `is_error` result.
- single effect → one `EffectsComplete`.
- regression guard: no `EffectsComplete` while any effect is non-terminal.

**SDK (vitest harness):**
- `effects.complete` with N results → `messageHistory` folds N tool messages in
  order; `llmLoop` emits exactly one `call.llm`.
- mixed results fold correctly; order preserved.
- `subAgents` emits `spawn.sub_agent` with `tool_call_id`; no tracker state.
- delete/replace the order-dependence and `appendToolResultOnce` tests (no
  longer meaningful once gating is engine-side).

## Rollout phases

1. **Engine:** spawn `tool_call_id` + effect index; batch-drain helper wired
   into completion *and* timeout paths; `EffectsComplete` trigger. Tests.
2. **Wire/SDK types:** `effects.complete`, `spawn.sub_agent.tool_call_id`.
3. **SDK middleware:** handle `effects.complete` in `messageHistory` + `llmLoop`;
   strip gating from `tools`; strip tracker/handlers from `subAgents`. Tests.
4. **Cleanup:** remove `pendingToolCalls`, `subAgentTracker`,
   `appendToolResultOnce`, and dead per-effect trigger handling.
5. **Follow-up:** parallel worker-tool execution (needs the Open question).

## Open questions

- **Parallel worker-tool execution & tool `stateSlice`.** To run worker tools
  concurrently we must resolve `worker_state` contention. Likely answer:
  parallel only for stateless tools; serialize any tool declaring a
  `stateSlice`. Alternatively make `tool.execute` a stateless side-effect that
  never writes back `worker_state`. Needs a decision before phase 5.
- **Result ordering source of truth.** Engine creation-order index vs. the
  assistant message's `tool_calls` order — these should coincide, but confirm
  the engine preserves action order when fanning out effects.
- **Streaming partial progress.** Batching removes per-result reactivity. If any
  agent needs "act on first result," it would need an opt-out — out of scope.
