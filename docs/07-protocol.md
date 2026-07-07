---
title: The decision protocol
---

The engine and your worker talk over one HTTP call, repeated: the engine sends a
**decision request** describing what just happened, your worker replies with a
**decision**: the actions to take and the conversation as it should now read.

That request/response contract is the whole protocol. The TypeScript SDK's
`toolLoop` is one implementation of it; this page is the language-neutral spec, so
you can implement a worker (and the tool loop itself) in any language.

Everything here is plain JSON. There are no streaming semantics to implement on
the worker (streaming is opt-in and covered at the end), no hidden state: a worker
is a pure function from a decision request to a decision.

One rule makes the whole protocol cohere:

> **A trigger is news that hasn't been written into the conversation yet.** The
> request carries the conversation as recorded and the news as `trigger`; your
> decision says how the conversation now reads (`messages`) and what to do
> next (`actions`).

A `client.messages`'s proposed messages are not in the transcript until you put
them there; an `llm.finished`'s assistant message isn't either; a
`tool.finished`'s result becomes a tool message only because you write one. Every
trigger is a different sort of news, and your reply writes it in.

## The exchange

The engine `POST`s a JSON **decision request** to your worker's endpoint. Your
worker responds with a JSON **decision**. One request, one response, once per
decision.

### Decision request (engine → worker)

```jsonc
{
  "session_id":    "…",          // the conversation this decision belongs to
  "decision_id":   "…",          // echo both back unchanged
  "agent_id":      "assistant",  // which agent the engine is asking
  "identity":      { "tenant_id": "…", "id": "user-42" },  // the end user
  "trigger":       { "type": "…", … },   // the news; the one field you switch on
  "state":         { … },        // your state as raw JSON (any JSON value)
  "calls": [                     // calls still in flight, a flat tagged list, see below
    { "id": "…", "kind": "tool_call", "status": "pending", "attempt": 0,
      "name": "get_weather", "arguments": "{…}", "handler": "worker" }
  ],
  "pending_calls": 1,            // tool_call/sub_agent calls still in flight: the step gate

  "messages":  [ { "role": "user", "content": "hi", "id": "…" }, … ],
  "turn_id":     "…",
  "attempts":    0
}
```

- **`trigger`** is what you switch on. It's the one field that says what happened.
- **`messages`** is the active conversation as a flat list, oldest first: all a
  tool loop needs. (`message_tree` carries the full branch structure.)
- **`pending_calls`** is the step gate as a number: how many
  `tool_call`/`sub_agent` calls are still in flight.
  `llm_call` calls don't count toward it; they don't block the next prompt.
- **`calls`** is the same in-flight calls as a flat, tagged list, for workers
  that need more than the count. Each is a stable envelope: `id`, `kind`
  (`"tool_call"` | `"sub_agent"` | `"llm_call"` | …), `status` (`"pending"` |
  `"retry_scheduled"`), `attempt`, and `anchor` (the tree node the call was
  requested at), plus kind-specific fields (a tool's `name`/`arguments`, a
  sub-agent's `agent_id`/`session_id`). `kind` and `status` are **open**: a worker
  ignores kinds it doesn't handle, so new call kinds (timers, approvals) and new
  statuses are additive, never a wire break.
- **`state`** is your state as raw JSON, opaque to the engine: it round-trips it
  untouched. Read it on the way in, return the next value on the way out under the
  same name. `null` when you keep state in your own database.

### Decision (worker → engine)

```jsonc
{
  "session_id":  "…",          // from the request
  "decision_id": "…",          // from the request
  "actions":     [ … ],        // what to do next; see Actions
  "messages":    [ … ],        // the conversation as it should now read
  "state":       { … }         // your state as raw JSON; omit the field to keep it. See State.
}
```

The engine carries out the `actions`, reconciles the `messages` into the
conversation tree (messages with a known `id` continue the branch; id-less or
unknown messages are appended, forking automatically), records everything, and
calls back with the next decision when there's something to react to. That loop,
request, decide, act, request again, is the agent loop.

**Empty actions end the exchange.** The engine only calls back when something
happens — a client message, or a call you scheduled finishing. A decision with
no actions (and nothing left in flight) parks the session: the engine records the
transcript and state and waits, it does not poll or nudge you to make more
progress. So finish a turn deliberately with a `done` action; returning no actions
just means "nothing to do right now," not "the turn is over."

## Triggers

What the engine sends in `trigger`. A worker handles the ones it cares about and
ignores the rest — unknown types are additive, never a wire break.

| `type` | Fields | Meaning |
|---|---|---|
| `client.messages` | `messages`, `new_from` | The client proposed the conversation — a new message, an edit, or a full view, all one shape. `messages` is the full proposed transcript; `messages[new_from..]` is unrecorded. Echo it (or amend it) and prompt. |
| `client.action` | `name`, `args?` | A non-message signal from the client (mode switch, approval). |
| `llm.finished` | `id`, `ok`, `message`, `truncated`, `usage?`, `cost?` \| `error`, `code?`, `detail?` | The model's final word on call `id`. |
| `tool.finished` | `id`, `ok`, `name`, `result` \| `error` | A tool's final word (retries exhausted on failure). |
| `sub_agent.finished` | `id`, `ok`, `session_id`, `agent_id`, `result` \| `error` | A child agent's turn ended. `id` is the model call it answers — the same id you'd fold a tool result under; `session_id` is the child session. |
| `tool.execute` | `id`, `name`, `arguments`, `attempt`, `deadline?` | Run this tool yourself and answer with `tool.result`/`tool.error`. |
| `llm.execute` | `id`, `request`, `stream`, `attempt`, `deadline?` | Make this model call yourself (worker-handled model) and answer with `llm.result`/`llm.error`. |
| `interrupt.resumed` | `interrupt_id`, `payload?` | A paused session resumed. |

`ok` is uniform across the `*.finished` triggers: the payload when true, `error`
when not. **Finished means final** — you will never hear about that call id
again, and every `*.finished` you receive has its requesting call on the
delivered transcript (see Forks and in-flight calls).

### The client transcript

All conversational input arrives as one trigger. A client that submits a bare
message (`{"type": "message", ...}`) and a client that submits its full view
(`{"type": "messages", ...}`, e.g. AG-UI) produce the same shape on your wire:
the **full proposed conversation** in `messages`, with `new_from` marking where
the unrecorded suffix starts. The engine computes both at delivery time against
the tree, so a message that queued behind another decision is materialized
*after* that decision's writes — the annotation is exact, not advisory, and
identical across redeliveries.

The simple worker ignores `new_from` and echoes the trigger's `messages` straight
back as its decision's `messages`, then prompts. A worker with per-turn policy gets
everything the annotation implies:

- **The news**: `messages[new_from..]` — exactly what reconciling your echo will
  write. Moderate it, transform it (the news' ids are assigned but unrecorded, so
  you may rewrite content before echoing), or bill by it.
- **Append vs. edit**: the proposal is a pure append iff
  `messages[new_from - 1].id` is the last id of the request's `messages`;
  anything else forks.
- **Reject**: return the request's `messages` unchanged (or omit `messages`)
  and the proposal is never written — not echoing is the veto.
- **Empty news** (`new_from == len(messages)`): nothing to write. A strict-prefix
  proposal is a *regenerate* gesture — prompt from it and the fork lands when
  your next assistant message reconciles.

Three contracts keep client views well-formed:

- **An edit is a new message.** Reconcile keys on id: a message that keeps its
  recorded id *is* the recorded node, content changes ignored. Edits must carry a
  fresh (or no) id.
- **The news doesn't stop.** Once reconcile writes its first new node, every
  later message in the echo is written as a new node (a known id past that point
  is re-recorded fresh) — a fork rewrites its whole tail.
- **Full views must echo recorded ids.** An id-less resend of the whole
  conversation is all news and forks at the root, every turn. Clients that don't
  track ids should submit bare messages instead.

A client transcript may also answer **client-handled tool calls**: any message
whose `tool_call_id` matches a pending client tool settles that call. How the
submission then delivers depends on its shape, decided by the same reconcile plan
that will write your echo:

- **Fast path** — the submission's only change is exactly one answer to one
  pending call. The engine runs the structured flow: it settles the call and
  sends you a `tool.finished`, identical to the settle endpoint. You append the
  result node and prompt; the submitted view is discarded.
- **Bedrock** — anything else (several answers at once, an answer plus a new
  message, an edit, a whole imported history). Every answered call settles
  silently — **no `tool.finished`** — and the entire view arrives as one
  `client.messages`. Your echo records the client's tool messages along with
  the rest.

Before either, the engine **normalizes**: a client that resends its own copy of
an already-recorded tool result has that message folded onto the recorded node
(matched by `tool_call_id`), so local-memory clients that replay their tool
messages every turn resend rather than fork. (AG-UI clients must therefore key
tool messages by `tool_call_id`, per the protocol; the settle echoes back a
`TOOL_CALL_RESULT` under that id.)

While a session is interrupted, a submission that settles pending tools is
accepted — the resulting decision (fast-path `tool.finished` or bedrock
transcript) queues until resume; one that settles nothing is rejected.

One deliberate limitation: an echo of entirely-known ids writes nothing, so a
**pure head move** (truncating or switching branches with no new message and no
follow-up prompt) is a no-op — branches materialize only when a new node lands.

## Actions

What the worker returns in `actions`. Each is a plain tagged object; building one
is just constructing the struct.

| `type` | Fields | Effect |
|---|---|---|
| `llm.call` | `id`, `request`, `handler`, `stream?`, `retry?` | Make an LLM call named by `id` (like `tool.call`). `handler` is required: `"server"` lets the engine call its provider; `"worker"` hands the call back to you as an `llm.execute` trigger. You may emit several in one decision, and each finishes independently as its own `llm.finished`. |
| `tool.call` | `id`, `name`, `arguments`, `handler`, `retry?` | Schedule a tool call named by `id` (the model's tool call id). `handler: "worker"` runs on your worker (you'll get a `tool.execute` trigger); `handler: "client"` routes it to the browser. |
| `tool.result` | `id`, `result`, `attempt?` | Answer a `tool.execute` with the tool's output. Omit `attempt` to settle the current attempt; echo the trigger's to fence out a stale executor. |
| `llm.result` | `id`, `response`, `attempt?` | Answer an `llm.execute` with the provider's `LlmResponse`. |
| `tool.error` / `llm.error` | `id`, `error`, `retryable`, `attempt?`, `code?`, `detail?` | Answer an execute with a failure. |
| `sub_agent.spawn` | `session_id`, `agent_id`, `tool_call_id`, `retry?` | Start a child agent in a new session, linked to the tool call that requested it. |
| `message.send` | `session_id`, `message` | Deliver a message to a session; used to seed a spawned sub-agent. |
| `interrupt` | `interrupt_id?`, `reason`, `payload?` | Pause the session. |
| `done` | `data` | End the turn with a result. |

Each subject reads as one lifecycle, in both directions:

```
you say            engine may delegate      you answer                    everyone hears
llm.call     ──▶   llm.execute        ──▶   llm.result | llm.error  ──▶   llm.finished
tool.call    ──▶   tool.execute       ──▶   tool.result | tool.error ──▶  tool.finished
sub_agent.spawn + message.send ──────────────────────────────────────▶   sub_agent.finished
```

## Message shapes

The types the triggers and actions carry. All snake_case on the wire.

```jsonc
// Message: a conversation turn.
{
  "id":           "…",          // node id; omit on a message you're creating
  "role":         "user" | "assistant" | "tool" | "system",
  "content":      "…",          // string, or an array of content parts
  "tool_calls":   [ ToolCall ], // on an assistant message that calls tools
  "tool_call_id": "…",          // on a tool message: which call it answers
  "name":         "…"           // on a tool message: the tool's name
}

// ToolCall: one requested call inside an assistant message.
{ "id": "call_abc", "type": "function", "function": { "name": "getWeather", "arguments": "{\"city\":\"NYC\"}" } }

// LlmRequest: the prompt you send with llm.call.
{ "model": "anthropic/claude-sonnet-4-6", "messages": [ Message ], "tools": [ LlmTool ] }

// LlmTool: a tool as the model sees it.
{ "function": { "name": "getWeather", "description": "…", "parameters": { /* JSON Schema */ } } }

// LlmResponse: what a worker-run model returns (see llm.execute).
{ "model": "…", "content": "…", "tool_calls": [ ToolCall ], "finish_reason": "…" }
```

## The tool loop, minimally

The reference implementation, as pseudocode. Two layers: the **boundary** (read
state, run the loop, return state) and the **loop** itself (a switch on the
trigger). Both together are ~40 lines; this is what you port.

```python
# ── Boundary: the HTTP handler. State is raw JSON, in and out. ──
def handle(request):
    state = request.get("state")                     # raw JSON; None when the session has none
    out   = decide(request, state)                   # run the loop
    resp  = {
        "session_id":  request["session_id"],
        "decision_id": request["decision_id"],
        "actions":     out.get("actions", []),
        "messages":    out.get("messages", request.get("messages", [])),
    }
    if "state" in out:
        resp["state"] = out["state"]     # omitted = keep; echoes are deduped engine-side
    return resp

# ── Loop: a pure function of (trigger, messages, pending_calls) → decision. ──
def decide(req, state):
    trigger    = req["trigger"]
    history    = req.get("messages", [])
    tools      = { t.name: t for t in MY_TOOLS }         # your tool registry
    schemas    = [ tool_schema(t) for t in MY_TOOLS ]    # tools as the model sees them

    match trigger["type"]:

        # The client proposed the conversation → write it in, prompt the model.
        case "client.messages":
            messages = trigger["messages"]
            return { "messages": messages, "actions": [ call_llm(messages, schemas) ] }

        # The model's final word → finish the turn, or run its tools.
        case "llm.finished":
            if not trigger["ok"]:
                return { "actions": [ done({ "error": trigger["error"] }) ] }
            assistant  = trigger["message"]
            messages   = history + [ assistant ]
            calls      = assistant.get("tool_calls", [])
            if not calls:
                return { "messages": messages, "actions": [ done(assistant.get("content")) ] }
            actions = [ call_tool(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in calls ]
            return { "messages": messages, "actions": actions }

        # The engine hands a tool to us → run it, answer.
        case "tool.execute":
            try:
                result = tools[ trigger["name"] ].run( trigger["arguments"] )
                return { "actions": [ tool_result(trigger["id"], result) ] }
            except Exception as e:
                return { "actions": [ tool_error(trigger["id"], str(e)) ] }

        # A tool's final word → write it in; prompt again once the step is done.
        case "tool.finished":
            content    = trigger["result"] if trigger["ok"] else trigger["error"]
            node       = { "role": "tool", "content": content,
                           "tool_call_id": trigger["id"], "name": trigger["name"] }
            messages   = history + [ node ]
            if req["pending_calls"] > 0:
                return { "messages": messages }                              # siblings still running, just record
            return { "messages": messages, "actions": [ call_llm(messages, schemas) ] }

        case _:
            return {}

# Actions are plain data; the "builders" just construct the tagged struct.
def call_llm(messages, tools):     return { "type": "llm.call", "id": fresh_id(), "request": { "model": MODEL, "messages": [system_message()] + messages, "tools": tools }, "handler": "server" }
def call_tool(id, name, args):     return { "type": "tool.call", "id": id, "name": name, "arguments": args, "handler": "worker" }
def tool_result(id, result):       return { "type": "tool.result", "id": id, "result": result }
def tool_error(id, error):         return { "type": "tool.error", "id": id, "error": error, "retryable": False }
def done(data):                    return { "type": "done", "data": data }

# system_message(): { "role": "system", "content": INSTRUCTIONS }. Instructions are
# worker config, prepended to each llm.call prompt at build time — never stored in the
# transcript (the tree is pure conversation). The exact prompt sent is durably recorded
# on the llm.call.requested event; see Patterns / System prompt.
```

That's the entire loop: four flat cases, one per trigger type, and
`trigger["type"]` is the only discriminator in the whole protocol.
`client.messages` prompts, `llm.finished` drives the loop, `tool.execute` runs
delegated work, and `tool.finished` folds results in (prompting again once
nothing is pending). Everything below is an optional extension on the same
skeleton.

## Extensions

**Sub-agents.** Present each child agent to the model as a tool (an `LlmTool`
whose `parameters` is `{ message: string }`). When the model calls one, instead of
`tool.call`, emit two actions: `sub_agent.spawn` (a fresh child `session_id`, the
child's `agent_id`, and the originating `tool_call_id`) and `message.send` (that
same `session_id`, a user message with the unwrapped `message` argument). The
child's turn runs in its own session; when it finishes, the engine sends the
parent a `sub_agent.finished` whose `id` is that same `tool_call_id` — the shape
a tool produces, so the loop above handles the result with one more `case` that
folds it in exactly like `tool.finished` (using `agent_id` as the name).

**Worker-run models.** If you call the LLM provider yourself rather than letting
the engine do it, set `handler: "worker"` on your `llm.call`. The engine then
sends you an `llm.execute`; run the call and reply with `llm.result` (or
`llm.error`). This is also where token streaming lives: the request's transport
may hold the connection open so you can push token deltas as they arrive, then
return the final `LlmResponse`. Server-handled models (`handler: "server"`)
stream on the engine side and never delegate the call.

**Parallel LLM calls.** A worker may fan out several `llm.call` actions in one
decision, each with its own `id`, and they are all in flight at once, each
finishing independently. Rules:

- **Concurrency lives in calls, never in decisions.** You never receive two
  decisions at once; the resulting `llm.finished` triggers arrive serialized in
  *completion* order (not request order), one decision at a time. Fold each in and
  branch on the `calls` list. `pending_calls` counts only `tool_call`/
  `sub_agent` calls, so llm finishes never gate the loop; drive your own logic
  off `calls` when you fan out model calls.
- **Ids MUST be fresh per logical call.** Reusing a pending/completed id is an
  idempotent no-op (deliberate: it makes decision redelivery and retry-after-
  interrupt safe). An accidentally reused id silently loses the call.
- **Engine-handled fan-out:** emit N `llm.call { handler: "server" }`; each
  returns as its own `llm.finished`, no async machinery in your worker. The
  engine *executes* a session's calls one at a time (a deliberate bound on
  per-session provider pressure), so this decouples your loop but doesn't shrink
  wall-clock time. Note that a call's retry `timeout_secs` clock starts when the
  call is requested, and keeps ticking while it waits behind its siblings, so give
  fanned-out engine-handled calls generous deadlines.
- **Worker-handled fan-out (deferred):** on each `llm.execute`, start the
  provider call in the background and return the decision immediately with no
  actions (the next `llm.execute` promotes at once). Settle each call whenever it
  finishes by POSTing an `llm.result`/`llm.error` to
  `/api/machine/sessions/{id}/calls/settle`. This is the path to true wall-clock
  overlap, since your worker owns the provider connection and its concurrency.
  Echoing each trigger's `attempt` fences out settles from a stale executor after
  a retry; omit it to settle whatever attempt is current. Engine-handled
  (`handler: "server"`) calls are never externally settleable.
- **`done` with llm calls still in flight** is allowed; a late finish simply fires
  a new decision after the turn ends. Don't fan out `stream: true` behind an AG-UI
  front-end, because concurrent token streams interleave on a single message channel.

**Forks and in-flight calls.** Every call records an `anchor`: the tree node
that was the active head when it was requested. A fork abandons the branch it
leaves, and outstanding work anchored below the fork point dies with it: the
engine voids those calls (`call.voided` on the session log) and rejects any
late settle for them. Work anchored on the still-shared prefix is untouched.
A voided sub-agent delegation cancels the child session, recursively through
its sub-tree. Workers need no staleness checks — every `*.finished` you
receive has its requesting call on the delivered transcript.

**Stop conditions.** To cap the loop (e.g. stop after N assistant steps), check
your condition in `tool.finished` before prompting again and emit `done` instead
of `llm.call`.

**Client actions & modes.** Handle `client.action` to react to non-message signals
(a mode switch, an approval, a cancel) without them appearing in the transcript.
This is how human-in-the-loop approval and modal (plan → execute) agents work; see
[Patterns](./06-patterns.md).

## State

`state` is a raw JSON value the engine round-trips untouched: delivered on the
request, stored from the decision, both under the same name. State persists
across decisions without the engine ever reading it. If you'd rather keep
state in your own database, leave it empty and load/save around the loop, keyed by
`identity.id` or `session_id`. See [SDK / State](./04-sdk.md#state).

**Submitting state is a PUT with dedup.** On each decision:

- **Omitted `state` field** — or a present `null` — means "no opinion, keep the
  current state." The wire field deserializes as `Option`, so `null` collapses to
  the same "keep" as omission; only a present non-null value carries an opinion.
- **A present non-null value** replaces the state — but a value equal to the
  currently resolved state writes nothing, so echoing state on every decision is free.
- **Explicit clear** is a present non-null empty value, e.g. `{}` (any value you
  treat as empty). A present `null` does *not* clear — it keeps.

Dedup equality is structural JSON equality on parsed values: object key order and
wire formatting never matter; arrays are ordered (`[1,2] != [2,1]`); `null` is not
absence (`{"a": null} != {}`); numbers compare by parsed variant (integer `1` !=
float `1.0`).

State is **branch-scoped**: each write is a version anchored to the tree position
it was made at, and the `state` a decision carries is resolved from the
active branch — the newest version whose anchor is on the root-to-head path. When
the transcript forks to an earlier point, the new branch sees state as of the fork
point; submit state on the forking decision to carry it over instead. See
[Concepts / State](./02-concepts.md#state-is-branch-scoped).
