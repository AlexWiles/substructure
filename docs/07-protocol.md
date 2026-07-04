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

## The exchange

The engine `POST`s a JSON **decision request** to your worker's endpoint. Your
worker responds with a JSON **decision**. One request, one response, once per
decision.

### Decision request (engine → worker)

```jsonc
{
  "session_id":  "…",          // the conversation this decision belongs to
  "agent_id":    "assistant",  // which agent the engine is asking
  "decision_id": "…",          // echo this back unchanged
  "tenant_id":   "…",
  "identity":    { "tenant_id": "…", "id": "user-42" },  // the end user
  "trigger":     { "type": "…", … },   // what happened; see Triggers
  "worker_state": "eyJ…",      // arbitrary state string
  "effects": [                 // effects still in flight, a flat tagged list, see below
    { "id": "…", "kind": "tool_call", "status": "pending", "attempt": 0,
      "name": "get_weather", "arguments": "{…}", "handler": "worker" }
  ],
  "pending_effects": 1,        // tool_call/sub_agent effects still in flight: the step gate

  "transcript":  [ { "role": "user", "content": "hi", "id": "…" }, … ],
  "turn_id":     "…",
  "attempts":    0
}
```

- **`trigger`** is what you switch on. It's the one field that says what happened.
- **`transcript`** is the active conversation as a flat list, oldest first: all a
  tool loop needs. (`message_tree` carries the full branch structure.)
- **`pending_effects`** is the step gate as a number: how many
  `tool_call`/`sub_agent` effects are still in flight.
  `llm_call` effects don't count toward it; they don't block the next prompt.
- **`effects`** is the same in-flight effects as a flat, tagged list, for workers
  that need more than the count. Each is a stable envelope: `id`, `kind`
  (`"tool_call"` | `"sub_agent"` | `"llm_call"` | …), `status` (`"pending"` |
  `"retry_scheduled"`), and `attempt`, plus kind-specific fields (a tool's
  `name`/`arguments`, a sub-agent's `agent_id`/`session_id`). `kind` and `status`
  are **open**: a worker ignores kinds it doesn't handle, so new effect kinds
  (timers, approvals) and new statuses are additive, never a wire break.
- **`worker_state`** is your state, opaque to the engine. Decode it on the way in,
  encode it on the way out. Empty when you keep state in your own database.

### Decision (worker → engine)

```jsonc
{
  "session_id":  "…",          // from the request
  "decision_id": "…",          // from the request
  "actions":     [ … ],        // what to do next; see Actions
  "transcript":  [ … ],        // the conversation as it should now read
  "state":       "eyJ…"        // base64(JSON) of your state
}
```

The engine carries out the `actions`, reconciles the `transcript` into the
conversation tree (messages with a known `id` continue the branch; id-less or
unknown messages are appended, forking automatically), records everything, and
calls back with the next decision when there's something to react to. That loop,
request, decide, act, request again, is the agent loop.

**Empty actions end the exchange.** The engine only calls back when something
happens — a client message, or an effect you scheduled settling. A decision with
no actions (and nothing left in flight) parks the session: the engine records the
transcript and state and waits, it does not poll or nudge you to make more
progress. So finish a turn deliberately with a `done` action; returning no actions
just means "nothing to do right now," not "the turn is over."

## Triggers

What the engine sends in `trigger`. A worker handles the ones it cares about and
ignores the rest.

| `type` | Fields | Meaning |
|---|---|---|
| `user.message` | `message` | A new user turn. Prompt the model. |
| `user.transcript` | `messages` | A full client-supplied transcript (e.g. AG-UI). Reconcile it and prompt. |
| `client.action` | `name`, `args?` | A non-message signal from the client (mode switch, approval). |
| `effect.execute` | `kind`, `id`, `attempt`, `deadline?`, + work | Run this effect's work yourself and answer with `effect.result`/`effect.error`. |
| `effect.settled` | `kind`, `id`, `ok`, + outcome | An effect landed (successfully or not). Fold it in. |
| `interrupt.resumed` | `interrupt_id`, `payload?` | A paused session resumed. |

The two effect triggers use the same `kind` discriminator as the `effects` list,
so one correlation model covers the whole protocol: an effect is named by `id`,
described by `kind`, and every message about it, whether in flight, delegated to
you, or settled, carries that same pair.

**`effect.execute` work**, by `kind`:

| `kind` | Fields | Work |
|---|---|---|
| `tool_call` | `name`, `arguments` | Run the tool. |
| `llm_call` | `request`, `stream` | Make the LLM call (worker-handled model). |

**`effect.settled` outcome**, by `kind`. `ok` says whether it succeeded:

| `kind` | Fields | Outcome |
|---|---|---|
| `tool_call` | `name`, `result` | A tool completed; `result` is the error text when `ok` is false, and it folds into the transcript either way. |
| `sub_agent` | `tool_call_id`, `agent_id`, `result` | A sub-agent's turn completed. `id` is its session; `tool_call_id` is the model call it answers. |
| `llm_call` | `message`, `truncated`, `usage?`, `cost?` (or `error`, `code?`, `detail?` when `ok` is false) | The model replied (or the call failed after retries). |

## Actions

What the worker returns in `actions`. Each is a plain tagged object; building one
is just constructing the struct.

| `type` | Fields | Effect |
|---|---|---|
| `call.llm` | `id`, `request`, `handler`, `stream?`, `retry?` | Make an LLM call named by `id` (like `call.tool`). `handler` is required: `"server"` lets the engine call its provider; `"worker"` hands the call back to you as an `effect.execute` trigger. You may emit several in one decision, and each settles independently as its own `effect.settled`. |
| `call.tool` | `id`, `name`, `arguments`, `handler`, `retry?` | Schedule a tool call named by `id` (the model's tool call id). `handler: "worker"` runs on your worker (you'll get an `effect.execute` trigger); `handler: "client"` routes it to the browser. |
| `effect.result` | `kind`, `id`, `attempt`, + result | Answer an `effect.execute`: a `tool_call` carries `result`, an `llm_call` carries `response`. |
| `effect.error` | `kind`, `id`, `attempt`, `error`, `retryable`, `code?`, `detail?` | Answer an `effect.execute` with a failure; uniform across kinds. |
| `spawn.sub_agent` | `session_id`, `agent_id`, `tool_call_id`, `retry?` | Start a child agent in a new session, linked to the tool call that requested it. |
| `send.message` | `session_id`, `message` | Deliver a message to a session; used to seed a spawned sub-agent. |
| `interrupt` | `interrupt_id?`, `reason`, `payload?` | Pause the session. |
| `done` | `data` | End the turn with a result. |

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

// LlmRequest: the prompt you send with call.llm.
{ "model": "anthropic/claude-sonnet-4-6", "messages": [ Message ], "tools": [ LlmTool ] }

// LlmTool: a tool as the model sees it.
{ "function": { "name": "getWeather", "description": "…", "parameters": { /* JSON Schema */ } } }

// LlmResponse: what a worker-run model returns (see effect.execute / llm_call).
{ "model": "…", "content": "…", "tool_calls": [ ToolCall ], "finish_reason": "…" }
```

## The tool loop, minimally

The reference implementation, as pseudocode. Two layers: the **boundary** (decode
state, run the loop, encode state) and the **loop** itself (a switch on the
trigger). Both together are ~40 lines; this is what you port.

```python
# ── Boundary: the HTTP handler. Decode state in, encode state out. ──
def handle(request):
    state = decode(request["worker_state"])          # base64(JSON) → your object
    out   = decide(request, state)                   # run the loop
    return {
        "session_id":  request["session_id"],
        "decision_id": request["decision_id"],
        "actions":     out.get("actions", []),
        "transcript":  out.get("transcript", request.get("transcript", [])),
        "state":       encode(out.get("state", state)),   # echo state if unchanged
    }

# ── Loop: a pure function of (trigger, transcript, pending_*) → decision. ──
def decide(req, state):
    trigger    = req["trigger"]
    history    = req.get("transcript", [])
    tools      = { t.name: t for t in MY_TOOLS }         # your tool registry
    schemas    = [ tool_schema(t) for t in MY_TOOLS ]    # tools as the model sees them

    match trigger["type"]:

        # A new user turn: prompt the model.
        case "user.message":
            transcript = ensure_system(history) + [ trigger["message"] ]
            return { "transcript": transcript, "actions": [ call_llm(transcript, schemas) ] }

        # An effect landed. The model replied → finish or run its tools;
        # a tool/sub-agent finished → record it, prompt once the step is done.
        case "effect.settled":
            if trigger["kind"] == "llm_call":
                assistant  = trigger["message"]
                transcript = history + [ assistant ]
                calls      = assistant.get("tool_calls", [])
                if not calls:
                    return { "transcript": transcript, "actions": [ done(assistant.get("content")) ] }
                actions = [ call_tool(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in calls ]
                return { "transcript": transcript, "actions": actions }

            call_id    = trigger["tool_call_id"] if trigger["kind"] == "sub_agent" else trigger["id"]
            name       = trigger["agent_id"]     if trigger["kind"] == "sub_agent" else trigger["name"]
            node       = { "role": "tool", "content": trigger["result"],
                           "tool_call_id": call_id, "name": name, "id": new_id() }
            transcript = history + [ node ]
            if req["pending_effects"] > 0:
                return { "transcript": transcript }                          # step still open, just record
            return { "transcript": transcript, "actions": [ call_llm(transcript, schemas) ] }

        # The engine wants YOU to run the effect's work (here: a tool).
        case "effect.execute":
            try:
                result = tools[ trigger["name"] ].run( trigger["arguments"] )
                return { "actions": [ effect_result(trigger["id"], result) ] }
            except Exception as e:
                return { "actions": [ effect_error(trigger["id"], str(e)) ] }

        case _:
            return {}

# Actions are plain data; the "builders" just construct the tagged struct.
def call_llm(messages, tools):     return { "type": "call.llm", "id": fresh_id(), "request": { "model": MODEL, "messages": messages, "tools": tools }, "handler": "server" }
def call_tool(id, name, args):     return { "type": "call.tool", "id": id, "name": name, "arguments": args, "handler": "worker" }
def effect_result(id, result):     return { "type": "effect.result", "kind": "tool_call", "id": id, "result": result, "attempt": 0 }
def effect_error(id, error):       return { "type": "effect.error", "kind": "tool_call", "id": id, "error": error, "retryable": False, "attempt": 0 }
def done(data):                    return { "type": "done", "data": data }

# ensure_system(history): if history doesn't start with a system message, prepend
# one carrying your instructions (with a fresh id). Otherwise return it unchanged.
```

That's the entire loop: three cases. `user.message` prompts, `effect.settled`
folds outcomes in (dispatching tools when the model replies, prompting again
once nothing is pending), and `effect.execute` runs delegated work. Everything
below is an optional extension on the same skeleton.

## Extensions

**Sub-agents.** Present each child agent to the model as a tool (an `LlmTool`
whose `parameters` is `{ message: string }`). When the model calls one, instead of
`call.tool`, emit two actions: `spawn.sub_agent` (a fresh child `session_id`, the
child's `agent_id`, and the originating `tool_call_id`) and `send.message` (that
same `session_id`, a user message with the unwrapped `message` argument). The
child's turn runs in its own session; when it finishes, the engine sends the
parent an `effect.settled` with `kind: "sub_agent"`, the same trigger a tool
produces, so the loop above handles the result with no extra case.

**Worker-run models.** If you call the LLM provider yourself rather than letting
the engine do it, set `handler: "worker"` on your `call.llm`. The engine then
sends you an `effect.execute` with `kind: "llm_call"`; run the call and reply with
`effect.result` (or `effect.error`). This is also where token streaming lives:
the request's transport may hold the connection open so you can push token deltas
as they arrive, then return the final `LlmResponse`. Server-handled models
(`handler: "server"`) stream on the engine side and never delegate the call.

**Parallel LLM calls.** A worker may fan out several `call.llm` actions in one
decision, each with its own `id`, and they are all in flight at once, each
settling independently. Rules:

- **Concurrency lives in effects, never in decisions.** You never receive two
  decisions at once; the resulting `effect.settled` triggers arrive serialized in
  *completion* order (not request order), one decision at a time. Fold each in and
  branch on the `effects` list. `pending_effects` counts only `tool_call`/
  `sub_agent` effects, so llm settles never gate the loop; drive your own logic
  off `effects` when you fan out model calls.
- **Ids MUST be fresh per logical call.** Reusing a pending/completed id is an
  idempotent no-op (deliberate: it makes decision redelivery and retry-after-
  interrupt safe). An accidentally reused id silently loses the call.
- **Engine-handled fan-out:** emit N `call.llm { handler: "server" }`; each
  returns as its own `effect.settled`, no async machinery in your worker. The
  engine *executes* a session's calls one at a time (a deliberate bound on
  per-session provider pressure), so this decouples your loop but doesn't shrink
  wall-clock time. Note that a call's retry `timeout_secs` clock starts when the
  call is requested, and keeps ticking while it waits behind its siblings, so give
  fanned-out engine-handled calls generous deadlines.
- **Worker-handled fan-out (deferred):** on each `effect.execute { kind: "llm_call" }`,
  start the provider call in the background and return the decision immediately
  with no actions (the next `effect.execute` promotes at once). Settle each call
  whenever it finishes by POSTing an `effect.result`/`effect.error` to
  `/api/machine/sessions/{id}/effects/settle`. This is the path to true wall-clock
  overlap, since your worker owns the provider connection and its concurrency. Echo
  the `attempt` from each call's own trigger. Engine-handled (`handler: "server"`)
  calls are never externally settleable.
- **`done` with llm calls still in flight** is allowed; a late settle simply fires
  a new decision after the turn ends. Don't fan out `stream: true` behind an AG-UI
  front-end, because concurrent token streams interleave on a single message channel.

**Stop conditions.** To cap the loop (e.g. stop after N assistant steps), check
your condition in the tool branch of `effect.settled` before prompting again and
emit `done` instead of `call.llm`.

**Client actions & modes.** Handle `client.action` to react to non-message signals
(a mode switch, an approval, a cancel) without them appearing in the transcript.
This is how human-in-the-loop approval and modal (plan → execute) agents work; see
[Patterns](./06-patterns.md).

## State

`worker_state` is a base64-encoded JSON blob the engine round-trips untouched. The
boundary decodes it into the request and re-encodes whatever the decision returns,
so state persists across decisions without the engine ever reading it. If you'd
rather keep state in your own database, leave it empty and load/save around the
loop, keyed by `identity.id` or `session_id`. See [SDK / State](./04-sdk.md#state).
