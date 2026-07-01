---
title: The decision protocol
---

The engine and your worker talk over one HTTP call, repeated: the engine sends a
**decision request** describing what just happened, your worker replies with a
**decision** — the actions to take and the conversation as it should now read.
That request/response contract is the whole protocol. The TypeScript SDK's
`toolLoop` is one implementation of it; this page is the language-neutral spec, so
you can implement a worker — and the tool loop itself — in any language.

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
  "trigger":     { "type": "…", … },   // what happened — see Triggers
  "worker_state": "eyJ…",      // arbitrary state string
  "pending":     { "tool_calls": 0, "sub_agents": 0, "llm_calls": 0 },
  "transcript":  [ { "role": "user", "content": "hi", "id": "…" }, … ],
  "turn_id":     "…",
  "attempts":    0
}
```

- **`trigger`** is what you switch on. It's the one field that says what happened.
- **`transcript`** is the active conversation as a flat list, oldest first — all a
  tool loop needs. (`message_tree` carries the full branch structure for clients
  that need it; workers can ignore it.)
- **`pending`** is the count of effects still in flight — LLM calls, tool calls,
  and sub-agent turns the engine is still waiting on. Branch on these instead of
  tracking rounds yourself: when they're all zero, the round is complete.
- **`worker_state`** is your state, opaque to the engine. Decode it on the way in,
  encode it on the way out. Empty when you keep state in your own database.

### Decision (worker → engine)

```jsonc
{
  "session_id":  "…",          // from the request
  "decision_id": "…",          // from the request
  "actions":     [ … ],        // what to do next — see Actions
  "transcript":  [ … ],        // the conversation as it should now read
  "state":       "eyJ…"        // base64(JSON) of your state
}
```

The engine carries out the `actions`, reconciles the `transcript` into the
conversation tree (messages with a known `id` continue the branch; id-less or
unknown messages are appended, forking automatically), records everything, and
calls back with the next decision when there's something to react to. That loop —
request, decide, act, request again — is the agent loop.

## Triggers

What the engine sends in `trigger`. A worker handles the ones it cares about and
ignores the rest.

| `type` | Fields | Meaning |
|---|---|---|
| `user.message` | `message` | A new user turn. Prompt the model. |
| `user.transcript` | `messages` | A full client-supplied transcript (e.g. AG-UI). Reconcile it and prompt. |
| `client.action` | `name`, `args?` | A non-message signal from the client (mode switch, approval). |
| `llm.response` | `call_id`, `message`, `truncated`, `usage?`, `cost?` | The model replied. Record it; call tools or finish. |
| `llm.error` | `call_id`, `error`, `code?`, `detail?` | An LLM call failed after retries. |
| `llm.request` | `call_id`, `request`, `stream`, `attempt` | Run this LLM call yourself (worker-handled model). |
| `tool.execute` | `tool_call_id`, `name`, `arguments`, `attempt`, `deadline?` | Run this tool yourself and return the result. |
| `tool.result` | `tool_call_id`, `name`, `result`, `is_error?` | A tool or sub-agent completed. |
| `interrupt.resumed` | `interrupt_id`, `payload?` | A paused session resumed. |
| `stall` | — | Nothing is pending; a nudge to make progress. |

## Actions

What the worker returns in `actions`. Each is a plain tagged object — building one
is just constructing the struct.

| `type` | Fields | Effect |
|---|---|---|
| `call.llm` | `request`, `stream?`, `retry?`, `handler?` | Make an LLM call. `handler: "server"` (default) lets the engine call its provider; `handler: "worker"` hands the call back to you as an `llm.request` trigger. |
| `call.tool` | `tool_call_id`, `name`, `arguments`, `handler`, `retry?` | Schedule a tool call. `handler: "worker"` runs on your worker (you'll get a `tool.execute` trigger); `handler: "client"` routes it to the browser. |
| `return.tool.result` | `tool_call_id`, `result`, `attempt` | A worker-run tool finished. |
| `return.tool.error` | `tool_call_id`, `error`, `retryable`, `attempt` | A worker-run tool failed. |
| `return.llm.result` | `call_id`, `response`, `attempt` | A worker-run LLM call finished. |
| `return.llm.error` | `call_id`, `error`, `retryable`, `code?`, `detail?`, `attempt` | A worker-run LLM call failed. |
| `spawn.sub_agent` | `session_id`, `agent_id`, `tool_call_id`, `retry?` | Start a child agent in a new session, linked to the tool call that requested it. |
| `send.message` | `session_id`, `message` | Deliver a message to a session — used to seed a spawned sub-agent. |
| `interrupt` | `interrupt_id?`, `reason`, `payload?` | Pause the session. |
| `done` | `data` | End the turn with a result. |

## Message shapes

The types the triggers and actions carry. All snake_case on the wire.

```jsonc
// Message — a conversation turn.
{
  "id":           "…",          // node id; omit on a message you're creating
  "role":         "user" | "assistant" | "tool" | "system",
  "content":      "…",          // string, or an array of content parts
  "tool_calls":   [ ToolCall ], // on an assistant message that calls tools
  "tool_call_id": "…",          // on a tool message: which call it answers
  "name":         "…"           // on a tool message: the tool's name
}

// ToolCall — one requested call inside an assistant message.
{ "id": "call_abc", "type": "function", "function": { "name": "getWeather", "arguments": "{\"city\":\"NYC\"}" } }

// LlmRequest — the prompt you send with call.llm.
{ "model": "anthropic/claude-sonnet-4-6", "messages": [ Message ], "tools": [ LlmTool ] }

// LlmTool — a tool as the model sees it.
{ "function": { "name": "getWeather", "description": "…", "parameters": { /* JSON Schema */ } } }

// LlmResponse — what a worker-run model returns (see llm.request).
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

# ── Loop: a pure function of (trigger, transcript, pending) → decision. ──
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

        # The model replied: finish, or run the tools it asked for.
        case "llm.response":
            assistant  = trigger["message"]
            transcript = history + [ assistant ]
            calls      = assistant.get("tool_calls", [])
            if not calls:
                return { "transcript": transcript, "actions": [ done(assistant.get("content")) ] }
            actions = [ call_tool(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in calls ]
            return { "transcript": transcript, "actions": actions }

        # A tool finished: record it; prompt again once the round is complete.
        case "tool.result":
            node       = { "role": "tool", "content": trigger["result"],
                           "tool_call_id": trigger["tool_call_id"], "name": trigger["name"], "id": new_id() }
            transcript = history + [ node ]
            round_open = req["pending"]["tool_calls"] + req["pending"]["sub_agents"] > 0
            if round_open:
                return { "transcript": transcript }                          # just record
            return { "transcript": transcript, "actions": [ call_llm(transcript, schemas) ] }

        # The engine wants YOU to run the tool.
        case "tool.execute":
            try:
                result = tools[ trigger["name"] ].run( trigger["arguments"] )
                return { "actions": [ return_tool_result(trigger["tool_call_id"], result) ] }
            except Exception as e:
                return { "actions": [ return_tool_error(trigger["tool_call_id"], str(e)) ] }

        case _:
            return {}

# Actions are plain data — the "builders" just construct the tagged struct.
def call_llm(messages, tools):        return { "type": "call.llm", "request": { "model": MODEL, "messages": messages, "tools": tools }, "handler": "server" }
def call_tool(id, name, args):        return { "type": "call.tool", "tool_call_id": id, "name": name, "arguments": args, "handler": "worker" }
def return_tool_result(id, result):   return { "type": "return.tool.result", "tool_call_id": id, "result": result, "attempt": 0 }
def return_tool_error(id, error):     return { "type": "return.tool.error", "tool_call_id": id, "error": error, "retryable": False, "attempt": 0 }
def done(data):                       return { "type": "done", "data": data }

# ensure_system(history): if history doesn't start with a system message, prepend
# one carrying your instructions (with a fresh id). Otherwise return it unchanged.
```

That's the entire loop: four cases. `user.message` prompts, `llm.response`
dispatches tools or finishes, `tool.execute` runs a tool, `tool.result` prompts
again once nothing is pending. Everything below is an optional extension on the
same four-case skeleton.

## Extensions

**Sub-agents.** Present each child agent to the model as a tool (an `LlmTool`
whose `parameters` is `{ message: string }`). When the model calls one, instead of
`call.tool`, emit two actions: `spawn.sub_agent` (a fresh child `session_id`, the
child's `agent_id`, and the originating `tool_call_id`) and `send.message` (that
same `session_id`, a user message with the unwrapped `message` argument). The
child's turn runs in its own session; when it finishes, the engine sends the
parent a `tool.result` — the same trigger a tool produces — so the loop above
handles the result with no extra case.

**Worker-run models.** If you call the LLM provider yourself rather than letting
the engine do it, set `handler: "worker"` on your `call.llm`. The engine then
sends you an `llm.request` trigger; run the call and reply with `return.llm.result`
(or `return.llm.error`). This is also where token streaming lives: the request's
transport may hold the connection open so you can push token deltas as they
arrive, then return the final `LlmResponse`. Server-handled models (`handler:
"server"`) stream on the engine side and never produce an `llm.request`.

**Stop conditions.** To cap the loop (e.g. stop after N assistant rounds), check
your condition in the `tool.result` case before prompting again and emit `done`
instead of `call.llm`.

**Client actions & modes.** Handle `client.action` to react to non-message signals
— a mode switch, an approval, a cancel — without them appearing in the transcript.
This is how human-in-the-loop approval and modal (plan → execute) agents work; see
[Patterns](./06-patterns.md).

## State

`worker_state` is a base64-encoded JSON blob the engine round-trips untouched. The
boundary decodes it into the request and re-encodes whatever the decision returns
— so state persists across decisions without the engine ever reading it. If you'd
rather keep state in your own database, leave it empty and load/save around the
loop, keyed by `identity.id` or `session_id`. See [SDK / State](./04-sdk.md#state).
