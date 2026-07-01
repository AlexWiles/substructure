# Worker protocol — flat transcript, one writer

> **Status:** design spec, not yet implemented. Replaces the on-disk
> *worker-appends-nodes* design.

## The idea

One rule above all: **the tree is mutated in exactly one place — when the engine
reconciles a worker's returned transcript, during a worker decision.** Client
input doesn't write the tree. Effect completion doesn't write the tree. Only a
committed worker decision does.

That gives two tiny contracts:

- **Worker:** `(trigger, transcript) → (transcript, actions)`. It sees a flat
  `Message[]` (the active path) plus what just happened, and returns the transcript
  as it should now be, plus what to do next. A pure function over a flat list — no
  parents, ids, forks, or paths.
- **Engine:** forward triggers, run-and-log effects, and **reconcile the returned
  transcript into the tree on every decision** — diff it against the current path,
  append the new tail, fork where it diverged, set the active path.

The worker owns the transcript; the engine owns the tree.

## Two layers: effect log vs. tree

Keeping these apart is what makes the one-writer rule exact.

- **Effect log** — the engine records a completion event the moment an
  LLM / tool / sub-agent finishes. This is what live streaming reads and what
  survives a crash. It is *not* conversation.
- **Tree** — the durable conversation, all branches. Mutated only by committing a
  worker's returned transcript.

So a tool result exists in the effect log (and streams to the UI) the instant it
lands, but becomes a tree node only when the worker folds it into the transcript it
returns. Brief window, already true today, covered by streaming.

## The worker contract

Each decision the engine sends a **trigger**, the current **transcript**, and
`pending` — counts of in-flight effects (`tool_calls`, `sub_agents`, `llm_calls`)
the worker branches on instead of tracking rounds:

| Trigger | Means |
| --- | --- |
| `user.message` | a client sent a chat message |
| `user.transcript` | a client sent a full transcript (AG-UI view, an edit, or a regenerate) |
| `client.action` | a client sent a typed action (e.g. a mode switch) |
| `llm.response` | an LLM call finished; carries the assistant message |
| `tool.result` | a tool / sub-agent call finished; carries the result (each fires as it lands) |
| `interrupt.resumed` | a paused session was resumed; carries `{ status, payload }` |
| `stall` | nothing has progressed for a while |

The worker returns the **transcript** (flat, as it should now be) and **actions**:

| Action | Effect |
| --- | --- |
| `call.llm { prompt, spec }` | run an LLM call; result returns as `llm.response` |
| `call.tool { id, name, args, executor? }` | run a tool; result returns as `tool.result` |
| `spawn { id, agent, input }` | run a child session; result returns as `tool.result` |
| `interrupt { payload }` | pause for external input |
| `done { data }` | finish the turn |

A minimal agent is a switch over the trigger:

```
decide(trigger, transcript):
  user.message(m):        transcript += m
  user.transcript(view):  transcript  = ensure_system(view)   # adopt client view; add system root on cold start
  llm.response(a):        transcript += a
                          return a.tool_calls
                            ? { transcript, do: a.tool_calls.map(call.tool) }
                            : { transcript, do: done(a.content) }
  tool.result(r):         transcript += r
                          return no_results_pending(pending) ? { transcript, do: call.llm(transcript) }
                                                             : { transcript }

  # the message triggers then:
  return { transcript, do: call.llm(transcript) }
```

No ids, no parents, no fork logic. `+=` is the whole "append."

## `call.llm` is a disposable prompt

```
call.llm { prompt, spec }
```

`prompt` is what the model sees *this call* — disposable. Default it to the
transcript, or shape it freely (compact, inject context, reorder); the engine never
looks inside it. When the call finishes the engine hands the assistant back as
`llm.response`, and the worker folds it into the transcript it returns.

This is the line that does a lot of work:

- **Change the transcript → a record change** (append, branch, ratify an edit).
- **Change only the prompt → ephemeral** (compaction, RAG injection). The record is
  untouched, so the UI never collapses and clients don't re-sync.

A side-call (summarize, classify, route) is just a `call.llm` whose prompt isn't the
conversation and whose response the worker doesn't append; flag it `display:false`
so it doesn't stream.

## `call.tool` and the round

```
call.tool { id, name, args, executor }   // executor = worker | client
```

- **worker** → the engine hands the worker a transcript-free `tool.execute`
  decision; it runs the function and returns the result. **client** → the engine
  surfaces the call to the browser, which returns the result. Either way it arrives
  as `tool.result`. Identical downstream.

All of a decision's tool calls form one **round**: the engine schedules them
concurrently and fires a `tool.result` decision **as each one lands**, so the
worker folds it into the transcript and the tree fills incrementally (a fast
parallel call isn't held behind a slow one). Each decision carries `pending`
(in-flight effect counts), so the worker prompts exactly once — when no
tool/sub-agent result is still pending. The engine derives that state; the worker
never walks the transcript to ask "are all the tools done?"

A sub-agent is `spawn` whose executor is a child session; its result arrives as
`tool.result` like any other. The child runs its own flat transcript in isolation;
the parent only ever sees the returned value.

> Background (fire-and-forget) tools and sub-agents — where the turn doesn't wait
> and the result lands on a later turn — are intentionally out of scope for now.
> They'd add a `scheduling` knob and an out-of-band wake; deferred.

## Reconcile & branching (engine-side)

The worker returns a flat transcript; the engine reconciles it. Identity is by
**id**: the engine stamps every committed message and hands those ids back in the
transcript; the worker carries them through and adds new messages with none. The
diff is then mechanical — known prefix, then the first unknown or divergent message
starts a new tail. If that point already has a child it's a **fork**: the old branch
is preserved and the active path moves to the new one.

The three branch shapes are just different divergence points:

- **Edit** — the client's transcript diverges at the edited message (which carries a
  new id) → fork there.
- **Regenerate** — the client's transcript is truncated at M and the worker calls the
  LLM from it → a fresh sibling under M.
- **Plan → exec** — the worker returns a brand-new transcript (`[system', seed]`)
  built from its state → diverges at the root → a fresh branch; the planning
  conversation is preserved off-path.

Crucially, **client input goes through the worker.** AG-UI forwards the client's
transcript as a `user.transcript` trigger; the worker returns the transcript it
wants (usually the client's, possibly validated or with its system root ensured);
the engine commits/forks during that decision. So edits and regenerates aren't a
separate engine path — they're the same "worker returns a transcript, engine
reconciles," and the worker is the gate for what a client may change.

## Interrupts

An interrupt is a control action, never part of the transcript. The worker returns
`{ transcript, do: interrupt{payload} }`; the engine pauses, fences in-flight
effects, queues later decisions, and surfaces it to the client as a run outcome
(not a message). The client resumes on a separate channel (in AG-UI, the top-level
`resume[]` field); the engine wakes the worker with `interrupt.resumed
{ status, payload }` and the current transcript, and the worker continues. Resume
requires privilege ≥ the interrupt's origin (`System > Machine > Frontend`). Nothing
about it touches the conversation.

## The cases, in one pass

Two families.

**Emit an action, fold the result back** — flat throughout:

- **Client tools** — `call.tool{executor: client}`; surfaced to the browser; result
  returns as `tool.result`. A worker tool differs only in execution (a
  transcript-free `tool.execute` decision).
- **Parallel tools** — N `call.tool`; the engine runs them concurrently and fires a
  `tool.result` as each lands (incremental commits); the worker prompts when
  `pending` shows no result left in flight.
- **Sub-agents** — `spawn`; runs as an isolated child session; result arrives as
  `tool.result`.
- **Interrupts** — the `interrupt` action above.

**Engine forks a divergent transcript** — the worker mostly receives the new active
path and prompts:

- **Edit / regenerate** — client transcript forwarded as `user.transcript`; the
  worker returns it; the engine forks at the divergence.
- **Plan → exec** — the worker returns a fresh transcript; the engine forks at the
  root.

## A complete worker in another language (Python, no SDK)

The engine calls the worker over HTTP: one POST per decision. The request is
`{ trigger, transcript, pending }`; the reply is `{ transcript?, actions }` — omit
`transcript` when the decision doesn't change the record (e.g. running a tool). A
worker that needs memory beyond the transcript gets/returns an opaque `state` blob
too; a basic agent needs none, because the transcript carries the whole
conversation.

Here is a full tool-calling agent — no SDK, no state, ~50 lines:

```python
# A complete Substructure worker. The engine POSTs one decision; we reply.
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

MODEL  = "anthropic/claude-sonnet-4-6"
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
TOOLS  = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city.",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

def get_weather(args): return f"18°C and clear in {args['city']}."
WORKER_TOOLS = {"get_weather": get_weather}

def with_system(msgs):
    return msgs if msgs and msgs[0]["role"] == "system" else [SYSTEM, *msgs]

def call_llm(transcript):
    return {"type": "call.llm", "prompt": transcript,
            "spec": {"model": MODEL, "tools": TOOLS}}

def decide(trigger, transcript, pending):
    kind = trigger["type"]

    if kind == "user.message":              # a new chat message
        transcript = with_system(transcript) + [trigger["message"]]
        return {"transcript": transcript, "actions": [call_llm(transcript)]}

    if kind == "user.transcript":           # full client view (AG-UI, edit, regenerate)
        sys = transcript[0] if transcript and transcript[0]["role"] == "system" else SYSTEM
        transcript = [sys, *trigger["messages"]]   # reuse our committed system (carries its id)
        return {"transcript": transcript, "actions": [call_llm(transcript)]}

    if kind == "llm.response":              # the model answered
        msg = trigger["message"]
        transcript = transcript + [msg]
        calls = msg.get("tool_calls") or []
        if calls:
            actions = [{"type": "call.tool", "executor": "worker", "id": c["id"],
                        "name": c["function"]["name"],
                        "args": json.loads(c["function"]["arguments"])} for c in calls]
            return {"transcript": transcript, "actions": actions}
        return {"transcript": transcript,
                "actions": [{"type": "done", "data": msg["content"]}]}

    if kind == "tool.execute":              # run a worker tool (no transcript change)
        result = WORKER_TOOLS[trigger["name"]](trigger["args"])
        return {"actions": [{"type": "return.tool.result",
                             "id": trigger["id"], "result": result}]}

    if kind == "tool.result":               # one result landed — fold it in
        node = {"role": "tool", "tool_call_id": trigger["tool_call_id"],
                "name": trigger["name"], "content": trigger["result"]}
        transcript = transcript + [node]
        waiting = pending["tool_calls"] + pending["sub_agents"]   # results still in flight
        actions = [call_llm(transcript)] if waiting == 0 else []
        return {"transcript": transcript, "actions": actions}

    return {"actions": []}                  # stall, etc.

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        req  = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        pending = req.get("pending", {"tool_calls": 0, "sub_agents": 0, "llm_calls": 0})
        body = json.dumps(decide(req["trigger"], req.get("transcript", []), pending)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

HTTPServer(("", 8787), H).serve_forever()
```

That's the entire harness. It never touches a parent, an id, a fork, or a path; it
holds no state; the transcript carries the conversation and every tree mechanic is
on the engine side. The same code, with `display:false` on a side `call.llm`, does
ephemeral compaction; returning a fresh `[system', seed]` does a modal switch;
emitting an `interrupt` action pauses. If this is as hard as a worker gets in a new
language, the protocol is doing its job.

Building it surfaced one subtlety worth stating as a rule: **ids express
continuity.** The worker carries through the ids the engine assigned to messages it
continues, and only omits or changes an id where it means to fork. AG-UI clients
drop the system message, so the worker reuses its committed system root
(`transcript[0]`, which carries its id) instead of minting a fresh one — otherwise
the id-less system reads as a new root and every turn forks. A modal switch does the
opposite on purpose: it returns a *different* system with no matching id, and the
fork is what you want. That's the entire reconcile contract from the worker's side,
and it's one line either way.

## What changes from the on-disk version

The worker still authors every message — that's right. What changes:

- The worker's interface is a **flat transcript**, not tree nodes with
  parents/ids. All tree mechanics move to the engine.
- **One mutation path:** the engine reconciles the worker's returned transcript on
  every decision; nothing else writes the tree.
- **Client input is forwarded to the worker** (`user.transcript`) instead of
  reconciled by the engine directly — so edits/regenerates pass through the worker.
- `call.llm` is an explicit **disposable prompt**, making compaction and side-calls
  first-class.
- `call.tool` gains an **executor** (worker | client), unifying client tools and
  sub-agents under one path.
- The engine owns **round-gating**: it fires a `tool.result` per completion and
  surfaces `pending` effect counts on every decision, so the harness folds each
  result in and prompts when nothing's left — never tracking the round itself.

## Costs & open questions

- **Full-transcript round-trip.** The engine hands the worker the whole active path
  and gets it back each decision — trivial for short chats, heavier on long ones.
  Fine to start; optimizable later with deltas, no contract change.
- **Switch back to a prior branch.** Forward plan→exec needs nothing; returning to an
  earlier branch needs an opaque branch handle the worker carries in state
  (`switch_branch{id}`), not tree work.
- **Edit ids.** Confirm the AG-UI client mints a new id on edit (the one rule
  reconcile depends on).
