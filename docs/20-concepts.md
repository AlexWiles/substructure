---
title: Core concepts
group: Getting started
---

The vocabulary the rest of the docs assume.

## Three roles

- **Engine**: runs the agent loop. Calls LLMs, dispatches tools, persists
  every step, streams events. The `subs` binary, or the cloud.
- **Worker**: your code, and optional. An HTTP endpoint that receives a
  decision request and returns a decision. Hosts one or more agents, told
  apart by `agent_id`.
- **Client**: anything that sends messages and streams events back. A
  browser, a backend, the CLI.

An agent is declared in `substructure.toml`, and `worker` on its section is the
whole switch: set it and decisions POST there; leave it off and the engine
decides by accepting its own proposal. Both kinds live in one app.

## Sessions and turns

A **session** is one conversation: its messages, tool calls, state, and
config, persisted by the engine. A **turn** starts when a client submits
input and ends when the agent is done responding.

A session runs one turn at a time. A submit that arrives while a turn is
running is refused with `409`, naming the turn that holds the session — to
redirect a working agent, [interrupt](./140-interrupts.md) it first, then
submit.

A submitter that would rather wait than be refused sends `queue: true`. The
message is taken and held, and becomes the next turn the moment the running
one completes — one turn at a time, in arrival order. Its content is composed
against the conversation as it stands when its turn starts, so a queued
message sees the reply it waited for. This is what a chat transport wants: a
mention that lands mid-turn is a follow-up question, not an error.

Events carry the turn that was running when they happened. A queued turn's
`decision.queued` is therefore stamped with the *previous* turn — it was
recorded while that one held the session — and everything belonging to the
queued turn itself falls between its `turn.started` and `turn.completed`.

Three things move independently, and reading them as one will mislead you:
the turn (running, or not), each call in flight (its own lifecycle and
retries), and any open [interrupt](./140-interrupts.md) (which parks a branch
of the conversation). An interrupted session still has its turn, and its tools
keep running; a turn does not end because a call failed, or because someone
paused a branch.

## The decision loop

Whenever the engine needs to know what happens next, it derives a decision
request. For an agent with a `worker`, it POSTs it there:

```jsonc
{
    "session_id": "…",
    "agent_id": "my-agent",
    "trigger": { "type": "tool.execute", "name": "get_current_time", … },
    "proposed": { … },   // the engine's default continuation; empty when it has none
    "state": { … },      // your agent state
    "agent": { … },      // the current agent config
    "calls": [ … ]       // in-flight tool/LLM calls
}
```

Your worker returns a decision:

```jsonc
{
    "messages": [ … ],   // messages to record
    "actions": [ … ],    // what to do next
    "state": { … },      // omit to keep current state
    "agent": { … }       // omit to keep current config
}
```

The engine only acts on decisions your worker returns.

## Triggers

The `trigger` says why the engine is asking.

| Trigger | Meaning |
| --- | --- |
| `session.start` | A new session. Its `proposed` is the config the file declares. |
| `client.messages` | The client sent or edited messages. |
| `client.action` | The client invoked a named action you define. |
| `tool.execute` | The model called one of your tools. Run it. |
| `tool.finished` | A tool call settled. |
| `llm.execute` | Your worker makes this LLM call (per agent config). |
| `llm.finished` | An LLM call settled. |
| `sub_agent.finished` | A child session completed. |
| `interrupt.resumed` | A paused branch was resumed. |

## Proposals

For most triggers, `proposed` carries the decision an ordinary agent loop
would make next: record the reply, dispatch the tool calls, finish the
turn. Return it as-is to accept. `proposed` is empty when only your worker
knows what to do, like running one of your own tools.

A worker that returns `proposed` unchanged is a complete agent — and an agent
with no worker at all is that same loop, with the engine accepting its own
proposals. An empty proposal on an engine-hosted agent fails the decision
loudly rather than settling it as a silent no-op.

## Actions

A decision's `actions` say what the engine should do.

| Action | Meaning |
| --- | --- |
| `llm.call` | Run an LLM call. |
| `tool.call` | Dispatch a tool call. |
| `tool.result` / `tool.error` | Settle a tool call. |
| `llm.result` / `llm.error` | Settle a worker-executed LLM call. |
| `sub_agent.spawn` | Start a child session. |
| `message.send` | Write a message into a session. |
| `interrupt` | Pause the active branch until someone resumes it. |
| `done` | End the turn. |

## Agent config

The config declares what the agent is: `llm`, `model`, `system`, `tools`,
`sub_agents`. An `[agent.<id>]` section *is* this config — same field names,
with `sub_agents` naming other sections by id — so what the file declares
arrives as the `session.start` proposal. A worker
takes it from there: echo it, amend it, or replace it, and rewrite it again
mid-conversation to swap the model, add a tool, change the system prompt.

## Where things run

| Call | Default | Alternative |
| --- | --- | --- |
| Decisions | The engine | Your worker (`worker` on `[agent.<id>]`) |
| Tools | Your worker | The browser ([client-side tools](./90-client-tools.md)) |
| LLM calls | The engine | Your worker (`type = "worker"` on `[llm.<id>]`, [LLMs](./50-llms.md)) |

## Durability

Every trigger, decision, and call is persisted before it's acted on. If the
engine or your worker dies, the run resumes from the last recorded step. See
[Durability](./110-durability.md).

## Next

- [Tool calls](./30-tools.md): declaring and running tools.
- [Protocol](./150-protocol.md): the full wire reference.
