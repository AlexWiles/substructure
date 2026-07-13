---
title: Core concepts
---

The vocabulary the rest of the docs assume.

## Three roles

- **Engine**: runs the agent loop. Calls LLMs, dispatches tools, persists
  every step, streams events. The `subs` binary, or the cloud.
- **Worker**: your code. An HTTP endpoint that receives a decision request
  and returns a decision. Hosts one or more agents, told apart by `agent_id`.
- **Client**: anything that sends messages and streams events back. A
  browser, a backend, the CLI.

## Sessions and turns

A **session** is one conversation: its messages, tool calls, state, and
config, persisted by the engine. A **turn** starts when a client submits
input and ends when the agent is done responding.

## The decision loop

Whenever the engine needs to know what happens next, it POSTs a decision
request to your worker:

```jsonc
{
    "session_id": "…",
    "agent_id": "my-agent",
    "trigger": { "type": "tool.execute", "name": "get_current_time", … },
    "proposed": { … },   // the engine's default continuation, or null
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
| `session.start` | A new session. Usually answered with the agent config. |
| `client.messages` | The client sent or edited messages. |
| `client.action` | The client invoked a named action you define. |
| `tool.execute` | The model called one of your tools. Run it. |
| `tool.finished` | A tool call settled. |
| `llm.execute` | Your worker makes this LLM call (per agent config). |
| `llm.finished` | An LLM call settled. |
| `sub_agent.finished` | A child session completed. |
| `interrupt.resumed` | A paused session was resumed. |

## Proposals

For most triggers, `proposed` carries the decision an ordinary agent loop
would make next: record the reply, dispatch the tool calls, finish the
turn. Return it as-is to accept. `proposed` is `null` when only your worker
knows what to do, like running one of your own tools.

A worker that returns `proposed` unchanged is a complete agent.

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
| `interrupt` | Pause the session until someone resumes it. |
| `done` | End the turn. |

## Agent config

The config declares what the agent is: `model`, `system`, `tools`,
`sub_agents`, and where LLM calls run. It's usually written once at
`session.start`, but a decision can rewrite it mid-conversation: swap the
model, add a tool, change the system prompt.

## Where things run

| Call | Default | Alternative |
| --- | --- | --- |
| Tools | Your worker | The browser ([client-side tools](./90-client-tools.md)) |
| LLM calls | The engine | Your worker ([LLMs](./50-llms.md)) |

## Durability

Every trigger, decision, and call is persisted before it's acted on. If the
engine or your worker dies, the run resumes from the last recorded step. See
[Durability](./110-durability.md).

## Next

- [Tool calls](./30-tools.md): declaring and running tools.
- [Protocol](./150-protocol.md): the full wire reference.
