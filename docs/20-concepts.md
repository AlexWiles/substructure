---
title: Core concepts
group: Getting started
---

The words the rest of the docs use.

## Three roles

- **Engine**: runs the agent loop. It calls LLMs, runs tools, saves every step,
  and streams events. It is the `subs` binary, or the cloud.
- **Worker**: your code. It is optional. It is an HTTP endpoint that receives a
  decision request and returns a decision. One worker can host several agents,
  identified by `agent_id`.
- **Client**: anything that sends messages and reads events. A browser, a
  backend, or the CLI.

You declare an agent in `substructure.toml`. The `worker` key on its section
selects who decides. Set it, and the engine POSTs each decision there. Leave it
off, and the engine accepts its own proposal. One app can hold both kinds.

## Sessions and turns

A **session** is one conversation. It holds the messages, tool calls, state, and
config, and the engine saves all of it. A **turn** starts when a client sends
input. It ends when the agent stops responding.

A session runs one turn at a time. If a submit arrives while a turn runs, the
engine refuses it with `409` and names the turn. To redirect an agent that is
working, [interrupt](./140-interrupts.md) it first, then submit.

Send `queue: true` to wait instead of being refused. The engine holds the
message and starts it as the next turn when the running turn completes. Queued
turns run one at a time, in the order they arrived. The engine composes a
queued message against the conversation as it is when its turn starts, so the
message sees the reply it waited for. Chat transports need this: a mention that
arrives during a turn is a follow-up question, not an error.

Each event names the turn that was running when it happened. So a queued turn's
`decision.queued` event names the *previous* turn, because the engine recorded
it while that turn held the session. Everything else that belongs to the queued
turn falls between its `turn.started` and `turn.completed`.

Three things move on their own. Do not read them as one:

- The turn. It is running or it is not.
- Each call in flight. Each has its own lifecycle and retries.
- Each open [interrupt](./140-interrupts.md). Each pauses one branch of the
  conversation.

An interrupted session still has its turn, and its tools still run. A turn does
not end because a call failed or because someone paused a branch.

## The decision loop

When the engine needs to know what happens next, it builds a decision request.
If the agent has a `worker`, the engine POSTs the request there:

```jsonc
{
    "session_id": "…",
    "agent_id": "my-agent",
    "trigger": { "type": "tool.execute", "name": "get_current_time", … },
    "proposed": { … },   // what the engine plans to do; empty when it has no plan
    "state": { … },      // your agent state
    "agent": { … },      // the current agent config
    "calls": [ … ]       // tool and LLM calls in flight
}
```

Your worker returns a decision:

```jsonc
{
    "messages": [ … ],   // messages to record
    "actions": [ … ],    // what to do next
    "state": { … },      // omit to keep the current state
    "agent": { … }       // omit to keep the current config
}
```

The engine acts only on the decisions your worker returns.

## Triggers

The `trigger` says why the engine is asking.

| Trigger | Meaning |
| --- | --- |
| `session.start` | A new session. Its `proposed` is the config from the file. |
| `client.messages` | The client sent or edited messages. |
| `client.action` | The client called a named action that you defined. |
| `tool.execute` | The model called one of your tools. Run it. |
| `tool.finished` | A tool call ended. |
| `llm.execute` | Your worker makes this LLM call. The agent config selects this. |
| `llm.finished` | An LLM call ended. |
| `sub_agent.finished` | A child session completed. |
| `interrupt.resumed` | Someone resumed a paused branch. |

## Proposals

For most triggers, `proposed` holds the next step of a standard agent loop:
record the reply, start the tool calls, or end the turn. Return it unchanged to
accept it. `proposed` is empty when only your worker knows what to do, such as
when it must run one of your tools.

A worker that always returns `proposed` unchanged is a complete agent. An agent
with no worker runs the same loop, and the engine accepts its own proposals. If
an engine-hosted agent gets an empty proposal, the decision fails with an
error.

## Actions

The `actions` in a decision tell the engine what to do.

| Action | Meaning |
| --- | --- |
| `llm.call` | Make an LLM call. |
| `tool.call` | Start a tool call. |
| `tool.result` / `tool.error` | End a tool call. |
| `llm.result` / `llm.error` | End an LLM call that the worker made. |
| `sub_agent.spawn` | Start a child session. |
| `message.send` | Write a message into a session. |
| `interrupt` | Pause the active branch until someone resumes it. |
| `done` | End the turn. |

## Agent config

The config says what the agent is: `llm`, `model`, `system`, `tools`, and
`sub_agents`. An `[agent.<id>]` section holds this config with the same field
names. Its `sub_agents` key names other sections by id. The engine sends what
the file declares as the `session.start` proposal. Your worker can accept it,
change it, or replace it. It can also rewrite the config later in the
conversation to change the model, add a tool, or change the system prompt.

## Where things run

| Call | Default | Alternative |
| --- | --- | --- |
| Decisions | The engine | Your worker (`worker` on `[agent.<id>]`) |
| Tools | Your worker | The browser ([client-side tools](./90-client-tools.md)) |
| LLM calls | The engine | Your worker (`type = "worker"` on `[llm.<id>]`, [LLMs](./50-llms.md)) |

## Durability

The engine saves every trigger, decision, and call before it acts on it. If the
engine or your worker stops, the run continues from the last saved step. See
[Durability](./110-durability.md).

## Next

- [Tool calls](./30-tools.md): declaring and running tools.
- [Protocol](./150-protocol.md): the full wire reference.
