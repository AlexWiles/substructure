---
title: How it works
group: Getting started
---

This page defines the terms the rest of the docs use.

## Engine, worker, and client

| Part | What it is |
| --- | --- |
| Engine | Runs the agent loop. It calls the model, runs tools, saves every step, and streams events. It is the hosted cloud, or the `subs` binary. |
| Worker | Your code. It is an HTTP endpoint that answers a decision request. It is optional. |
| Client | Anything that sends messages and reads events. Slack, a browser, your backend, or the CLI. |

## Agents

You declare an agent in `subs.toml`.

```toml
[agent.oncall]
llm = "openrouter"
model = "anthropic/claude-sonnet-4-5"
system = "You are the on-call assistant."
```

The section sets the agent's model, its prompt, its tools, and the other agents
it can call. See [Agents](./30-agents.md).

The `worker` key sets who decides. Set it and the engine sends every decision
for that agent to your code. Leave it off and the engine decides. One project
can hold both kinds.

## Sessions and turns

A session is one conversation. It holds the messages, the tool calls, the state,
and the config. The engine saves all of it.

A turn starts when a client sends input. It ends when the agent stops
responding. A session runs one turn at a time.

## Decisions

At every step of a turn the engine asks what happens next. That question is a
decision request.

```jsonc
{
    "trigger": { "type": "tool.execute", "name": "get_time" },  // why the engine is asking
    "proposed": { },   // what the engine plans to do
    "state": { },      // your agent state
    "agent": { },      // the current config
    "messages": [ ]    // the conversation so far
}
```

The answer is a decision response.

```jsonc
{
    "actions": [ ],    // what to do next
    "messages": [ ],   // messages to record
    "state": { },      // omit to keep the current state
    "agent": { }       // omit to keep the current config
}
```

For most triggers the engine already has a plan, and `proposed` holds it. Return
`proposed` unchanged to accept it. That is a complete agent.

An agent with no `worker` runs the same loop. The engine proposes, then accepts
its own proposal.

See [Workers](./50-workers.md) to write the code, and
[Protocol](./230-protocol.md) for every field.

## Triggers

The trigger says why the engine is asking.

| Trigger | Meaning |
| --- | --- |
| `session.start` | A session was created. |
| `client.messages` | A client sent or edited messages. |
| `client.action` | A client called a named action. |
| `tool.execute` | The model called one of your tools. Run it. |
| `tool.finished` | A tool call ended. |
| `llm.execute` | Your worker makes this model call. |
| `llm.finished` | A model call ended. |
| `sub_agent.finished` | A child session completed. |
| `interrupt.resumed` | Someone resumed a paused branch. |
| `turn.finished` | A turn completed. It carries the turn's cost and output. |

## Actions

The actions in a decision tell the engine what to do.

| Action | Meaning |
| --- | --- |
| `llm.call` | Make a model call. |
| `tool.call` | Start a tool call. |
| `tool.result` / `tool.error` | End a tool call. |
| `llm.result` / `llm.error` | End a model call the worker made. |
| `sub_agent.spawn` | Start a child session. |
| `message.send` | Write a message into a session. |
| `interrupt` | Pause the branch until someone resumes it. |
| `interrupt.resolve` | Clear an open interrupt and resume. |
| `connector.sync` | Fetch a connection's tools again. |
| `done` | End the turn. |

## Where each call runs

| Call | Default | Alternative |
| --- | --- | --- |
| Decisions | The engine | Your worker. Set `worker` on the agent. |
| Tools | Your worker | The browser. See [Client-side tools](./150-client-tools.md). |
| Model calls | The engine | Your worker. Set `type = "worker"` on the LLM block. See [LLMs](./70-llms.md). |

## Durability

The engine saves every trigger, decision, and call before it acts. If the engine
or your worker stops, the run continues from the last saved step. See
[Durability](./200-durability.md).

## Next steps

- [Agents](./30-agents.md): what you can declare.
- [Workers](./50-workers.md): decide with your own code.
- [Protocol](./230-protocol.md): every wire type.
