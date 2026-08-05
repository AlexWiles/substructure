---
title: Sub-agents
group: Building agents
---

A sub-agent is another agent that the model sees as a tool. When the model calls
it, the engine starts a child session. The child's result goes back to the
parent as the tool's result.

## Example

Two agents are two sections. The parent names the sub-agent. The child can use a
different model, often a cheaper one:

```toml title="substructure.toml"
[llm.claude]
type = "anthropic"

[llm.cheap]
type = "anthropic"

[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
sub_agents = ["poet"]

[agent.poet]
description = "Writes a haiku on any topic."
llm = "cheap"
model = "claude-haiku-4-5"
system = "You are a poet. Respond with a single haiku."
```

Neither agent needs a worker. If one does, add `worker` to that section only.
Routing is per agent, and a child session carries the sub-agent's own
`agent_id`. So an engine-hosted parent can call a worker-hosted child, or the
reverse:

```javascript title="server.mjs"
// One worker, two agents, identified by agent_id.
function assistant({ trigger, proposed }) {
    // The config from the file arrives as the proposal. Change what you need.
    if (trigger.type === "session.start") {
        return { agent: { ...proposed.agent, system: "Delegate poetry to the poet." } };
    }
    return proposed;
}

function poet({ proposed }) {
    return proposed;
}

const decide = (req) => (req.agent_id === "poet" ? poet(req) : assistant(req));
```

## Declaring

`sub_agents` lists by id the agents this agent can call. The model sees each one
as a tool with that id as its name. The tool takes one `message` argument. The
ids and the tool names share one namespace, so an id must not match a tool name.

The tool's description comes from the `description` on the section it names. You
write it once, on the agent it describes, so two parents that call the same
child describe it the same way. An agent that exists only to be called can carry
just a `description` and a `worker`.

A worker can also write or replace the description. The expanded `sub_agents`
arrive in the `session.start` proposal, and a worker that returns its own `agent`
sets its own descriptions. So `description` is required only when nothing else
can supply it: an agent with no `worker`, where the file is the whole
declaration.

## Calling

When the model calls a sub-agent, `proposed` starts a child session. It also
carries the child's first message, taken from the `message` argument, so the
message cannot arrive before its session exists. The child runs as a normal
session with its own `agent_id`, transcript, and cost. Its decision requests
carry an `ancestry` list of the parent sessions above it.

## Completing

When the child's turn ends, the parent receives a `sub_agent.finished` trigger.
`proposed` records the result as the tool's result and prompts the parent again,
the same way `tool.finished` does. The child's cost and token use are added to
the parent's turn.

## Spec

```typescript
type SubAgent = { id: string; description?: string }

// the actions that start a child; the engine proposes them for you
{ type: "sub_agent.spawn"; session_id: string; agent_id: string; tool_call_id: string; retry?: RetryOverride }
{ type: "message.send"; session_id: string; message: DraftMessage }

// trigger
{ type: "sub_agent.finished"; id: string; ok: boolean; session_id: string; agent_id: string; result?: string; error?: string }
```

On `sub_agent.finished`, `id` is the tool call and `session_id` is the child. For
the full types, see [Protocol](./150-protocol.md).

## Next

- [Tool calls](./30-tools.md): the child's result comes back as a tool result.
- [Durability](./110-durability.md): the engine saves child sessions and restores them.
- [Protocol](./150-protocol.md): the spawn actions and the finish trigger.
