---
title: Sub-agents
group: Building agents
---

A sub-agent is another agent that the model sees as a tool.

When the model calls it, the engine starts a child session. The child's result
goes back to the parent as the tool's result.

## Example

Two agents are two sections. The parent names the child. The child can use a
cheaper model.

```toml title="subs.toml"
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

Neither agent needs a worker. Routing is per agent, so an engine-hosted parent
can call a worker-hosted child.

One worker can serve both. Route on `agent_id`.

```javascript title="server.mjs"
function assistant({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return { agent: { ...proposed.agent, system: "Delegate poetry to the poet." } };
    }
    return proposed;
}

const poet = ({ proposed }) => proposed;

const decide = (req) => (req.agent_id === "poet" ? poet(req) : assistant(req));
```

## Declare a sub-agent

`sub_agents` lists by ID the agents this agent can call. The model sees each
one as a tool with that ID as its name. The tool takes one `message` argument.

IDs and tool names share one namespace. An ID must not match a tool name.

The tool's description comes from the `description` on the section that it
names. Two
parents that call the same child describe it the same way.

An agent that exists only to be called can carry only a `description` and a
`worker`.

A worker can also write the description. The expanded `sub_agents` arrive in the
`session.start` proposal. So `description` is required only for an agent with no
worker.

## Call a sub-agent

When the model calls a sub-agent, `proposed` starts a child session. It carries
the child's first message, taken from the `message` argument.

The child runs as a normal session with its own `agent_id`, transcript, and
cost. Its decision requests carry an `ancestry` list of the sessions above it.

## Complete a call

When the child's turn ends, the parent receives a `sub_agent.finished` trigger.

`proposed` records the result as the tool's result and prompts the parent again.
The engine adds the child's cost and token use to the parent's turn.

## Spec

```typescript
type SubAgent = { id: string; description?: string }

// the actions that start a child. the engine proposes them for you
{ type: "sub_agent.spawn", session_id: string, agent_id: string, tool_call_id: string }
{ type: "message.send", session_id: string, message: DraftMessage }

// the trigger
{ type: "sub_agent.finished", id: string, ok: boolean, session_id: string, agent_id: string }
```

On `sub_agent.finished`, `id` is the tool call and `session_id` is the child.

## Next

- [Tool calls](./60-tools.md): the child's result comes back as a tool result.
- [Agents](./30-agents.md): the section a child agent declares.
- [Durability](./200-durability.md): the engine saves child sessions.
