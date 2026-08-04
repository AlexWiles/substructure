---
title: Sub-agents
group: Building agents
---

A sub-agent is a delegation target the model sees as a tool. Calling it spawns
a full child session; the child's result folds back into the parent like a
tool result.

## Example

A team is two sections. The parent declares the sub-agent; the child can run on
a different — often cheaper — model:

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

Neither needs a worker. When one does, add `worker` to that section alone —
routing is per `(tenant, agent)`, and a child session carries the sub-agent's
own `agent_id`, so an engine-hosted parent can delegate to a worker-hosted
specialist, or the reverse:

```javascript title="server.mjs"
// One worker, two agents told apart by agent_id.
function assistant({ trigger, proposed }) {
    // The declared config arrives as the proposal; amend what you need.
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

`sub_agents` lists the agents this one can delegate to, by id. Each appears to
the model as a tool named by that id, taking a single `message` argument. The
id shares the model's tool namespace, so it must not collide with a tool name.

The tool's description is the `description` on the section it names — declared
once, on the agent it describes, so two parents delegating to one specialist
cannot describe it differently. An agent whose whole job is to be delegated to
can carry nothing but a `description` and a `worker`.

A worker may supply or replace it instead: the expanded `sub_agents` arrive in
the `session.start` proposal, and a worker that returns its own `agent` writes
whatever descriptions it likes. So `description` is required only where nothing
later could fill it in — an agent with no `worker`, whose file is the whole
account of it.

## Delegating

When the model calls a sub-agent, `proposed` spawns a child session. The spawn
carries the child's opening message, the `message` argument, so the message
cannot arrive before the session it opens. The child runs as an ordinary
session with its own `agent_id`, transcript, and cost. Its decision requests
carry an `ancestry` list of the parent sessions above it.

## Completing

When the child's turn finishes, the parent receives a `sub_agent.finished`
trigger. `proposed` records the result as the delegating tool's result and
re-prompts the parent, exactly as a `tool.finished` does. The child's cost and
token usage roll up into the parent's turn.

## Spec

```typescript
type SubAgent = { id: string; description?: string }

// actions that open a child, proposed for you
{ type: "sub_agent.spawn"; session_id: string; agent_id: string; tool_call_id: string; retry?: RetryOverride }
{ type: "message.send"; session_id: string; message: DraftMessage }

// trigger
{ type: "sub_agent.finished"; id: string; ok: boolean; session_id: string; agent_id: string; result?: string; error?: string }
```

On `sub_agent.finished`, `id` is the delegating tool call and `session_id` is
the child. Full types in [Protocol](./150-protocol.md).

## Next

- [Tool calls](./30-tools.md): delegation folds back like a tool result.
- [Durability](./110-durability.md): child sessions are persisted and resume.
- [Protocol](./150-protocol.md): the spawn actions and finish trigger.
