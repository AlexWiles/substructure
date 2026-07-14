---
title: Sub-agents
group: Building agents
---

A sub-agent is a delegation target the model sees as a tool. Calling it spawns
a full child session; the child's result folds back into the parent like a
tool result.

## Example

The parent declares a sub-agent. Both agents live on one worker, told apart by
`agent_id`.

```javascript title="server.mjs"
function assistant({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true,
                sub_agents: [{ id: "poet", description: "Writes a haiku on any topic." }]
            }
        };
    }
    return proposed;
}

function poet({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true,
                system: "You are a poet. Respond with a single haiku."
            }
        };
    }
    return proposed;
}

// One worker, two agents told apart by agent_id.
const decide = (req) => (req.agent_id === "poet" ? poet(req) : assistant(req));
```

## Declaring

`sub_agents` lists the agents this one can delegate to. Each appears to the
model as a tool named by its `id`, taking a single `message` argument. That id
shares the model's tool namespace, so it must not collide with a tool name.

## Delegating

When the model calls a sub-agent, `proposed` spawns a child session and sends
its opening message, the `message` argument. The child runs as an ordinary
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
{ type: "sub_agent.spawn"; session_id: string; agent_id: string; tool_call_id: string; retry?: RetryPolicy }
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
