---
title: Agent state
group: Building agents
---

Agent state is any JSON. The engine stores it and never reads it.

It travels with every decision. The engine sends it on the request as `state`,
and your worker can write a new value on the response. A worker that holds no
state of its own gets a memory for each session.

## Example

```javascript title="server.mjs"
function decide({ trigger, proposed, state }) {
    if (trigger.type === "session.start") {
        return { agent: { model: "claude-haiku-4-5" }, state: { turns: 0 } };
    }

    if (trigger.type === "client.messages") {
        return { ...proposed, state: { turns: (state?.turns ?? 0) + 1 } };
    }

    return proposed;   // no state key keeps the current state
}
```

## Read and write state

The engine sends the current state on every request. To change it, return a new
`state` on the decision.

| Response `state` | Effect |
| --- | --- |
| a value | Replaces the state. |
| `{}` | Clears the state. |
| omitted or `null` | Keeps the current state. |

Writing the same value again records no new version. A session with no state
reads as `null`, so check for it.

## Branches

The engine attaches state to the message tree. If you edit an earlier message or
branch
the conversation, the engine reads the state as it was at that point.

`DecisionRequest.state` is always correct for the active path. See
[Conversations](./120-conversations.md).

## State and config

Both travel with the decision, and the engine versions them the same way. It
reads them differently.

The agent config is a typed document that the engine reads to propose model
calls.
State is memory the engine stores and returns unchanged.

## Spec

```typescript
// DecisionRequest
state: unknown          // your state, stored exactly as it is

// DecisionResponse
state?: unknown         // omitted or null keeps it. {} clears it
```

## Next

- [Workers](./50-workers.md): where you read and write it.
- [Conversations](./120-conversations.md): the tree state attaches to.
- [Durability](./200-durability.md): the engine saves state and restores it.
