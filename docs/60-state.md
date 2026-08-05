---
title: Agent state
group: Building agents
---

Agent state is any JSON. The engine stores it exactly as it is and never reads
it. It travels with every decision: the engine sends it on the request as
`state`, and your worker can write a new value on the response. This gives a
worker that holds no state of its own a memory for each session.

## Example

```javascript title="server.mjs"
function decide({ trigger, proposed, state }) {
    if (trigger.type === "session.start") {
        return { agent: { model: "claude-haiku-4-5-20251001" }, state: { turns: 0 } };
    }

    if (trigger.type === "client.messages") {
        return { ...proposed, state: { turns: (state?.turns ?? 0) + 1 } };
    }

    return proposed;   // no state key keeps the current state
}
```

## Reading and writing

The engine sends the current state on every `DecisionRequest` as `state`. To
change it, return a new `state` on the decision.

| Response `state` | Effect |
| --- | --- |
| a value | Replaces the state. |
| `{}` | Clears the state. |
| omitted or `null` | Keeps the current state. |

If you write the same value again, the engine records no new version. A session
that has no state yet reads as `null`, so check for it.

## Branching

State is attached to the message tree. If you edit an earlier message or branch
the conversation, the engine reads the state as it was at that point, not the
last value written. So `DecisionRequest.state` is always correct for the active
path. See [Conversations](./70-conversations.md).

## State and config

Both travel with the decision and are versioned the same way. The engine reads
them differently. The agent `config` is a typed document that the engine reads
to propose LLM calls. State is memory that the engine stores and returns
unchanged.

## Spec

```typescript
// DecisionRequest
state: unknown          // your state, stored exactly as it is

// DecisionResponse
state?: unknown         // omitted or null keeps the current state; {} clears it
```

For the full types, see [Protocol](./150-protocol.md).

## Next

- [Conversations](./70-conversations.md): the tree that state attaches to.
- [Durability](./110-durability.md): the engine saves state and restores it.
- [Protocol](./150-protocol.md): the decision request and response.
