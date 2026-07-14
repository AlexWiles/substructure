---
title: Agent state
group: Building agents
---

Agent state is arbitrary JSON the engine stores verbatim and never interprets.
It rides every decision: delivered on the request as `state`, optionally
rewritten on the response. It gives a stateless worker durable per-session
memory.

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

The current state arrives on every `DecisionRequest` as `state`. To change it,
return a new `state` on the decision.

| Response `state` | Effect |
| --- | --- |
| a value | Overwrites. |
| `{}` | Clears to empty. |
| omitted or `null` | Keeps the current state. |

A write equal to the current state records no new version. A session with no
state yet reads back as `null`, so read defensively.

## Branching

State anchors to the message tree. Editing an earlier message or branching a
conversation resolves the state that was current at the fork point, not the
latest linear write, so re-deriving from `DecisionRequest.state` is always
correct for the active path. See [Conversations](./70-conversations.md).

## State versus config

Both ride the decision and version the same way, and the engine reads them
differently. The agent `config` is a typed document it interprets to propose
LLM calls. State is opaque memory it stores and returns untouched.

## Spec

```typescript
// DecisionRequest
state: unknown          // your state, stored verbatim

// DecisionResponse
state?: unknown         // omitted or null keeps current; {} clears
```

Full shapes in [Protocol](./150-protocol.md).

## Next

- [Conversations](./70-conversations.md): the tree state anchors to.
- [Durability](./110-durability.md): state is persisted and resumes.
- [Protocol](./150-protocol.md): the decision request and response.
