---
title: Conversations
group: Building agents
---

A session's history is a tree of messages, persisted by the engine. The worker
builds no chat backend. A client edits, regenerates, and branches by
resubmitting its view of the conversation, which the engine reconciles into the
tree.

## The tree

Each message is a node with a parent. The `head` is the active leaf, and the
path from head to root is the conversation the model sees. A decision request
carries that active path as `messages`, and the whole tree as `message_tree`.

## Submitting

A client sends input two ways.

| Input | Effect |
| --- | --- |
| `client.message` | Append one message to the active path. |
| `client.messages` | Submit the full conversation view; edits and branches reconcile against the tree. |

The engine matches a submitted view against the tree by message `id`. Known ids
at the front line up with existing nodes; the first new or id-less message
begins a new branch, and `new_from` on the trigger marks its index.

## Editing and branching

To edit, resubmit the view with an earlier message changed and given a new id
or none. The engine forks a branch at the last matching node, records the
changed message and everything after it, and moves the head to the tip. The
original branch stays in the tree. Regenerating is the same move: resubmit
without the last assistant reply, and a new one is recorded as a sibling.

Resubmitting a message under its existing id changes nothing, since
reconciliation keys on id, not content.

## Example

An active path of `u1 → a1 → u2`. The client edits back to the first question:

```jsonc
{
    "type": "client.messages",
    "agent_id": "bot",
    "messages": [
        { "id": "u1", "role": "user", "content": "weather in NYC?" },
        { "role": "user", "content": "weather in LA instead" }
    ]
}
```

`u1` matches; the id-less message forks a branch under it. The worker sees
`messages: [u1, e1]` with `new_from: 1`, and the `a1 → u2` branch is kept.

## Next

- [Agent state](./60-state.md): state resolves along the active path.
- [AG-UI](./100-ag-ui.md): drives this from a browser chat UI.
- [Protocol](./150-protocol.md): messages, the tree, and client inputs.
