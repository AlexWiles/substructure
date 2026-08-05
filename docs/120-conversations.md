---
title: Conversations
group: Building agents
---

A session's history is a tree of messages. The engine saves it. You do not build
a chat backend.

To edit, regenerate, or branch, a client sends its view of the conversation
again. The engine merges that view into the tree.

## The tree

Each message is a node with a parent. The head is the active leaf. The path from
the head to the root is the conversation the model sees.

A decision request holds that path as `messages` and the whole tree as
`message_tree`.

## Submitting

A client sends input in two ways.

| Input | Effect |
| --- | --- |
| `client.message` | Adds one message to the active path. |
| `client.messages` | Sends the full conversation view. The engine merges edits and branches into the tree. |

The engine matches the view against the tree by message `id`. Known ids at the
start match existing nodes. The first message that is new or has no id starts a
branch. `new_from` on the trigger gives its index.

## Editing and branching

To edit, send the view again with an earlier message changed. Give that message
a new id, or no id.

The engine starts a branch at the last matching node, records the changed
message and everything after it, and moves the head to the end. The original
branch stays in the tree.

To regenerate, send the view without the last assistant reply. The engine
records a new reply beside the old one.

Sending a message again under its existing id changes nothing. The engine
matches on id, not on content.

A view that stops at an earlier node writes nothing. It moves the head to that
node and emits `head.moved`. The next reply branches there, and work in flight
on the branch you left is cancelled. This is how a client changes branches.

## Example

An active path of `u1 → a1 → u2`. The client edits back to the first question.

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

`u1` matches. The message with no id starts a branch under it. The worker sees
`messages: [u1, e1]` with `new_from: 1`. The `a1 → u2` branch stays.

## One turn at a time

A session runs one turn at a time. A submit that arrives during a turn is
refused with `409` and names the turn.

Send `queue: true` to wait instead. The engine holds the message and starts it
when the running turn completes. A queued message sees the reply it waited for,
so it reads as a follow-up question. Queued turns run in the order they arrived.

To redirect an agent that is working, [interrupt](./100-interrupts.md) it first,
then submit.

## Next

- [Agent state](./90-state.md): the engine reads state along the active path.
- [AG-UI](./140-ag-ui.md): do this from a browser chat UI.
- [Protocol](./230-protocol.md): messages, the tree, and client inputs.
