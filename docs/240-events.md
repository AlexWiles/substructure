---
title: Events
group: Reference
---

A session is an event log. The engine appends an event before it acts, and
streams every event to any client that is watching.

Use events to render a conversation, to follow a turn, and to monitor a run.

## The stream

```
GET /api/client/sessions/{session_id}/events/stream
```

The response is `text/event-stream`. Each frame carries three fields.

| Frame field | Holds |
| --- | --- |
| `id` | The event's `seq`. |
| `event` | The event type. |
| `data` | An `Event` object. |

```
id: 7
event: message.new
data: {"session_id":"0193a1","seq":7,"occurred_at":"2026-07-14T10:00:03Z","payload":{"type":"message.new","message":{"id":"m2","role":"assistant","content":"Hi"}}}
```

To resume after a drop, pass `?after_seq=<n>`. `EventSource` does this for you.
On reconnect the browser sends the last frame ID in a `Last-Event-ID` header,
which wins over the query parameter.

Each stream covers one session. A sub-agent is its own session with its own
stream. Open a second stream with the child's ID to watch it.

## Event shape

```typescript
type Event = {
    session_id: string
    seq: number
    occurred_at: string         // RFC 3339
    turn_id?: string            // the turn running when this happened
    payload: EventPayload       // tagged by `type`
}
```

`seq` orders the log. It starts at 1 and never repeats.

## Conversation

Render a chat from these.

| Event | Payload |
| --- | --- |
| `session.created` | The session exists. |
| `message.new` | A message joined the tree: `{ message, parent_id? }`. |
| `head.moved` | The active leaf changed. A client branched or edited. |
| `turn.started` | A turn began: `{ turn_id }`. |
| `turn.completed` | A turn ended, with its cost and output. |
| `session.done` | The session finished its work. |
| `session.cancelled` | The session was canceled. |

## Model calls

| Event | Meaning |
| --- | --- |
| `llm.call.requested` | A call was recorded. |
| `llm.call.dispatched` | The call went out. |
| `llm.call.completed` | The call returned. |
| `llm.call.errored` | The call failed. |

Partial output arrives between events as `llm.token.delta` frames.

```
event: llm.token.delta
data: {"type":"llm.token.delta","session_id":"0193a1","call_id":"llm-1","seq":3,"text":"Hi"}
```

These carry a [`StreamDelta`](./230-protocol.md#streaming). The engine does not
record them in the log.

## Tool calls

| Event | Meaning |
| --- | --- |
| `tool.call.requested` | The model asked for a tool: `{ id, name }`. |
| `tool.call.dispatched` | The call went to its handler. |
| `tool.call.completed` | The call returned: `{ id, result }`. |
| `tool.call.errored` | The call failed: `{ id, error }`. |

## Sub-agents

| Event | Meaning |
| --- | --- |
| `sub_agent.requested` | The model called a child agent. |
| `sub_agent.dispatched` | The spawn went out. |
| `sub_agent.started` | The child session exists. |
| `sub_agent.turn_completed` | The child's turn ended. |
| `sub_agent.errored` | The child failed. |

The `id` on each of these is the child session. Open a stream with it to watch
the child.

## Connectors

| Event | Meaning |
| --- | --- |
| `connector.sync.requested` | The engine is fetching a connection's tools. |
| `connector.sync.completed` | The tools arrived. |
| `connector.sync.errored` | The fetch failed. The turn runs without those tools. |

## Interrupts

| Event | Payload |
| --- | --- |
| `session.interrupted` | `{ interrupt_id, origin, reason, payload, anchor? }`. `anchor` is the head it pauses. Without it, every branch pauses. |
| `session.interrupt_resumed` | `{ interrupt_id, payload }`. |

## State and config

| Event | Meaning |
| --- | --- |
| `worker.state.updated` | A decision wrote new state. |
| `agent.updated` | A decision wrote a new config. |
| `channels.updated` | A decision wrote a frontend view. |

## Decisions

Engine bookkeeping. Useful for monitoring.

| Event | Meaning |
| --- | --- |
| `decision.queued` | A decision is waiting for the slot. |
| `decision.dispatched` | The request went to the worker. |
| `decision.completed` | The worker answered. |
| `decision.errored` | The decision failed. |
| `decision.dropped` | The decision was abandoned. |
| `call.voided` | A call was canceled before it settled. |
| `session.message_requested` | A `message.send` targeted another session. |

## AG-UI

The AG-UI endpoints send AG-UI protocol events instead of these. See
[AG-UI](./140-ag-ui.md).

## Next

- [REST API](./250-api.md): the endpoints that serve these streams.
- [Durability](./200-durability.md): why the log is the source of truth.
- [Conversations](./120-conversations.md): the tree that `message.new` builds.
