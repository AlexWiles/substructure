---
title: Durability
group: Reliability
---

Every session is an event log. The engine records each trigger, decision, and
effect before it acts on it. If the engine or your worker stops, the run
continues from the last saved step. If you submit the same work twice, it runs
once.

## What is saved

A session's events are the source of truth. They hold the messages, the tool and
LLM calls with their results, the state and config updates, and the start and
end of each turn. The engine writes each event and a snapshot together, in one
atomic write, before it makes any external call. It starts an effect only after
the event is committed, so the intent is saved first.

## Recovery

A call or decision that passes its deadline fails, and the engine retries it
under its policy. A decision has ten attempts by default, so a worker deploy or
a short outage does not fail the run. When the engine restarts, it fails its own
calls that were in flight and sends them again. It does not wait for their
deadlines. Because the engine commits every step first, a restarted engine or
worker continues from where it stopped. It does not repeat finished work.

## Idempotency

| You submit | Result |
| --- | --- |
| A `turn_id` that is running or complete | The engine returns that turn. It does not run again. |
| A `decision_id` that is answered | Nothing happens. Only one decision is live at a time. |
| A result for a call that is not open | The engine refuses it. An `attempt` blocks an old executor. |

## Reconnects

A client that lost its connection resumes the event stream from a cursor. It
reads the events it missed, then goes live. If it sends its conversation view
again, the engine matches the messages against the tree by id, so a reconnect
records nothing new.

## Storage

The engine saves to a store that you can replace. The CLI uses SQLite in
`substructure.db`. Stop the engine, come back later, and the session continues.

## Next

- [Deferred tools](./130-deferred-tools.md): work in flight that survives a restart.
- [Interrupts](./140-interrupts.md): a paused branch, saved to disk.
- [Retries and timeouts](./120-retries.md): deadlines and redelivery.
