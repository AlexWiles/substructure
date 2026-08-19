---
title: Durability
group: Running it
---

Every session is an event log. The engine records each trigger, decision, and
effect before it acts.

If the engine or your worker stops, the run continues from the last saved step.
If you submit the same work twice, it runs once.

## What is saved

A session's events are the source of truth. They hold the messages, the tool and
model calls with their results, the state and config updates, and the start and
end of each turn.

The engine commits each event before it makes any external call.

## Recovery

A call or decision that passes its deadline fails, and the engine retries it
under its policy. A decision has ten attempts by default, so a worker deploy or
a short outage does not fail the run.

When the engine restarts, it fails its own calls that were in flight and sends
them again. It does not wait for their deadlines.

A restarted engine or worker continues from where it stopped. It does not repeat
finished work.

## Idempotency

| You submit | Result |
| --- | --- |
| A `turn_id` that is running or complete | The engine returns that turn. It does not run again. |
| A `decision_id` that is answered | Nothing happens. |
| A result for a call that is not open | The engine refuses it. |

An `attempt` on a result blocks an old executor that a retry replaced.

## Reconnects

A client that lost its connection resumes the event stream from a cursor. It
reads the events it missed, then goes live.

If it sends its conversation view again, the engine matches the messages against
the tree by ID. A reconnect records nothing new.

## Storage

The engine saves to a store that you can replace. The CLI uses SQLite in
`substructure.db`.

Stop the engine, come back later, and the session continues.

## Next

- [Retries](./210-retries.md): deadlines and redelivery.
- [Events](./240-events.md): what the log holds.
- [Interrupts](./100-interrupts.md): a paused branch, saved to disk.
