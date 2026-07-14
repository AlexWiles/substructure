---
title: Durability
group: Reliability
---

Every session is an event log. Each trigger, decision, and effect is recorded
before it is acted on, so a crash of the engine or your worker resumes from the
last saved step. Resubmitting the same work does not repeat it.

## What's persisted

The session's events are the source of truth: messages, tool and LLM calls with
their results, state and config updates, and turn boundaries. Each is appended
with a snapshot atomically, under optimistic concurrency, before any external
call runs. Side effects are dispatched only from committed events, so intent is
durable first.

## Recovery

An unanswered decision stays pending and is redelivered; a call past its
deadline fails and retries per its policy. Because every step was committed
first, a restarted engine or worker continues where it stopped rather than
redoing finished work.

## Idempotency

| Resubmit | Result |
| --- | --- |
| A completed or active `turn_id` | Returns the existing turn; it does not re-run. |
| A `decision_id` already answered | No-op. At most one decision is live at a time. |
| A settle for a call that is not pending | Rejected. An `attempt` fences a stale executor. |

## Reconnects

A dropped client resumes its event stream from a cursor, replaying what it
missed before going live. A resent conversation view is deduplicated against the
tree by id, so a reconnect re-records nothing.

## Storage

The engine persists through a pluggable store. The CLI uses SQLite in `data.db`;
stop it, come back later, and the session resumes.

## Next

- [Deferred tools](./130-deferred-tools.md): in-flight work that survives a restart.
- [Interrupts](./140-interrupts.md): a paused session held as durable state.
- [Retries and timeouts](./120-retries.md): deadlines and redelivery.
