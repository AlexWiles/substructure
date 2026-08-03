---
title: Retries and timeouts
group: Reliability
---

Every effect the engine runs — an LLM call, a tool call, a sub-agent spawn, a
connector fetch, a worker decision — carries a retry policy. It bounds the
effect with two timeouts and re-issues it after a backoff when a failure is
retryable, until the attempts run out or the failure is terminal.

## Policy

```typescript
type RetryPolicy = {
    attempt_timeout_secs: number | null  // deadline for one attempt; null waits forever
    total_timeout_secs: number | null    // deadline for the whole effect; null is unbounded
    max_attempts: number                 // cap on total attempts, not on retries
    backoff_base_secs: number            // backoff delay base
    backoff_max_secs: number             // backoff delay ceiling
}
```

## Two timeouts

`attempt_timeout_secs` bounds a single dispatch-to-settle span and is carried on
the `tool.execute` and `llm.execute` triggers as `deadline`. It restarts with
each attempt. An attempt that lapses fails and retries.

`total_timeout_secs` bounds the effect as a whole, measured from its first
dispatch — every attempt, the backoff between them, and any time the effect
spends running. A retry cannot buy more time by restarting the clock.

The second exists because the first cannot express everything. A delegation
whose child turn is in flight has already cleared its attempt: the spawn landed,
and the child may legitimately run far longer than the call that started it.
Only the total bounds it, which is what settles a parent whose child stopped
coming back.

## Defaults

Each kind gets its own, because the kinds are not alike:

| Effect | attempt | total | attempts |
|---|---|---|---|
| LLM call | 180s | 1800s | 5 |
| Tool call — worker | 120s | 600s | **1** |
| Tool call — client | none | none | 1 |
| Tool call — connector | 60s | 300s | 2 |
| Sub-agent spawn | 60s | 3600s | 3 |
| Connector fetch | 30s | 120s | 3 |
| Worker decision | 300s | 1800s | 10 |

A worker tool is bounded but never repeated: the engine cannot vouch for a
tool's idempotency, so a retry is the author's call. A client tool is the one
thing left unbounded — a deferred call waits for a human (see
[Deferred tools](./130-deferred-tools.md)). A sub-agent spawn does retry: the
child session id is derived deterministically and an already-created session
counts as success, so re-issuing one is safe.

## Overrides

Declare policies per kind on the agent config, in `substructure.toml` or in the
`agent` a worker returns on a decision:

```toml
[agent.assistant.retry]
default = { attempt_timeout_secs = 180, total_timeout_secs = 900, max_attempts = 3, backoff_base_secs = 2, backoff_max_secs = 30 }
tool    = { attempt_timeout_secs = 120, total_timeout_secs = 600, max_attempts = 1 }
```

The keys are `default`, `llm`, `tool`, `sub_agent`, and `connector`. `default`
covers every kind that names nothing. A kind that names nothing and has no
`default` falls to the engine's default above.

A single action may still carry its own `retry`, which wins over all of it:

```jsonc
{
    "type": "tool.call",
    "name": "render_report",
    "arguments": { "topic": "q3" },
    "retry": { "attempt_timeout_secs": 30, "total_timeout_secs": 300, "max_attempts": 3, "backoff_base_secs": 2, "backoff_max_secs": 60 }
}
```

Resolution runs most specific first:

```
action retry → agent per-kind → agent default → engine default
```

Worker decisions are the exception: they are not the agent's to declare, since
the decision is the call that produces the config.

## Backoff

The delay before attempt `n` is `min(backoff_base_secs ^ n, backoff_max_secs)`
seconds, for `n` = 1, 2, 3… So `backoff_base_secs: 2` gives 2s, 4s, 8s, capped
at `backoff_max_secs`.

## Terminal and retryable

A `tool.error` or `llm.error` is terminal unless it sets `retryable: true`. A
retryable failure is re-issued while attempts remain. `max_attempts` counts
attempts, so `1` allows a single try and `3` allows up to two retries.

A failure that is not retryable, a policy out of attempts, or a lapsed
`total_timeout_secs` settles as a terminal error. An attempt timeout is
retryable — the next attempt may well land. A total timeout is not: the budget
covers every attempt, so there is nothing left to retry into.

`code` labels a failure but does not decide whether it retries; `retryable`
does. The engine stamps `deadline_exceeded` on either timeout.

```typescript
type ErrorCode = "provider_error" | "rate_limited" | "refused" | "budget_exceeded" | "deadline_exceeded"
```

A worker signals a retryable failure with:

```javascript
return { actions: [{ type: "tool.error", id: trigger.id, error: "upstream 503", retryable: true, code: "provider_error" }] };
```

## Next

- [Tool calls](./30-tools.md): the `tool.error` a retry acts on.
- [Deferred tools](./130-deferred-tools.md): bounding an open-ended wait.
- [Protocol](./150-protocol.md): `RetryPolicy`, `ErrorCode`, and `Call.status`.
