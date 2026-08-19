---
title: Retries and timeouts
group: Running it
---

Every effect the engine runs has a retry policy. An effect is a model call, a
tool call, a sub-agent start, a connector fetch, or a worker decision.

When an effect fails and can be retried, the engine waits, then sends it again.
It stops when the attempts run out or when a failure cannot be retried.

## Defaults

| Effect | attempt | total | attempts |
|---|---|---|---|
| Model call | 180s | 1800s | 5 |
| Tool call (worker) | 120s | 600s | **1** |
| Tool call (client) | none | none | 1 |
| Tool call (connector) | 60s | 300s | 2 |
| Sub-agent start | 60s | 3600s | 3 |
| Connector fetch | 30s | 120s | 3 |
| Worker decision | 300s | 1800s | 10 |

A worker tool has timeouts, and the engine never repeats it. The engine cannot
know whether your tool is safe to run twice. You decide when to retry.

A client tool has no limit, because an async call can wait for a person. See
[Async tools](./110-async-tools.md).

The engine does retry a sub-agent start. A second attempt cannot create a second
child.

## Two timeouts

```typescript
type RetryPolicy = {
    attempt_timeout_secs: number | null  // one attempt. null waits forever
    total_timeout_secs: number | null    // the whole effect. null has no limit
    max_attempts: number                 // attempts, not retries
    backoff_base_secs: number
    backoff_max_secs: number
}
```

`attempt_timeout_secs` limits one attempt. The engine sends it on the
`tool.execute` and `llm.execute` triggers as `deadline`. It restarts with each
attempt.

`total_timeout_secs` limits the whole effect, from the first attempt. It covers
every attempt and every wait between them. A retry does not restart this clock.

Some work outlives its attempt. A sub-agent start finishes as soon as the child
session exists, and the child can then run for much longer. Only the total
timeout ends a parent whose child stopped answering.

## Overrides

Set an override per kind on the agent config. Write it in `substructure.toml`,
or in the `agent` your worker returns.

```toml
[agent.assistant.retry]
tool = { max_attempts = 3 }
```

That worker tool now has three attempts. It keeps the 120s attempt timeout and
the 600s total. An override names only the fields it changes.

The keys are `default`, `llm`, `tool`, `sub_agent`, and `connector`. They stack.
`default` sets the base, and each kind changes it.

```toml
[agent.assistant.retry]
default = { max_attempts = 3, backoff_max_secs = 30 }
tool    = { max_attempts = 1 }
connector = { attempt_timeout_secs = 10 }
```

One action can carry its own `retry`. The engine applies it last.

```jsonc
{
    "type": "tool.call",
    "name": "render_report",
    "arguments": { "topic": "q3" },
    "retry": { "attempt_timeout_secs": 30, "max_attempts": 3 }
}
```

The layers, widest first:

```
engine default → agent default → agent per-kind → action retry
```

An override cannot remove a timeout. To wait almost forever, set a large number.

An override cannot change a worker decision. That call produces the config, so
the policy cannot come from it.

## Backoff

The wait before attempt `n` is `min(backoff_base_secs ^ n, backoff_max_secs)`
seconds. So `backoff_base_secs: 2` gives 2s, 4s, and 8s, up to
`backoff_max_secs`.

## Which failures retry

The engine does not retry a `tool.error` or an `llm.error` unless it sets
`retryable: true`.

```javascript
return { actions: [{ type: "tool.error", error: "upstream 503", retryable: true, code: "provider_error" }] };
```

`max_attempts` counts attempts. `1` allows one try. `3` allows two retries.

The call ends with a final error in three cases.

- A failure that is not retryable.
- A policy with no attempts left.
- An expired `total_timeout_secs`.

An attempt timeout is retryable. The next attempt might work. A total timeout is
not.

`code` describes a failure. `retryable` decides whether the engine tries again.
The engine sets `deadline_exceeded` on both kinds of timeout.

```typescript
type ErrorCode = "provider_error" | "rate_limited" | "refused" | "budget_exceeded" | "deadline_exceeded"
```

## Next

- [Tool calls](./60-tools.md): the `tool.error` a retry acts on.
- [Async tools](./110-async-tools.md): put a limit on a long wait.
- [Durability](./200-durability.md): why a retry does not repeat finished work.
