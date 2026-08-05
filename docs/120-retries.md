---
title: Retries and timeouts
group: Reliability
---

Every effect the engine runs has a retry policy. An effect is an LLM call, a tool
call, a sub-agent start, a connector fetch, or a worker decision. The policy sets
two timeouts. When an effect fails and can be retried, the engine waits, then
sends it again. It stops when the attempts run out or when a failure cannot be
retried.

## Policy

```typescript
type RetryPolicy = {
    attempt_timeout_secs: number | null  // limit for one attempt; null waits forever
    total_timeout_secs: number | null    // limit for the whole effect; null has no limit
    max_attempts: number                 // limit on total attempts, not on retries
    backoff_base_secs: number            // base of the wait between attempts
    backoff_max_secs: number             // longest wait between attempts
}
```

## Two timeouts

`attempt_timeout_secs` limits one attempt, from start to end. The engine sends it
on the `tool.execute` and `llm.execute` triggers as `deadline`. It restarts with
each attempt. An attempt that passes it fails, and the engine retries.

`total_timeout_secs` limits the whole effect, measured from the first attempt. It
covers every attempt, the waits between them, and all the time the effect runs. A
retry does not restart this clock.

The second timeout exists because the first cannot cover everything. Take a
sub-agent whose child turn is running. Its attempt is already complete: the
child started, and the child can run much longer than the call that started it.
Only the total timeout limits it. That is what ends a parent whose child stopped
answering.

## Defaults

Each kind of effect has its own defaults:

| Effect | attempt | total | attempts |
|---|---|---|---|
| LLM call | 180s | 1800s | 5 |
| Tool call — worker | 120s | 600s | **1** |
| Tool call — client | none | none | 1 |
| Tool call — connector | 60s | 300s | 2 |
| Sub-agent start | 60s | 3600s | 3 |
| Connector fetch | 30s | 120s | 3 |
| Worker decision | 300s | 1800s | 10 |

A worker tool has timeouts, but the engine never repeats it. The engine cannot
know whether your tool is safe to run twice, so you decide when to retry. A
client tool is the one effect with no limit, because a deferred call can wait
for a person. See [Deferred tools](./130-deferred-tools.md). The engine does
retry a sub-agent start. The child session id is derived from the call, and a
session that already exists counts as success, so a second attempt is safe.

## Overrides

Set an override for each kind on the agent config. Write it in
`substructure.toml`, or in the `agent` your worker returns on a decision. An
override names only the fields it changes. Every field it leaves out keeps the
default above:

```toml
[agent.assistant.retry]
tool = { max_attempts = 3 }
```

That worker tool now has three attempts. It keeps the 120s attempt timeout and
the 600s total. One field changes one field. An override never removes the
timeouts it does not name.

```typescript
type RetryOverride = {
    attempt_timeout_secs?: number
    total_timeout_secs?: number
    max_attempts?: number
    backoff_base_secs?: number
    backoff_max_secs?: number
}
```

The keys are `default`, `llm`, `tool`, `sub_agent`, and `connector`. They stack
instead of replacing each other. `default` sets the base, and each kind changes
it:

```toml
[agent.assistant.retry]
default = { max_attempts = 3, backoff_max_secs = 30 }
tool    = { max_attempts = 1 }          # only tools are never repeated
connector = { attempt_timeout_secs = 10 }
```

One action can carry its own `retry`. The engine applies it last:

```jsonc
{
    "type": "tool.call",
    "name": "render_report",
    "arguments": { "topic": "q3" },
    "retry": { "attempt_timeout_secs": 30, "max_attempts": 3 }
}
```

So the layers are, widest first:

```
engine default → agent default → agent per-kind → action retry
```

An override cannot do two things. It cannot remove a timeout. To wait almost
forever, set a large number. And it cannot change a worker decision, because
that call produces the config, so the policy cannot come from it.

## Backoff

The wait before attempt `n` is `min(backoff_base_secs ^ n, backoff_max_secs)`
seconds, for `n` = 1, 2, 3, and so on. So `backoff_base_secs: 2` gives 2s, 4s,
and 8s, up to `backoff_max_secs`.

## Which failures retry

The engine does not retry a `tool.error` or an `llm.error` unless it sets
`retryable: true`. It sends a retryable failure again while attempts remain.
`max_attempts` counts attempts, so `1` allows one try and `3` allows two
retries.

The call ends with a final error in three cases: a failure that is not
retryable, a policy with no attempts left, or an expired `total_timeout_secs`.
An attempt timeout is retryable, because the next attempt may work. A total
timeout is not, because it covers every attempt and there is no time left.

`code` describes a failure. It does not decide whether the engine retries.
`retryable` decides that. The engine sets `deadline_exceeded` on both kinds of
timeout.

```typescript
type ErrorCode = "provider_error" | "rate_limited" | "refused" | "budget_exceeded" | "deadline_exceeded"
```

A worker reports a retryable failure like this:

```javascript
return { actions: [{ type: "tool.error", id: trigger.id, error: "upstream 503", retryable: true, code: "provider_error" }] };
```

## Next

- [Tool calls](./30-tools.md): the `tool.error` that a retry acts on.
- [Deferred tools](./130-deferred-tools.md): put a limit on a long wait.
- [Protocol](./150-protocol.md): `RetryPolicy`, `ErrorCode`, and `Call.status`.
