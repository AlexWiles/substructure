---
title: Retries and timeouts
group: Reliability
---

A tool call, an LLM call, or a sub-agent spawn each carries a retry policy. It
bounds the call with a timeout and re-issues it after a backoff when a failure
is retryable, until the attempts run out or the failure is terminal.

## Policy

```typescript
type RetryPolicy = {
    timeout_secs: number | null   // deadline for one attempt; null waits forever
    max_retries: number           // cap on total attempts
    backoff_base_secs: number     // backoff delay base
    backoff_max_secs: number      // backoff delay ceiling
}
```

Attach a policy to a `tool.call`, `llm.call`, or `sub_agent.spawn` action, or
set `retry` on the agent config for its LLM calls. The default is no retry and
no timeout. Engine-proposed tool and sub-agent calls take that default, so a
model-driven tool call retries only if your worker adds a policy.

## Timeouts

`timeout_secs` sets a deadline for each attempt, carried on the `tool.execute`
and `llm.execute` triggers as `deadline`. When it lapses the call fails and
retries. A call with no timeout waits indefinitely, which is how a deferred or
client-side tool stays open (see [Deferred tools](./130-deferred-tools.md)).

## Backoff

The delay before attempt `n` is `min(backoff_base_secs ^ n, backoff_max_secs)`
seconds, for `n` = 1, 2, 3… So `backoff_base_secs: 2` gives 2s, 4s, 8s, capped
at `backoff_max_secs`.

## Terminal and retryable

A `tool.error` or `llm.error` is terminal unless it sets `retryable: true`. A
retryable failure is re-issued while attempts remain; `max_retries` caps total
attempts, so `1` allows a single try and `3` allows up to two retries. A
failure that is not retryable, or a policy out of attempts, settles as a
terminal error. A timeout is always retryable.

`code` labels a failure but does not decide whether it retries; `retryable`
does. The engine stamps `deadline_exceeded` on a timeout.

```typescript
type ErrorCode = "provider_error" | "rate_limited" | "refused" | "budget_exceeded" | "deadline_exceeded"
```

## Example

```jsonc
{
    "type": "tool.call",
    "name": "render_report",
    "arguments": { "topic": "q3" },
    "retry": { "timeout_secs": 30, "max_retries": 3, "backoff_base_secs": 2, "backoff_max_secs": 60 }
}
```

It runs, waits 2s and retries on a retryable failure, waits 4s and retries
again, then settles terminal after the third attempt fails. A worker signals a
retryable failure with:

```javascript
return { actions: [{ type: "tool.error", id: trigger.id, error: "upstream 503", retryable: true, code: "provider_error" }] };
```

## Next

- [Tool calls](./30-tools.md): the `tool.error` a retry acts on.
- [Deferred tools](./130-deferred-tools.md): bounding an open-ended wait.
- [Protocol](./150-protocol.md): `RetryPolicy`, `ErrorCode`, and `Call.status`.
