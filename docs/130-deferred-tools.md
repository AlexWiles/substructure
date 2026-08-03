---
title: Deferred tools
group: Building agents
---

A tool doesn't have to answer in the same decision. Your worker can
acknowledge the call, do the work on its own schedule, and report the result
later. The turn stays open, and the run resumes when the answer arrives.

## Example

Answer `tool.execute` without settling the call: start the work, return an
empty decision, and the engine leaves the call in flight.

```javascript title="server.mjs"
const tools = [
    {
        name: "render_report",
        description: "Render a report. This takes a while.",
        input: {
            type: "object",
            properties: { topic: { type: "string" } },
            required: ["topic"]
        }
    }
];

function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return { agent: { model: "claude-haiku-4-5-20251001", tools } };
    }

    // Kick off the work, keyed by the call id, and leave the call pending.
    if (trigger.type === "tool.execute") {
        startRender(trigger.id, trigger.input.value);   // runs elsewhere, on its own schedule
        return {};                                       // no tool.result yet
    }

    return proposed;
}
```

When the render finishes, settle the call out of band by its `id`:

```jsonc
{ "type": "tool.result", "id": "<toolCallId>", "result": "https://reports/42.pdf" }
```

## Deferring

Answering a `tool.execute` with no `tool.result` or `tool.error` leaves the
call in flight. It is the same state a client-side tool sits in while the
browser works (see [Client-side tools](./90-client-tools.md)); a deferred tool
is the worker-handled version. The declaration is an ordinary tool, run on
your own schedule.

## Waiting

While a call is in flight the turn stays open. Sibling tool calls that finish
first only record their results; the model is not re-prompted until every
in-flight call has settled, so it never sees a half-finished turn. The wait is
just persisted state: it holds no compute and survives a restart of the engine
or your worker.

## Settling

Report the outcome with a `tool.result` or `tool.error`, addressed by the call
`id`:

```typescript
{ type: "tool.result"; id: string; attempt?: number; result?: unknown }
{ type: "tool.error"; id: string; error: string; retryable: boolean; attempt?: number }
```

Your service submits it to the engine's client API. `attempt` is optional;
include it to fence a settle from a stale executor that a retry has
superseded. The engine records the result and, once nothing is in flight,
re-prompts the model and the turn continues.

## Timeouts

A client-handled call waits indefinitely by default — it is the one effect the
engine leaves unbounded, because a human may be answering it. Give the
`tool.call` a `retry` policy with an `attempt_timeout_secs` or a
`total_timeout_secs` to bound the wait: when either lapses the call fails, then
retries or settles as a terminal error per the policy. See
[Retries](./120-retries.md).

## Next

- [Tool calls](./30-tools.md): the tool contract these build on.
- [Client-side tools](./90-client-tools.md): the same wait, driven by the browser.
- [Retries and timeouts](./120-retries.md): bounding a deferred call.
- [Interrupts](./140-interrupts.md): pausing the conversation for a human.
