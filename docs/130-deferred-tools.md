---
title: Deferred tools
group: Building agents
---

A tool does not have to answer in the same decision. Your worker can accept the
call, do the work on its own schedule, and report the result later. The turn
stays open, and the run continues when the answer arrives.

## Example

Answer `tool.execute` without ending the call. Start the work and return an
empty decision. The engine leaves the call open.

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

    // Start the work under the call id, and leave the call open.
    if (trigger.type === "tool.execute") {
        startRender(trigger.id, trigger.input.value);   // runs elsewhere, on its own schedule
        return {};                                       // no tool.result yet
    }

    return proposed;
}
```

When the render finishes, end the call separately, by its `id`:

```jsonc
{ "type": "tool.result", "id": "<toolCallId>", "result": "https://reports/42.pdf" }
```

## Deferring

If you answer a `tool.execute` with no `tool.result` and no `tool.error`, the
call stays open. This is the same state a client-side tool is in while the
browser works. See [Client-side tools](./90-client-tools.md). A deferred tool is
the worker's version of that. You declare it as an ordinary tool and run it on
your own schedule.

## Waiting

The turn stays open while a call is in flight. Other tool calls that finish
first record their results. The engine does not prompt the model again until
every open call has ended, so the model never sees a half-finished turn. The
wait is saved state. It uses no compute, and it survives a restart of the engine
or your worker.

## Ending the call

Report the result with a `tool.result` or a `tool.error`, and name the call by
`id`:

```typescript
{ type: "tool.result"; id: string; attempt?: number; result?: unknown }
{ type: "tool.error"; id: string; error: string; retryable: boolean; attempt?: number }
```

Your service sends this to the engine's client API. `attempt` is optional.
Include it to block a result from an old executor that a retry replaced. The
engine records the result. When no calls are in flight, it prompts the model
again and the turn continues.

## Timeouts

By default a client-handled call waits forever. It is the one effect with no
limit, because a person may be answering it. To limit the wait, give the
`tool.call` a `retry` policy with an `attempt_timeout_secs` or a
`total_timeout_secs`. When either one expires, the call fails. The engine then
retries it or ends it with a final error, under the policy. See
[Retries](./120-retries.md).

## Next

- [Tool calls](./30-tools.md): the tool rules these follow.
- [Client-side tools](./90-client-tools.md): the same wait, run by the browser.
- [Retries and timeouts](./120-retries.md): put a limit on a deferred call.
- [Interrupts](./140-interrupts.md): pause the conversation for a person.
