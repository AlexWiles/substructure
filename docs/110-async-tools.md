---
title: Async tools
group: Building agents
---

A tool does not have to answer in the same decision.

Your worker can accept the call, do the work on its own schedule, and report the
result later. The turn stays open. The run continues when the answer arrives.

## Example

Answer `tool.execute` without ending the call. Start the work and return an
empty decision.

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
        return { agent: { model: "claude-haiku-4-5", tools } };
    }

    // Start the work under the call id and leave the call open.
    if (trigger.type === "tool.execute") {
        startRender(trigger.id, trigger.input.value);
        return {};
    }

    return proposed;
}
```

When the render finishes, end the call by its `id`.

```jsonc
{ "type": "tool.result", "id": "<toolCallId>", "result": { "content": [{ "type": "text", "text": "https://reports/42.pdf" }] } }
```

Send it to the engine's machine API, which takes an API key rather than a
client token. See
[REST API](./250-api.md#post-apimachinesessionssession_idcallssettle).

## Leave a call open

Answer a `tool.execute` with no `tool.result` and no `tool.error` and the call
stays open.

A client-side tool is in the same state while the browser works. An async tool
is the worker's version of that. Declare it as an ordinary tool and run it on
your own schedule.

## While the call is open

The turn stays open while a call is in flight. Other tool calls that finish
first record their results.

The engine does not prompt the model again until every open call has ended. The
model never sees a half-finished turn.

The wait is saved state. It uses no compute, and it survives a restart of the
engine or your worker.

## End the call

Report the result and name the call by `id`.

```typescript
{ type: "tool.result", id: string, attempt?: number, result: ToolResult }
{ type: "tool.error", id: string, error: string, retryable: boolean, attempt?: number }
```

`attempt` is optional. Include it to block a result from an old executor that a
retry replaced.

The engine records the result. When no calls are in flight, it prompts the model
again.

## Set the deadline

An open call is still under the tool's retry policy. A worker tool defaults to a
120-second attempt timeout and a 600-second total timeout, so a call left open
past those bounds fails with `deadline_exceeded`.

Raise the bounds for a tool that takes longer. Set a `retry` override on the
agent, or on the `tool.call` action.

```toml title="subs.toml"
[agent.assistant.retry]
tool = { run_timeout_secs = 86400, total_timeout_secs = 86400 }
```

A client-handled tool is the exception. It has no bounds by default, because a
person might be answering it. See [Retries](./210-retries.md).

## Next steps

- [Tool calls](./60-tools.md): the rules these follow.
- [Client-side tools](./150-client-tools.md): the same wait, run by the browser.
- [Interrupts](./100-interrupts.md): pause the conversation for a person.
