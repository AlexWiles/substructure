---
title: Client-side tools
group: Frontends
---

A client-side tool runs in the client, usually the browser.

Declare it with `handler: "client"`. When the model calls it, the engine sends
the call to the client, waits for the result, and adds the result to the
conversation.

## Example

The worker declares the tool and runs nothing.

```javascript title="server.mjs"
function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                model: "claude-haiku-4-5",
                system: "For location questions, call get_location instead of guessing.",
                tools: [
                    {
                        name: "get_location",
                        description: "Get the user's current city. Only the client can answer this.",
                        handler: "client"
                    }
                ]
            }
        };
    }

    return proposed;
}
```

No `tool.execute` reaches the worker for `get_location`. When the model calls
it, the run stops with the call open. The client ends the call by `id`.

```jsonc
{ "type": "tool.result", "id": "<toolCallId>", "result": "Lisbon" }
```

## The round trip

1. The worker declares the tool with `handler: "client"`.
2. The model calls it. The engine sends the call to the client. The run stops
   with the call open.
3. The client runs the tool and ends the call by `id`.
4. The engine records the result, sends `tool.finished`, and prompts the model
   again. The worker returns `proposed`.

Input and output schemas work the same as for worker tools. See
[Tool calls](./60-tools.md).

## Ending a call

The client answers an open call with one of these inputs.

```typescript
type ClientInput =
    | { type: "tool.result", id: string, attempt?: number, result?: unknown }
    | { type: "tool.error", id: string, error: string, retryable: boolean, attempt?: number }
```

Only the session's owner can end a call, and only a call with the `client`
handler.

A tool message inside a `client.messages` submit ends an open client call the
same way. A browser can answer from its transcript, with no separate input.

## Tools from the client

A client can add tools when it submits. The engine adds them to the config
through the submit's `client` context.

```typescript
type ClientContext = {
    tools?: AgentTool[]   // tools the client runs
    // …
}
```

Each of these tools gets `handler: "client"`.

The engine only adds tools. It ignores a name a worker tool or a sub-agent
already uses. Use this for browser features the worker does not know about. See
[AG-UI](./140-ag-ui.md).

## Next

- [Tool calls](./60-tools.md): worker tools, schemas, and errors.
- [AG-UI](./140-ag-ui.md): the browser protocol that carries these.
- [Async tools](./110-async-tools.md): the same wait, run by your worker.
