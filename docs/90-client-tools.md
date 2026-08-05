---
title: Client-side tools
group: Building agents
---

A client-side tool runs in the client, usually the browser. It does not run on
your worker. Declare it with `handler: "client"`. When the model calls it, the
engine sends the call to the client, waits for the result, and adds the result
to the conversation.

## Example

The worker declares the tool with `handler: "client"`. The worker runs nothing.

```javascript title="server.mjs"
function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
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

    // Nothing to run here: the client executes get_location.
    return proposed;
}
```

No `tool.execute` reaches the worker for `get_location`. When the model calls it,
the run stops with the call open. The client then ends the call by `id`:

```jsonc
{ "type": "tool.result", "id": "<toolCallId>", "result": "Lisbon" }
```

## The round trip

1. The worker declares the tool with `handler: "client"`.
2. The model calls it. The engine sends the call to the client, not to the
   worker. The run stops with the call open.
3. The client runs the tool and ends the call by `id` with a `tool.result` or a
   `tool.error`.
4. The engine records the result, sends `tool.finished`, and prompts the model
   again, the same way it does for a worker tool. The worker returns `proposed`.

Input and output schemas work the same as for worker tools. See
[Tool calls](./30-tools.md).

## Ending a call

The client answers an open call with one of these inputs. It names the call by
`id`:

```typescript
type ClientInput =
    | { type: "tool.result"; id: string; attempt?: number; result?: unknown }
    | { type: "tool.error"; id: string; error: string; retryable: boolean; attempt?: number }
```

Only the session's owner can end a call, and only a call with the `client`
handler. A tool message inside a `client.messages` submit ends an open client
call the same way. This lets a browser answer from its transcript, with no
separate input.

## Tools from the client

A client can also add tools when it submits. The engine adds them to the config
through the submit's `client` context:

```typescript
type ClientContext = {
    tools?: AgentTool[]   // tools the client runs, added to the config
    // …
}
```

Each of these tools has `handler: "client"`, and the engine sends its calls to
the client. The engine only adds tools. It ignores a name that a worker tool or
a sub-agent already uses, so the worker keeps its own names. Use this for
browser features that the worker does not know about. See
[AG-UI](./100-ag-ui.md).

## Next

- [Tool calls](./30-tools.md): worker tools, schemas, and errors.
- [AG-UI](./100-ag-ui.md): the browser protocol that carries these.
- [Protocol](./150-protocol.md): the full `ClientInput` and `ClientContext`.
