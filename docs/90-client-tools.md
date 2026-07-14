---
title: Client-side tools
group: Building agents
---

A client-side tool runs in the client, usually the browser, not on your
worker. You declare it with `handler: "client"`. When the model calls it, the
engine hands the call to the client, waits for the result, and folds it back
into the conversation.

## Example

The worker declares the tool and marks it `handler: "client"`. It runs nothing
itself.

```javascript title="server.mjs"
function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true,
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

No `tool.execute` reaches the worker for `get_location`. When the model calls
it, the run yields with the call pending, and the client settles it by `id`:

```jsonc
{ "type": "tool.result", "id": "<toolCallId>", "result": "Lisbon" }
```

## The round trip

1. The worker declares the tool `handler: "client"`.
2. The model calls it. The engine dispatches the call to the client, not the
   worker, and the run yields with the call pending.
3. The client runs the tool and settles the call by `id` with a `tool.result`
   or `tool.error`.
4. The engine records the result, fires `tool.finished`, and re-prompts the
   model, exactly as for a worker tool. The worker returns `proposed`.

Input and output schemas apply as they do for worker tools (see
[Tool calls](./30-tools.md)).

## Settling

The client answers a pending call with one of these inputs, addressing it by
the call `id`:

```typescript
type ClientInput =
    | { type: "tool.result"; id: string; attempt?: number; result?: unknown }
    | { type: "tool.error"; id: string; error: string; retryable: boolean; attempt?: number }
```

Only the session's owner may settle, and only a call whose handler is
`client`. A tool message carried in a `client.messages` submit settles a
pending client call the same way, which is how a browser transcript answers
without a separate input.

## Client-contributed tools

A client can also add tools at submit time, layered onto the config through
the submit's `client` context:

```typescript
type ClientContext = {
    tools?: AgentTool[]   // client-executed tools, added to the config
    // …
}
```

Each tool is declared `handler: "client"` and dispatched to the client like
any other. Layering is additive: a name already taken by a worker tool or
sub-agent is ignored, so the worker always wins its own names. Use it for
browser-native capabilities the worker doesn't know up front. See
[AG-UI](./100-ag-ui.md).

## Next

- [Tool calls](./30-tools.md): worker tools, schemas, and errors.
- [AG-UI](./100-ag-ui.md): the browser protocol that carries these.
- [Protocol](./150-protocol.md): the full `ClientInput` and `ClientContext`.
