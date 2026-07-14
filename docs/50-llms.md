---
title: LLMs
group: Building agents
---

The engine runs the agent's LLM calls for you, against the provider you
configure. A worker can run them itself instead: declare `handler: "worker"`
and answer the `llm.execute` trigger with the provider's response.

## Example

An agent whose calls run on the worker, in Anthropic's wire format, served
with Hono.

```javascript title="server.mjs"
import Anthropic from "@anthropic-ai/sdk";
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";

const anthropic = new Anthropic();
const app = new Hono();

app.post("/", async (c) => {
    const { trigger, proposed } = await c.req.json();

    if (trigger.type === "session.start") {
        return c.json({
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true,
                handler: "worker",   // run LLM calls here, not on the engine
                format: "anthropic"  // trigger.request is a Messages API body
            }
        });
    }

    if (trigger.type === "llm.execute") {
        return streamSSE(c, async (sse) => {
            const stream = anthropic.messages.stream(trigger.request);
            for await (const event of stream) {
                await sse.writeSSE({ event: "llm.token.delta", data: JSON.stringify(event) });
            }
            await sse.writeSSE({
                event: "decision.result",
                data: JSON.stringify({
                    actions: [{ type: "llm.result", response: await stream.finalMessage() }]
                })
            });
        });
    }

    return c.json(proposed ?? null);
});

serve({ fetch: app.fetch, port: 4444 });
```

## Where calls run

`handler` on the agent config decides where an LLM call runs.

| `handler` | Runs |
| --- | --- |
| `server` (default) | On the engine, against its provider. |
| `worker` | On your worker, which answers `llm.execute` with `llm.result` or `llm.error`. |

## Wire format

Without `format`, `llm.execute.request` is the engine's neutral `LlmRequest`,
and you answer with a neutral `LlmResponse`. Set `format` and the request is
the provider's own body, ready to send, and you return the provider's own
response.

| `format` | `request` and `response` |
| --- | --- |
| unset | Neutral `LlmRequest` / `LlmResponse`. |
| `"anthropic"` | Anthropic Messages API. |
| `"openai"` | OpenAI Chat Completions. |

`format` requires `handler: "worker"`.

## Streaming

When `trigger.stream` is set, answer with `text/event-stream`: one
`llm.token.delta` per chunk, then a single `decision.result` frame carrying
the `llm.result`. Deltas are the neutral `StreamDelta`, or the provider's own
stream events when `format` is set. The request omits the stream flag;
`trigger.stream` is authoritative.

## The loop

`llm.execute` runs one call. After any call settles, on the worker or the
engine, the engine fires `llm.finished`; its `proposed` records the assistant
message and dispatches the tool calls or ends the turn. Return `proposed`.

## Spec

```typescript
// agent config
handler?: "server" | "worker"      // default server
format?: "openai" | "anthropic"    // requires handler worker
stream?: boolean                   // default false

type LlmExecute = {
    type: "llm.execute"
    id: string
    request: unknown          // neutral LlmRequest, or provider-native when format is set
    format?: "openai" | "anthropic"
    stream: boolean
    attempt: number
    deadline?: string
}

type LlmResult = { type: "llm.result"; id?: string; attempt?: number; response: unknown }
type LlmError = { type: "llm.error"; id?: string; attempt?: number; error: string; retryable?: boolean; code?: ErrorCode; detail?: unknown }
```

Full `LlmRequest`, `LlmResponse`, and `StreamDelta` in
[Protocol](./150-protocol.md).

## Next

- [Tool calls](./30-tools.md): the same trigger, answer, finished loop.
- [Retries and timeouts](./120-retries.md): the `retry` policy and `llm.error`.
- [Protocol](./150-protocol.md): the streaming frames and neutral shapes.
