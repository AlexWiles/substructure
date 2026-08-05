---
title: LLMs
group: Building agents
---

An agent names an `[llm.<id>]` block, and its model calls run there. The block's
`type` says who makes the call. With `anthropic`, `openai`, or `openrouter`, the
engine calls the vendor. With `worker`, your worker calls the vendor and answers
the `llm.execute` trigger with the response.

The file is the only place this is declared, so nothing on the wire can change
it.

## Example

An agent whose calls run on the worker, in Anthropic's wire format, served with
Hono.

```toml title="substructure.toml"
[llm.byo]
type = "worker"        # the agent's worker makes the call
format = "anthropic"   # trigger.request is a Messages API body

[agent.my-agent]
llm = "byo"
model = "claude-haiku-4-5-20251001"
worker = "http://localhost:4444"
```

```javascript title="server.mjs"
import Anthropic from "@anthropic-ai/sdk";
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";

const anthropic = new Anthropic();
const app = new Hono();

app.post("/", async (c) => {
    const { trigger, proposed } = await c.req.json();

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

    return c.json(proposed);
});

serve({ fetch: app.fetch, port: 4444 });
```

## Where calls run

The block's `type` decides.

| `type` | Runs | Needs a key where the engine runs |
| --- | --- | --- |
| `anthropic` / `openai` / `openrouter` | On the engine, against that vendor. | yes |
| `worker` | On your worker. It answers `llm.execute` with `llm.result` or `llm.error`. | no |

An agent that uses a `worker` block must have a `worker` URL. Without one,
nothing can make the call. The file fails to parse instead of failing on the
first turn.

A worker can send one call to a different block. Name the block on the
`llm.call`. You do not have to change the config:

```javascript
{ type: "llm.call", llm: "cheap" }   // this call only
```

## Wire format

On a `worker` block with no `format`, `llm.execute.request` is the engine's own
`LlmRequest`, and you answer with an `LlmResponse`. Set `format`, and the request
is the provider's own body, ready to send. You then return the provider's own
response.

| `format` | `request` and `response` |
| --- | --- |
| unset | The engine's `LlmRequest` and `LlmResponse`. |
| `"anthropic"` | Anthropic Messages API. |
| `"openai"` | OpenAI Chat Completions. |

`format` applies only to `type = "worker"`. A call the engine makes always uses
the engine's own shapes.

## Streaming

When `trigger.stream` is set, answer with `text/event-stream`. Send one
`llm.token.delta` for each chunk, then one `decision.result` frame that holds the
`llm.result`. Each delta is a `StreamDelta`, or a provider stream event when
`format` is set. The request does not carry the stream flag. Read
`trigger.stream`.

## The loop

`llm.execute` runs one call. When any call ends, on the worker or on the engine,
the engine sends `llm.finished`. Its `proposed` records the assistant message,
then starts the tool calls or ends the turn. Return `proposed`.

## Spec

```toml
# substructure.toml
[llm.<id>]
type = "anthropic" | "openai" | "openrouter" | "worker"
api_key_env = "…"                  # engine-run types; defaults from `type`
base_url = "…"                     # engine-run types
format = "openai" | "anthropic"    # `worker` only
```

```typescript
// agent config
llm?: string                       // the [llm.<id>] to run on
stream?: boolean                   // default false

type LlmExecute = {
    type: "llm.execute"
    id: string
    request: unknown          // LlmRequest, or the provider's own body when format is set
    format?: "openai" | "anthropic"
    stream: boolean
    attempt: number
    deadline?: string
}

type CallLlm = { type: "llm.call"; llm?: string; model?: string; /* … */ }
type LlmResult = { type: "llm.result"; id?: string; attempt?: number; response: unknown }
type LlmError = { type: "llm.error"; id?: string; attempt?: number; error: string; retryable?: boolean; code?: ErrorCode; detail?: unknown }
```

For the full `LlmRequest`, `LlmResponse`, and `StreamDelta`, see
[Protocol](./150-protocol.md).

## Where the key lives

The block's `type` also decides who holds the key.

When you run the engine yourself, with `subs serve` or `subs run`, a block the
engine calls reads its key from the environment. `api_key_env` names the
variable. Without it, the engine uses the vendor's default variable:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`. The file only
names the variable. It never holds the key, so you cannot commit one.

```toml title="substructure.toml"
[llm.claude]
type = "anthropic"
api_key_env = "MY_ANTHROPIC_KEY"   # optional; local only
```

In the [cloud](./170-cloud.md), the deployment holds the key. It cannot read a
variable on your machine. `subs apply` removes `api_key_env`, and a deployment
refuses a document that carries one. Upload the key once:

```sh
subs llm set-key claude    # reads the key from stdin, never argv
subs llm list
```

Calls then run on your key. Until you set one, every call on that block fails
with an error that says so. There is no platform key to fall back to.

A `worker` block needs no key on either side. The call stays in your worker, and
your worker calls the provider with a key from its own environment.

## Next

- [Tool calls](./30-tools.md): the same trigger, answer, and finished loop.
- [Retries and timeouts](./120-retries.md): the `retry` policy and `llm.error`.
- [Protocol](./150-protocol.md): the streaming frames and the engine's shapes.
