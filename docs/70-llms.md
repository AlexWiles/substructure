---
title: LLMs
group: Building agents
---

An agent names an `[llm.<id>]` block. Its model calls run there.

The block's `type` says who makes the call.

| `type` | Runs on | Needs a key where the engine runs |
| --- | --- | --- |
| `anthropic` | The engine, against Anthropic. | yes |
| `openai` | The engine, against OpenAI. | yes |
| `openrouter` | The engine, against OpenRouter. | yes |
| `worker` | Your worker. | no |

The file is the only place that declares this. Nothing on the wire can change
it.

## Bring your own key

The engine calls the vendor for you.

```toml title="subs.toml"
[llm.claude]
type = "anthropic"

[agent.support]
llm = "claude"
model = "claude-sonnet-4-5"
```

There is no default block and no fallback. An agent names a block, or its calls
fail with an error that lists the blocks that the file declares.

## Call the model yourself

Set `type = "worker"` and the engine holds no key. Your worker answers an
`llm.execute` trigger.

```toml title="subs.toml"
[llm.byo]
type = "worker"        # the agent's worker makes the call
format = "anthropic"   # trigger.request is a Messages API body

[agent.my-agent]
llm = "byo"
model = "claude-haiku-4-5"
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

An agent that uses a `worker` block must have a `worker` URL. The file fails to
parse without one.

## Wire format

`format` sets the shape of the `llm.execute` your worker answers. It applies
only to `type = "worker"`.

| `format` | `request` and `response` |
| --- | --- |
| unset | The engine's `LlmRequest` and `LlmResponse`. |
| `"anthropic"` | Anthropic Messages API. |
| `"openai"` | OpenAI Chat Completions. |

Set it and the request is the provider's own body, ready to send. You return the
provider's own response.

A call that the engine makes always uses the engine's own shapes.

## Streaming

When `trigger.stream` is set, answer with `text/event-stream`. Send one
`llm.token.delta` per chunk, then one `decision.result` frame holding the
`llm.result`.

Each delta is a `StreamDelta`, or a provider stream event when `format` is set.

The request does not carry the stream flag. Read `trigger.stream`.

## One call on another block

Name a block on the `llm.call` action. The config stays as it is.

```javascript
{ type: "llm.call", llm: "cheap" }   // this call only
```

## The loop

`llm.execute` runs one call.

When any call ends, on the worker or on the engine, the engine sends
`llm.finished`. Its `proposed` records the assistant message, then starts the
tool calls or ends the turn. Return `proposed`.

## Where the key lives

The block's `type` decides who holds the key.

**On an engine you run**, a block that the engine calls reads its key from the
environment. `api_key_env` names the variable. Without it, the engine uses the
vendor's default: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or
`OPENROUTER_API_KEY`.

```toml title="subs.toml"
[llm.claude]
type = "anthropic"
api_key_env = "MY_ANTHROPIC_KEY"   # local only
```

**In the cloud**, the deployment holds the key. Upload it once.

```sh
subs auth llm.claude    # reads the key from stdin
subs list
```

Calls then run on your key. Until you set one, every call on that block fails
with an error that says so. There is no platform key to fall back to.

`subs apply` removes `api_key_env`. A deployment refuses a document that carries
one.

**A `worker` block needs no key on either side.** Your worker calls the provider
with a key from its own environment.

## Spec

```toml
[llm.<id>]
type = "anthropic" | "openai" | "openrouter" | "worker"
api_key_env = "…"                  # engine-run types. local only
base_url = "…"                     # engine-run types
format = "openai" | "anthropic"    # `worker` only
cache_ttl = "5m" | "1h"            # `anthropic`, `openrouter`
cache_ttl = "in_memory" | "24h"    # `openai`
```

## Prompt caching

Every engine-run call asks the vendor to cache the prompt: the tools, the system
prompt, and the transcript as it grows. A turn that hits the cache pays a
fraction of the input price, and the engine keeps the front of the prompt
unchanged so the cache keeps hitting.

A cached prefix lives about five minutes by default, and each hit resets the
clock. A session that turns faster than that never needs `cache_ttl`. Set it
when a session waits on a person or a slow job between turns, and the vendor
would otherwise read the prompt again in full: the longer life costs more to
write, and pays for itself from about the third turn that reads it.

```typescript
type LlmExecute = {
    type: "llm.execute"
    id: string
    request: unknown          // LlmRequest, or the provider's body when format is set
    format?: "openai" | "anthropic"
    stream: boolean
    attempt: number
    deadline?: string
}

type LlmResult = { type: "llm.result"; id?: string; attempt?: number; response: unknown }
type LlmError = { type: "llm.error"; id?: string; attempt?: number; error: string; retryable?: boolean }
```

For `LlmRequest`, `LlmResponse`, and `StreamDelta`, see
[Protocol](./230-protocol.md).

## Next

- [Tool calls](./60-tools.md): the same trigger and answer loop.
- [Retries](./210-retries.md): the `retry` policy and `llm.error`.
- [Cloud](./170-cloud.md): uploading a provider key.
