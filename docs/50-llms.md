---
title: LLMs
group: Building agents
---

An agent's model calls run on the `[llm.<id>]` block it names, and the block's
`type` is the venue: `anthropic`, `openai`, and `openrouter` mean the engine
calls the vendor itself; `worker` means your worker does, answering the
`llm.execute` trigger with the provider's response.

Declaring it in one place is the point — the file says where a call runs, and
nothing on the wire can disagree.

## Example

An agent whose calls run on the worker, in Anthropic's wire format, served
with Hono.

```toml title="substructure.toml"
[llm.byo]
type = "worker"        # this agent's worker makes the call
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

    // The declared config arrives as the proposal; add streaming to it.
    if (trigger.type === "session.start") {
        return c.json({ agent: { ...proposed.agent, stream: true } });
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

    return c.json(proposed);
});

serve({ fetch: app.fetch, port: 4444 });
```

## Where calls run

The block's `type` decides.

| `type` | Runs | Needs a key where the engine runs |
| --- | --- | --- |
| `anthropic` / `openai` / `openrouter` | On the engine, against that vendor. | yes |
| `worker` | On your worker, which answers `llm.execute` with `llm.result` or `llm.error`. | no |

An agent on a `worker` block must have a `worker` URL — otherwise nothing would
be there to make the call — and the file says so rather than failing at the
first turn.

A worker can move one call to another block by naming it on the `llm.call`
itself, so mixing venues per call needs no config rewrite:

```javascript
{ type: "llm.call", llm: "cheap" }   // this call only
```

## Wire format

On a `worker` block without `format`, `llm.execute.request` is the engine's
neutral `LlmRequest`, and you answer with a neutral `LlmResponse`. Set `format`
and the request is the provider's own body, ready to send, and you return the
provider's own response.

| `format` | `request` and `response` |
| --- | --- |
| unset | Neutral `LlmRequest` / `LlmResponse`. |
| `"anthropic"` | Anthropic Messages API. |
| `"openai"` | OpenAI Chat Completions. |

`format` only applies to `type = "worker"`: an engine-run call is always
neutral.

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
    request: unknown          // neutral LlmRequest, or provider-native when format is set
    format?: "openai" | "anthropic"
    stream: boolean
    attempt: number
    deadline?: string
}

type CallLlm = { type: "llm.call"; llm?: string; model?: string; /* … */ }
type LlmResult = { type: "llm.result"; id?: string; attempt?: number; response: unknown }
type LlmError = { type: "llm.error"; id?: string; attempt?: number; error: string; retryable?: boolean; code?: ErrorCode; detail?: unknown }
```

Full `LlmRequest`, `LlmResponse`, and `StreamDelta` in
[Protocol](./150-protocol.md).

## Where the key lives

The block's `type` also decides who holds the credential.

Running locally (`subs serve`, `subs run`), an engine-run block reads its key
from the environment. `api_key_env` names the variable; absent, the vendor's own
default (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`) is used.
The key is *named*, never written — a committed file must not be able to hold
one.

```toml title="substructure.toml"
[llm.claude]
type = "anthropic"
api_key_env = "MY_ANTHROPIC_KEY"   # optional; local only
```

In the [cloud](./170-cloud.md), the deployment holds the key instead — it cannot
read a variable on your machine, so `subs apply` strips `api_key_env` and a
deployment that receives one rejects the document. Upload the key once:

```sh
subs llm set-key claude    # reads the key from stdin, never argv
subs llm list
```

Calls then run on your key. Until one is set, a call on that block fails saying
so; there is no platform key to silently fall back to.

A `worker` block needs no key on either side: the call never leaves your worker,
which reaches the provider with a key from its own environment.

## Next

- [Tool calls](./30-tools.md): the same trigger, answer, finished loop.
- [Retries and timeouts](./120-retries.md): the `retry` policy and `llm.error`.
- [Protocol](./150-protocol.md): the streaming frames and neutral shapes.
