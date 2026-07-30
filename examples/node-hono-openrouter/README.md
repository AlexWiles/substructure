# node-hono-openrouter

Like [`node-hono-openai`](../node-hono-openai), but the worker brings its own
LLM through OpenRouter. The engine never touches an LLM provider — it just
routes decisions. OpenRouter speaks Chat Completions, so the OpenAI SDK works
as-is with `baseURL` pointed at `https://openrouter.ai/api/v1`, and the model
can be any OpenRouter slug (here `anthropic/claude-haiku-4.5`). The same
pattern fits any OpenAI-compatible gateway. The protocol types in
`protocol.ts` are generated from the JSON Schema, and Node runs the `.ts`
directly (>= 24).

The agent config declares `handler: "worker"`, so the engine routes its LLM
calls to this worker instead of running them server-side, and
`format: "openai"`, so the wire speaks the Chat Completions API natively: the
`llm.execute` trigger's `request` is a ready-to-send Chat Completions body,
each raw stream chunk goes back as an `llm.token.delta`, and the final
completion answers the `llm.result` verbatim. No translation code in the
worker.

The engine can also call OpenRouter itself (`subs run --llm-provider openrouter`);
run it on the worker instead when you want to hold the key, pick models per
request, or add headers the engine doesn't know about.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker** (it makes the OpenRouter calls, so it holds the key):

```sh
export OPENROUTER_API_KEY=sk-or-...
npm install
node server.ts
```

**2. Send a message with the CLI** (no `[llm]` section, the worker owns the LLM):

```sh
subs run -c substructure.toml \
    --input '{"type":"client.message","message":{"role":"user","content": "hi"}}'
```

## Regenerate types

`protocol.ts` is generated from `schemas/protocol.schema.json` and committed.
To regenerate after a protocol change:

```sh
npx quicktype --src-lang schema --lang typescript \
    --src ../../schemas/protocol.schema.json \
    --top-level Protocol --just-types --prefer-unions -o protocol.ts
npx @biomejs/biome format --write protocol.ts
```
