# node-hono-openrouter

Like [`node-hono-openai`](../node-hono-openai), but the worker brings its own
LLM through OpenRouter. The engine never touches an LLM provider — it just
routes decisions. OpenRouter speaks Chat Completions, so the OpenAI SDK works
as-is with `baseURL` pointed at `https://openrouter.ai/api/v1`, and the model
can be any OpenRouter slug (here `anthropic/claude-haiku-4.5`). The same
pattern fits any OpenAI-compatible gateway. The protocol types in
`protocol.ts` are generated from the JSON Schema, and Node runs the `.ts`
directly (>= 24).

`subs.toml` declares `[llm.byo]` with `type = "worker"`, so the engine
sends this agent's model calls back here as `llm.execute` rather than running
them itself, and `format = "openai"`, so the wire speaks the Chat Completions
API natively: the trigger's `request` is a ready-to-send Chat Completions body,
each raw stream chunk goes back as an `llm.token.delta`, and the final
completion answers the `llm.result` verbatim. No translation code in the
worker.

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
subs run -c subs.toml my-agent "hi"
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
