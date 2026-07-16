# node-hono-openai

Like [`node-hono-basic`](../node-hono-basic), but the worker makes the OpenAI
call itself and streams the tokens back. The engine never touches an LLM
provider — it just routes decisions. The protocol types in `protocol.ts` are
generated from the JSON Schema, and Node runs the `.ts` directly (>= 24).

The agent config declares `handler: "worker"`, so the engine routes its LLM
calls to this worker instead of running them server-side, and
`format: "openai"`, so the wire speaks the Chat Completions API natively: the
`llm.execute` trigger's `request` is a ready-to-send Chat Completions body,
each raw stream chunk goes back as an `llm.token.delta`, and the final
completion answers the `llm.result` verbatim. No translation code in the
worker.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker** (it makes the OpenAI calls, so it holds the key):

```sh
export OPENAI_API_KEY=sk-...
npm install
node server.ts
```

**2. Send a message with the CLI** (no `--provider`, the worker owns the LLM):

```sh
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --output pretty \
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
