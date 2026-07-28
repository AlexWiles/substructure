# node-hono-typescript-chat-with-tools

A chattable agent with two tools, served with [Hono](https://hono.dev). The
protocol types in `protocol.ts` are generated from the JSON Schema, and Node
runs the `.ts` directly (>= 24).

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker**:

```sh
npm install
node server.ts
```

**2. Send a message with the CLI**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --llm-provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "what time is it in my timezone?"}}'
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
