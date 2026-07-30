# node-hono-basic

The most basic chattable agent, served with [Hono](https://hono.dev).

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker**:

```sh
npm install
node server.mjs
```

**2. Send a message with the CLI**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run -c substructure.toml \
    --input '{"type":"client.message","message":{"role":"user","content": "hi"}}'
```

