# node-hono-signature

A chattable agent that verifies each request's HMAC signature in Hono
middleware before it decides, served with [Hono](https://hono.dev).

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

**2. Send a message with the CLI.** Pass the same secret the worker checks with
`--signing-secret`.

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --llm-provider anthropic \
    --output pretty \
    --signing-secret dev-secret-not-for-production \
    --input '{"type":"client.message","message":{"role":"user","content": "hi"}}'
```

Drop `--signing-secret` and the worker answers `401 invalid signature`.
