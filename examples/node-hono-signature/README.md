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

**2. Send a message with the CLI.** Export the same secret the worker checks;
`subs.toml` names the variable.

```sh
export ANTHROPIC_API_KEY=sk-ant-...
export SUBS_SIGNING_SECRET=dev-secret-not-for-production
subs run -c subs.toml my-agent "hi"
```

Unset `SUBS_SIGNING_SECRET` and the worker answers `401 invalid signature`.
