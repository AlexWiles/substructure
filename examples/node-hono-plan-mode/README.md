# node-hono-plan-mode

A chattable agent that plans over several turns, then executes the plan on a
signal — served with [Hono](https://hono.dev).

## Run

Two terminals.

**1. Start the worker**:

```sh
npm install
node server.mjs
```

**2. Drive a session with the CLI.** Reuse one `--session` across turns.

Plan:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent planner \
    --provider anthropic \
    --output pretty \
    --session plan-demo \
    --input '{"type":"client.message","message":{"role":"user","content": "plan a weekend trip to Lisbon"}}'
```

Flip to execution — a `client.action`, not a chat message, so it never lands in
the transcript:

```sh
subs run \
    --worker-url http://localhost:4444 \
    --agent planner \
    --provider anthropic \
    --output pretty \
    --session plan-demo \
    --input '{"type":"client.action","name":"set_mode","args":{"mode":"executing"}}'
```
