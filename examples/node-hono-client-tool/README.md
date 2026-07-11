# node-hono-client-tool

A chattable agent with a client-side tool, served with [Hono](https://hono.dev).
The tool is declared `handler: "client"`, so the engine hands the call to the
client instead of the worker.

## Run

Two terminals.

**1. Start the worker**:

```sh
npm install
node server.mjs
```

**2. Drive a session with the CLI.** Reuse one `--session` across turns.

Ask something location-dependent. The model calls `get_location`, and the run
yields to the client with the call pending. The `TOOL_CALL_START` carries
its `toolCallId`:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --provider anthropic \
    --session $(uuidgen) \
    --input '{"type":"client.message","message":{"role":"user","content": "recommend a coffee shop near me"}}'
```

Copy the continuation message and replace the input with:

```sh
--input '{"type":"tool.result","id":"<toolCallId>","result":"Lisbon"}'
```
