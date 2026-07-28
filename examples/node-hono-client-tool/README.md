# node-hono-client-tool

A chattable agent with a client-side tool, served with [Hono](https://hono.dev).
The tool is declared `handler: "client"`, so the engine hands the call to the
client instead of the worker.

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

**2. Drive a session with the CLI.** Reuse one `--session` across turns.

Ask something location-dependent. The model calls `get_location`, and the run
yields to the client with the call pending. Pretty output prints the pending
call with its id and a ready-to-edit settle input:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --llm-provider anthropic \
    --output pretty \
    --session $(uuidgen) \
    --input '{"type":"client.message","message":{"role":"user","content": "recommend a coffee shop near me"}}'
```

Take the printed `continue this session with` command and swap its input for the
settle line, filling in a location:

```sh
--input '{"type":"tool.result","id":"<toolCallId>","result":"Lisbon"}'
```
