# python-fast-api-anthropic

Like [`python-fast-api-basic`](../python-fast-api-basic), but the worker makes
the Claude call itself and streams the tokens back. The engine never touches
an LLM provider — it just routes decisions.

The agent config declares `handler: "worker"`, so the engine routes its LLM
calls to this worker instead of running them server-side, and
`format: "anthropic"`, so the wire speaks the Messages API natively: the
`llm.execute` trigger's `request` is a ready-to-send Messages API body, each
raw stream event goes back as an `llm.token.delta`, and the final message
answers the `llm.result` verbatim. No translation code in the worker.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker** (it makes the Claude calls, so it holds the key):

```sh
export ANTHROPIC_API_KEY=sk-ant-...
pip install -r requirements.txt
python3 main.py
```

**2. Send a message with the CLI** (no `--provider`, the worker owns the LLM):

```sh
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "hi"}}'
```
