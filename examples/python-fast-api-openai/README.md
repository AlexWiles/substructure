# python-fast-api-openai

Like [`python-fast-api-basic`](../python-fast-api-basic), but the worker makes
the OpenAI call itself and streams the tokens back. The engine never touches
an LLM provider — it just routes decisions.

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
pip install -r requirements.txt
python3 main.py
```

**2. Send a message with the CLI** (no `--llm-provider`, the worker owns the LLM):

```sh
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "hi"}}'
```
