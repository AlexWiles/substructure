# python-fast-api-anthropic

Like [`python-fast-api-basic`](../python-fast-api-basic), but the worker makes
the Claude call itself and streams the tokens back. The engine never touches
an LLM provider — it just routes decisions.

`substructure.toml` declares `[llm.byo]` with `type = "worker"`, so the engine
sends this agent's model calls back here as `llm.execute` rather than running
them itself, and `format = "anthropic"`, so the wire speaks the Messages API
natively: the trigger's `request` is a ready-to-send Messages API body, each
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

**2. Send a message with the CLI** (no `[llm]` section, the worker owns the LLM):

```sh
subs run -c substructure.toml my-agent "hi"
```
