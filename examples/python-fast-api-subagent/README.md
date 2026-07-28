# python-fast-api-subagent

A chattable agent that delegates weather questions to a sub-agent, served with
[FastAPI](https://fastapi.tiangolo.com).

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker**:

```sh
pip install -r requirements.txt
python3 main.py
```

**2. Send a message with the CLI**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent assistant \
    --llm-provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "what is the weather in Paris?"}}'
```
