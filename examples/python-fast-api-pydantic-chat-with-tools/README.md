# python-fast-api-pydantic-chat-with-tools

A chattable agent with two tools, served with [FastAPI](https://fastapi.tiangolo.com).
The Pydantic models in `protocol.py` are generated from the JSON Schema, so
FastAPI validates every decision request at the boundary.

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
    --agent my-agent \
    --llm-provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "what time is it in my timezone?"}}'
```

## Regenerate types

`protocol.py` is generated from `schemas/protocol.schema.json` and committed,
so running the example needs only the runtime deps. To regenerate after a
protocol change:

```sh
pip install datamodel-code-generator
datamodel-codegen \
    --input ../../schemas/protocol.schema.json --input-file-type jsonschema \
    --output-model-type pydantic_v2.BaseModel \
    --disable-timestamp --collapse-root-models --use-annotated \
    --output protocol.py
```
