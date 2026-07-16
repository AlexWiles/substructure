# python-fast-api-signature

A chattable agent that verifies each request's HMAC signature before it decides,
served with [FastAPI](https://fastapi.tiangolo.com).

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

**2. Send a message with the CLI.** Pass the same secret the worker checks with
`--signing-secret`.

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --provider anthropic \
    --output pretty \
    --signing-secret dev-secret-not-for-production \
    --input '{"type":"client.message","message":{"role":"user","content": "hi"}}'
```

Drop `--signing-secret` and the worker answers `401 invalid signature`.
