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

**2. Send a message with the CLI.** Export the same secret the worker checks;
`substructure.toml` names the variable.

```sh
export ANTHROPIC_API_KEY=sk-ant-...
export SUBS_SIGNING_SECRET=dev-secret-not-for-production
subs run -c substructure.toml my-agent "hi"
```

Unset `SUBS_SIGNING_SECRET` and the worker answers `401 invalid signature`.
