# go-tools

A chattable agent with two tools, served with Go's
[net/http](https://pkg.go.dev/net/http).

## Run

Two terminals.

**1. Start the worker**:

```sh
go run .
```

**2. Send a message with the CLI**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "what time is it in my timezone?"}}'
```
