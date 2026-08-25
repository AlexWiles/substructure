# go-basic

The most basic chattable agent in Go, served with `net/http`. The protocol
types in `protocol.go` are generated from the JSON Schema, and the standard
library does the rest.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker**:

```sh
go run .
```

**2. Send a message with the CLI**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run -c substructure.toml my-agent "hi"
```

## Regenerate types

`protocol.go` is generated from `schemas/protocol.schema.json` and committed,
so building the example needs no tooling. To regenerate after a protocol
change:

```sh
npx quicktype --src-lang schema --lang go \
    --src ../../schemas/protocol.schema.json \
    --top-level Protocol --package main -o protocol.go
gofmt -w protocol.go
```
