# go-mcp

An agent whose tools come from an [MCP](https://modelcontextprotocol.io) server,
in Go, served with `net/http`. The worker connects to an MCP server over stdio,
exposes its tools to the model, and forwards each tool call back to it. Here it
runs the filesystem server scoped to this directory; swap the command for any
MCP server.

The protocol types in `protocol.go` are generated from the JSON Schema.

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
subs run -c substructure.toml --agent my-agent "what files are in this directory?"
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
