# node-hono-mcp

An agent whose tools come from an [MCP](https://modelcontextprotocol.io) server,
served with [Hono](https://hono.dev).

The worker connects to an MCP server over stdio, exposes its tools to the model,
and forwards each tool call back to the MCP server. Here it runs the filesystem
server scoped to this directory; swap the command for any MCP server.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker**:

```sh
npm install
node server.mjs
```

**2. Send a message with the CLI**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "what files are in this directory?"}}'
```
