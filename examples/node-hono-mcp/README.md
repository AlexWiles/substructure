# node-hono-mcp

This example is an agent. Its tools come from an [MCP](https://modelcontextprotocol.io)
server. The agent uses [Hono](https://hono.dev) to serve the worker.

The worker connects to an MCP server with stdio. The worker shows the tools of
the server to the model. The worker sends each tool call to the MCP server.

In this example, the worker starts the filesystem server. The filesystem server
gets access to only this directory. To use a different MCP server, replace the
command.

## How to run the example

First, install the CLI:

```sh
npm i -g @substructure.ai/cli
```

You must use two terminals.

**1. Start the worker.**

```sh
npm install
node server.mjs
```

**2. Send a message with the CLI.**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "what files are in this directory?"}}'
```
