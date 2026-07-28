# node-hono-connectors

This example is an agent. Its tools come from a connector: an
[MCP](https://modelcontextprotocol.io) server the engine connects to. The agent
uses [Hono](https://hono.dev) to serve the worker.

The worker names a connection and nothing else. It does not see the URL, does
not hold the token, and does not run the tool. The engine fetches the tool list,
shows it to the model, and executes each call.

Compare with [node-hono-mcp](../node-hono-mcp), where the worker speaks MCP
itself.

The example includes a small MCP server, so it runs with no account anywhere.
The server uses the official MCP SDK.

## How to run the example

First, install the CLI:

```sh
npm i -g @substructure.ai/cli
```

You must use three terminals.

**1. Start the MCP server.**

```sh
npm install
node mcp-server.mjs
```

**2. Start the worker.**

```sh
node server.mjs
```

**3. Send a message with the CLI.** Run it from this directory, because the CLI
reads `substructure.toml` from here.

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --agent my-agent \
    --input '{"type":"client.message","message":{"role":"user","content": "which issues are open?"}}'
```

`substructure.toml` supplies the worker URL, the provider, and the output mode,
so they are not on the command line. A flag still wins over the file.

The model calls `issues__search_issues`. The tool name has the connection id in
front of it, because `prefix_tools` is on by default.

## The connection

`substructure.toml` holds the connections. The example server needs no
credential, so this connection has no `auth`:

```toml
[mcp.issues]
url = "http://localhost:4445/mcp"
```

A real service needs one. The file holds the name of the variable with the
token, never the token:

```toml
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"
auth = { header = "Sentry-Bearer", token_env = "SENTRY_TOKEN" }
```

## The filter

The MCP server offers three tools. The agent asks for the read-only ones:

```javascript
mcp: [{ id: "issues", tools: { read_only: true } }]
```

So the model sees `search_issues` and `get_issue`, but not `close_issue`. Ask it
to close an issue and it answers that it cannot.

A tool with no annotations fails the filter. A server that annotates nothing
gives you no tools under `read_only`, instead of every tool.

## Use a real service

Uncomment a connection in `substructure.toml`, set the variable it names, and
add the id to `mcp` in `server.mjs`. Nothing else changes.
