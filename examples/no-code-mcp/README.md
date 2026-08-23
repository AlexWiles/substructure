# no-code-mcp

An agent whose tools come from an [MCP](https://modelcontextprotocol.io) server,
in one file. There is no worker and no code.

The file names a connection. The engine dials it, shows its tools to the model,
and runs each call.

This example reads public GitHub repositories through
[DeepWiki](https://deepwiki.com), which needs no credential, so it runs as it is.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Deploy the file and give it a key:

```sh
subs login
subs apply
subs auth llm.openrouter
```

```sh
subs run "how does routing work in honojs/hono?"
```

The model calls `deepwiki__ask_question`. The connection id is in front of the
tool name, because `prefix_tools` is on by default.

## Run it here instead

Delete `[remote]` and the engine dials the connection from this machine, on your
own key:

```sh
export OPENROUTER_API_KEY=sk-or-...
subs run -c substructure.toml --agent docs "how does routing work in honojs/hono?"
```

## Fewer tools

A connection can offer a hundred tools, and above about forty a model chooses
badly. The filter belongs to the agent, so one connection can serve an agent
that gets everything and an agent that gets two tools:

```toml
[agent.reader]
mcp = [{ id = "deepwiki", tools = { include = ["read_*"] } }]
```

`read_only`, `non_destructive`, and `idempotent` read the server's annotations.
A tool with no annotation fails them, so a server that annotates nothing gives
you no tools under `read_only`, instead of every tool.

## A server that needs a credential

Most do. Declare it, then let a person consent:

```toml
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"
```

```sh
subs auth mcp.sentry
```

The file holds the id and the URL. The credential goes where the engine runs.
