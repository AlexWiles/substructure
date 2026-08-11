# no-code-deferred-tools

An agent that searches for a tool instead of reading a list of them, in one
file. There is no worker and no code.

`defer_tools` keeps every tool definition out of the request. The engine still
holds them, so the agent gets two tools in their place: `tool_search` and
`call_tool`.

```toml
[agent.docs]
mcp = ["deepwiki"]
defer_tools = true
```

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Deploy the file and give it a key:

```sh
subs login
subs apply
subs llm set-key openrouter
```

```sh
subs run "how does routing work in honojs/hono?"
```

The model calls `tool_search` with a query, reads the schema it gets back, then
calls `call_tool` with the name exactly as the search gave it:

```jsonc
{ "query": "read a repository" }
{ "name": "deepwiki__ask_question", "arguments": { "repoName": "honojs/hono", "question": "…" } }
```

Drop `defer_tools = true` and run it again. The model calls
`deepwiki__ask_question` directly, because the request now carries its
definition. That is the whole difference.

## Run it here instead

Delete `[remote]` and the turn runs on this machine, on your own key:

```sh
export OPENROUTER_API_KEY=sk-or-...
subs run -c substructure.toml --agent docs "how does routing work in honojs/hono?"
```

## Why

Tool definitions sit at the front of the request, before the conversation. A
provider caches an exact prefix, so a definition that moves costs the cache of
everything behind it — and above about forty tools a model chooses badly
anyway.

DeepWiki offers a handful of tools, so this example is a demonstration and not
an optimization. Reach for it when a connection offers a hundred.

## Adding a connection later

The two tools say nothing about which connections exist. Their definitions are
identical on turn 1 and turn 400, so a connection added to the file mid-session
costs no cache: the model reads the new connection in the next search.

```toml
[agent.docs]
mcp = ["deepwiki", "sentry"]
defer_tools = true

[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"
```

```sh
subs mcp login sentry
```

One search covers both connections. Each name says which connection holds the
tool, and an answer says how many tools it searched.

## Settings

`defer_tools = true` takes the defaults. Write a table to set them.

```toml
[agent.docs.defer_tools]
strategy = "search"    # which tools find the deferred ones. the only value today
max_matches = 10       # how many matches one search answers with. default 5
```

The presence of `defer_tools` is the switch, so an agent cannot carry a setting
that does nothing.

## Mixing

Deferral is a property of a tool, not of MCP. One connection can defer while
another is listed, and a tool your worker declares sets `defer` on its own
definition.

```toml
[agent.docs]
defer_tools = true
mcp = [
  "aws",                                           # deferred, from the agent
  { id = "deepwiki", tools = { defer = false } },  # this one is listed
]
```

The filter still applies. A search does not show a tool `include`/`exclude`
removed, and `call_tool` refuses one.

See [Deferred tools](../../docs/65-deferred-tools.md).
