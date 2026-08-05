---
title: Connectors
group: Building agents
---

A connector gives an agent the tools of a service: Sentry, GitHub, or anything
that speaks MCP.

The agent names a connection by id. The engine holds the URL and the credential,
reads the tools the connection offers, and runs every call. Your worker never
sees a token.

## Declare a connection

MCP servers go under `[mcp.<id>]`.

```toml title="substructure.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"

[mcp.github]
url = "https://api.githubcopilot.com/mcp/"
auth = { token_env = "GITHUB_TOKEN" }
prefix_tools = false
```

The file holds names and references. A `token` written in the file is a parse
error.

## Authorize it

A server that takes a static credential needs nothing more. The engine reads
`$token_env` when it makes the call.

Every other server uses OAuth. A person must consent in a browser.

```sh
subs mcp login sentry
subs mcp list
```

There is no `logout`. Delete the `[mcp.<id>]` section to disconnect. The
credential goes with it.

The credential belongs to the id. Declare one server twice to connect two
accounts. Authorize each id on its own.

```toml title="substructure.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"       # subs mcp login sentry

[mcp.sentry2]
url = "https://mcp.sentry.dev/mcp"       # subs mcp login sentry2
```

An agent with both sees `sentry__` and `sentry2__` tools.

## Where the credential lives

The file decides.

| The file has | The credential goes |
| --- | --- |
| No `[remote]` | Into that environment's `db`, with the sessions that use it. |
| A `[remote]` | Into the deployment. It never reaches your machine. |

**A local database holds credentials. Add `*.db*` to `.gitignore`.**

With a `[remote]`, authorizing takes two steps.

- **Declaring** records the id and the URL. `subs apply` declares every
  `[mcp.<id>]` in the file.
- **Authorizing** is the consent.

`subs mcp login` does both. A declared connection reaches nothing until someone
consents.

`auth` belongs to a local engine. It names a variable on this machine. `subs
apply` and `subs mcp login` refuse a connection that carries one. Remove `auth`
and authorize on the deployment instead.

Some deployments allow only the URLs in their own catalog. The error lists them.

## Give it to an agent

An agent names a connection by id. An id on its own takes every tool the
connection offers. Use the table form to take fewer.

```toml title="substructure.toml"
[agent.support]
mcp = ["sentry"]

[agent.triage]
mcp = [{ id = "sentry", tools = { read_only = true } }]
```

The filter belongs to the agent. These two agents share one connection and one
credential, and see different tools.

A worker declares the same thing in the config it returns.

```javascript
if (trigger.type === "session.start") {
    return {
        agent: {
            model: "claude-haiku-4-5",
            mcp: [{ id: "sentry", tools: { read_only: true } }]
        }
    };
}
```

The model now sees `sentry__search_issues` beside your own tools. When it calls
one, the engine runs it. Your worker still sees the call: `tool.finished`
arrives with the result.

## Filtering

A connection can offer a hundred tools. Above about forty, a model chooses
badly. Take fewer.

```typescript
type McpTools = {
    include?: string[]          // globs over the tool's name on the connection
    exclude?: string[]
    read_only?: boolean
    non_destructive?: boolean
    idempotent?: boolean
}
```

The engine applies the capability keys, then `include`, then `exclude`. Each one
can only remove tools.

The globs match the tool's name on the connection, not the prefixed name the
model sees.

The capability keys read the connection's MCP annotations. A tool with no
annotation fails them. Annotations are hints from the server. Use them to take
fewer tools, not as a security boundary.

## Names

The engine prefixes a connection's tools with its id, such as `sentry__search`.
Set `prefix_tools = false` to use their own names.

The engine resolves each name against everything else the model can call. If two
names match, it drops one.

- A tool you declared, or a sub-agent id, keeps its name.
- If two connectors have the same name, both lose it.

The engine reports every name it drops.

## When tools are fetched

The engine fetches a connection's tool list once per session, the first time a
config names it. It never refreshes the list during a session, so the model
never calls a tool that has disappeared.

The turn waits while a fetch runs. If the fetch fails, the turn runs without
those tools.

## Next

- [Tool calls](./60-tools.md): tools your worker runs.
- [Agents](./30-agents.md): the section that names a connection.
- [Sub-agents](./80-sub-agents.md): put a large connector behind a child agent.
