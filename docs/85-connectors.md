---
title: Connectors
group: Building agents
---

A connector gives an agent the tools of a service that the engine connects to:
Sentry, GitHub, or anything that speaks MCP. The agent names a connection by id.
The engine holds the URL and the credential, reads the tools the connection
offers, and runs the calls the model makes. Your worker never sees a token, and
never runs the tool.

## Configure a connection

Declare MCP servers under `[mcp.<id>]` in
[`substructure.toml`](./160-cli.md#environments). The file holds names and
references only. It never holds a token.

```toml
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"      # authorize with `subs mcp login sentry`

[mcp.github]
url = "https://api.githubcopilot.com/mcp/"
auth = { token_env = "GITHUB_TOKEN" }   # default: Authorization: Bearer
prefix_tools = false                    # default true
```

A `token` written in the file is a parse error. You cannot commit a secret by
accident.

## Authorize it

A server that takes a static credential needs nothing more. The engine reads
`$token_env` when it makes the call.

Every other server uses OAuth, and a person must consent in a browser:

```sh
subs mcp login sentry     # opens a browser; `list` shows what is authorized
```

There is no `logout`. The file is the whole declaration, so you disconnect a
connection by deleting its `[mcp.<id>]`. The credential goes with it. A local
engine drops it at its next start, and a deployment drops it at its next apply.

The credential belongs to the id. Declare one server twice to connect two
accounts of it. You authorize each id on its own, and each prefixes its own
tools. An agent that has both sees `sentry__` and `sentry2__` tools:

```toml
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"       # subs mcp login sentry

[mcp.sentry2]
url = "https://mcp.sentry.dev/mcp"       # subs mcp login sentry2
```

The file decides where the credential is stored.

Without a `[remote]` section, the engine on this machine makes the connection.
The credential goes in that environment's `db`, with the sessions that use it. A
login and the engine that uses it stay together, and two environments authorize
on their own. **That database now holds credentials. Add `*.db*` to
`.gitignore`.**

With a `[remote]` section, the server runs the flow, and the credential never
reaches your machine. This takes two steps:

- **Declaring** records the id and the URL, and nothing else. `subs apply`
  declares every `[mcp.<id>]` in the file and grants it to the pinned app.
- **Authorizing** is the consent.

`subs mcp login` does both. It declares the connection, which does nothing if
apply already did. It opens the deployment's consent URL, waits for the consent,
and grants the connection to the app the file pins. Pass `--no-grant` to skip the
grant. A declared connection reaches nothing until someone consents. So it is
safe to declare from a manifest, but not to consent. The connectors page in the
dashboard starts the same flow, and either one can finish what the other
started.

`auth` belongs to a local engine only. It names a variable on this machine, and a
deployment cannot read it. `subs apply` and `subs mcp login` refuse a connection
that carries one, because the deployment could not authenticate with that URL.
Remove `auth` and authorize on the deployment instead. A deployment is told the
id and the URL. Some deployments allow only the URLs in their own catalog. The
error lists those URLs.

## Declare it on the agent

An agent names a connection by id. An id on its own takes every tool the
connection offers. Use the table form to take fewer:

```toml title="substructure.toml"
[agent.support]
mcp = ["sentry"]

[agent.triage]
mcp = [{ id = "sentry", tools = { read_only = true } }]
```

The filter belongs to the agent, not to the connection. So these two agents use
one `[mcp.sentry]` and one credential, and still see different tools.

A worker declares the same thing in the config it returns:

```javascript
function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                mcp: [
                    { id: "sentry", tools: { read_only: true } }
                ]
            }
        };
    }
    return proposed;
}
```

The model now sees `sentry__search_issues` next to your own tools. When it calls
one, the engine runs it and your worker is not asked. You still see the call:
`tool.finished` arrives with the result, as it does for any other tool.

## Filtering

A connection can offer a hundred tools. Above about forty tools, a model
chooses badly. Take fewer:

```typescript
type McpTools = {
    include?: string[]          // globs over the tool's name on the connection
    exclude?: string[]
    read_only?: boolean
    non_destructive?: boolean
    idempotent?: boolean
}
```

The engine applies these in order: the capability keys, then `include`, then
`exclude`. Each one can only remove tools. A filter can never add a tool that
the connection does not offer.

The globs match the tool's name **on the connection**, the name its own docs
use. They do not match the prefixed name the model sees.

The capability keys read the connection's MCP annotations. A tool with no
annotation **fails them**. So a server that annotates nothing offers no tools
under `read_only: true`. It does not pass them all through. Annotations are
hints from the server, not guarantees. Use them to take fewer tools, not as a
security boundary.

## Names

By default the engine prefixes a connection's tools with its id, such as
`sentry__search`. Set `prefix_tools = false` on the connection to use their own
names.

The engine resolves each name against everything else the model can call. If two
names match, it drops one:

- A tool you declared, or a sub-agent id, always keeps its name.
- If two connectors have the same name, both lose it.

The engine reports every name it drops. So it is safe to turn off the prefix.
You are told, instead of losing a tool without notice.

## When tools are fetched

The engine fetches a connection's tool list the first time a config names it,
and records what the connection offered. It applies the filter to that record.
So it makes no new request when you change a filter, change `prefix_tools`, or
branch the conversation. It makes a request only when a config names a
connection this session has not fetched.

The turn waits while a fetch runs. The fetch appears on your decision as a call
in flight, with `kind: "connector_sync"`. If the fetch fails, the turn runs
anyway, without those tools. You decide whether a connector you cannot reach
should stop the turn.

The engine never refreshes the list during a session. So a server that changes
during a conversation cannot change what already happened, and the model never
calls a tool that no longer exists.

## Next

- [Tool calls](./30-tools.md): tools your worker runs.
- [Sub-agents](./80-sub-agents.md): put a large connector behind a sub-agent.
