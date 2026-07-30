---
title: Connectors
group: Building agents
---

A connector draws tools from a service the engine connects to — Sentry, GitHub,
anything speaking MCP. The agent names a connection by id; the engine holds the
URL and the credential, fetches what that connection offers, and executes the
calls the model makes against it. Your worker never sees a token, and never runs
the tool.

## Configure a connection

MCP servers live under `[mcp.<id>]` in
[`substructure.toml`](./160-cli.md#environments), next to the rest of the
environment. The file holds names and references only — never a token.

```toml
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"      # authorize with `subs mcp login sentry`

[mcp.github]
url = "https://api.githubcopilot.com/mcp/"
auth = { token_env = "GITHUB_TOKEN" }   # default: Authorization: Bearer
prefix_tools = false                    # default true
```

An inline `token` is a parse error, not a secret you can commit by accident.

## Authorize it

A server taking a static credential needs nothing more: the engine reads
`$token_env` at call time.

Everything else is OAuth, and consent is a human in a browser:

```sh
subs mcp login sentry     # opens a browser; `list` shows what is authorized
subs mcp logout sentry    # forgets the credential
```

Where the credential lands follows the environment. A `target = "local"` file
stores it in that environment's `db`, beside the sessions that use it — so a
login and the engine that uses it cannot drift apart, and two environments
authorize independently. **That database now holds credentials: gitignore
`*.db*`.**

A `target = "remote"` file has the server run the flow instead, and the
credential never touches your machine. `subs mcp login` opens the deployment's
consent URL, waits for it to land, and grants the connection to the app the file
pins (`--no-grant` to skip). The dashboard's connectors page starts the same
flow; either surface finishes the other's.

`auth` and `prefix_tools` are local-only. A remote connection declares a URL and
nothing else: the deployment holds the credential, and where it enforces a
catalog, that URL has to be one it offers — the error lists the ones it does.

## Declare it on the agent

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

The model now sees `sentry__search_issues` alongside your own tools, and calling
one settles without your worker being asked to run it. You still see the call:
`tool.finished` fires with the outcome, like any other tool.

## Filtering

A connection can offer a hundred tools. Past roughly forty, model selection
degrades badly, so narrow it:

```typescript
type McpTools = {
    include?: string[]          // globs over the tool's name on the connection
    exclude?: string[]
    read_only?: boolean
    non_destructive?: boolean
    idempotent?: boolean
}
```

Applied in order — capability predicates, then `include`, then `exclude` — and
only ever narrowing. A filter can never widen what the connection grants.

The globs match the tool's name **on the connection** (the name its own docs
use), not the prefixed name the model sees.

Capability predicates read the connection's MCP annotations, and an
**unannotated tool fails them**. A server that annotates nothing yields nothing
under `read_only: true` rather than quietly passing everything through.
Annotations are hints from the server, not guarantees — treat them as a way to
narrow, not as a security boundary.

## Names

By default a connection's tools are prefixed with its id — `sentry__search`.
Set `prefix_tools = false` on the connection to offer them under their own
names.

Names are resolved against everything else the model can call, and a clash is
dropped rather than shadowed:

- A tool you declared, or a sub-agent id, always wins its name.
- Two connectors landing on the same name both lose it.

Every dropped name is reported, so turning prefixing off is safe — you find out
rather than silently losing a tool.

## When tools are fetched

The engine fetches a connection's tool list the first time a config names it,
and records what it offered. Filtering is applied to that record, so editing a
filter, flipping `prefix_tools`, or forking the conversation costs no round
trip; only naming a connection the session has never fetched does.

While a fetch is in flight the turn waits, and the fetch appears on your
decision as an in-flight effect (`kind: "connector_sync"`). If it fails, the
turn runs anyway — without those tools — so you can decide whether a connector
you cannot reach is fatal.

The list is never refreshed mid-session. A server that changes underneath a live
conversation cannot rewrite what already happened, and the model is never left
having called a tool that no longer exists.

## Next

- [Tool calls](./30-tools.md): tools your worker runs.
- [Sub-agents](./80-sub-agents.md): put a large connector behind a delegate.
