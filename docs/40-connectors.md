---
title: Connectors
group: Building agents
---

A connector gives an agent the tools of a service: Sentry, GitHub, or anything
that speaks MCP.

The agent names a connection by ID. The engine holds the URL and the credential,
reads the tools the connection offers, and runs every call. Your worker never
sees a token.

## Declare a connection

MCP servers go under `[mcp.<id>]`.

```toml title="substructure.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"

[mcp.github]
url = "https://api.githubcopilot.com/mcp/"
auth = "token"
prefix_tools = false
```

The file holds names and references. A `token` written in the file is a parse
error.

Most connections need nothing but a URL. The engine asks the server how it
authenticates: a server that issues a challenge points at its OAuth metadata,
and a server that wants no credential answers without a challenge.

Write `auth` for the one thing that asking cannot answer — a static token — or
to override a server whose discovery is broken.

| `auth` | Means | What `subs auth` does |
| --- | --- | --- |
| unset (default) | Ask the server. | Asks, then opens a browser or says what to write. |
| `"token"` | A static token you hold. Nothing on the wire announces this. | Takes the token. |
| `"oauth"` | Use OAuth, whatever the server advertises. | Opens a browser for consent. |
| `"none"` | Send nothing, whatever the server says. | Says there is nothing to do. |

## Authorize it

```sh
subs auth mcp.sentry
subs auth mcp.github
subs list
```

One verb for every kind. `subs auth` reads what the connection declares and does
what that needs — consent in a browser, or a token you type at a prompt, pipe
in, or name a variable for. A token never appears in the command line.

```sh
subs auth mcp.github --env GITHUB_TOKEN
gh auth token | subs auth mcp.github
```

`subs revoke <path>` empties a connection's credentials — every holder's —
but keeps the declaration. If you delete the `[mcp.<id>]` section instead, the
engine disconnects the connection for good, and its credentials go with it.

The credential belongs to the path. Declare one server twice to connect two
accounts. Authorize each path on its own.

```toml title="substructure.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"       # subs auth mcp.sentry

[mcp.sentry2]
url = "https://mcp.sentry.dev/mcp"       # subs auth mcp.sentry2
```

An agent names them by path, and sees `sentry__` and `sentry2__` tools.

## What it asks for

`scopes` is the access that the engine asks consent for. If you declare no
scopes, a connection asks for everything the server advertises — that server's
maximum, not its recommendation. Sentry advertises writing to projects, teams,
and events, although reading issues needs none of that access.

```toml title="substructure.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"
scopes = ["org:read"]
```

## Servers that issue no client

Most servers hand out a client identity when asked, either from a metadata
document or by dynamic registration. A few do neither — Google and GitHub among
them — and need you to register a client by hand.

Register the client with that server, then name the variables that hold its ID
and its secret. The file names variables and never holds a secret, and `subs
apply` strips both variables before the document reaches a deployment: a client
belongs to whoever registered it against their own redirect URI.

```toml title="substructure.toml"
[mcp.gmail]
url = "https://gmailmcp.googleapis.com/mcp/v1"
credential = "user"
scopes = ["https://www.googleapis.com/auth/gmail.modify"]
client_id_env = "GMAIL_CLIENT_ID"
client_secret_env = "GMAIL_CLIENT_SECRET"
```

The redirect URI that you register differs by what runs the flow. An engine uses
`<base_url>/mcp/callback`, one address that every connection shares. The CLI
binds a fresh loopback port on every run, so register the CLI as a native or
desktop client, which does not match on the port. If a server issues no client
and the file names none, `subs auth` prints the URI that it bound.

## Where the credential lives

The file decides.

| The file has | The credential goes |
| --- | --- |
| No `[remote]` | Into that environment's `db`, with the sessions that use it. |
| A `[remote]` | Into the deployment. It never reaches your machine. |

**A local database holds credentials. Add `*.db*` to `.gitignore`.**

With a `[remote]`, authorizing takes two steps.

- **Declaring** records the ID and the URL. `subs apply` declares every
  `[mcp.<id>]` in the file.
- **Authorizing** is the consent.

`subs auth` and `subs auth` each do both. A declared connection
reaches nothing until it holds a credential.

If you change `auth` on a connection that already holds a credential, the engine
empties it: a credential that the engine obtained the other way is not what the
file now says to send. Authorize the connection again.

A server that wants a header other than `Authorization: Bearer` names it, and
only under `auth = "token"`.

```toml title="substructure.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"
auth = "token"
header = "sentry-bearer"
```

Some deployments allow only the URLs in their own catalog. The error lists them.

## Give it to an agent

An agent names a connection by its path — where the connection is declared. A
path on its own gives the agent every tool that the connection offers. To take
fewer, use the table form.

```toml title="substructure.toml"
[agent.support]
mcp = ["mcp.sentry"]

[agent.triage]
mcp = [{ id = "mcp.sentry", tools = { read_only = true } }]
```

The filter belongs to the agent. These two agents share one connection and one
credential, and see different tools.

A plugin's server has a path like any other, so an agent can take one of them
without the rest of the bundle.

```toml title="substructure.toml"
[agent.searcher]
mcp = ["plugin.reggu.mcp.code"]
```

Name it once: a server granted this way and through `plugins = ["reggu"]` would
carry two policies, and that is an error rather than a winner.

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

## Announce a connection

The model cannot see a connection. It sees tools, and it cannot even see a
deferred tool. So the engine tells the model that the connection exists: once
per connection, on the first request that can carry the notice.

```json
{ "mcp_server": "sentry", "tools": 12, "about": "…" }
```

`about` is what the server said it is for. The engine announces a server that
says nothing without it.

The engine takes the first place it can use:

| Place | When |
| --- | --- |
| The system prompt | While the engine has sent no request on this branch. The notice costs nothing there, because no cache exists yet. |
| The last user message | After that. An earlier system prompt must not change, because a change to it drops the cache. |
| A message of its own | When the turn ends on anything but a user message. |

The order is fixed, so it is not a setting. `mcp_announce` on the agent chooses
whether the engine announces at all.

```toml title="substructure.toml"
[agent.support]
mcp = ["mcp.sentry"]
mcp_announce = "never"
```

| Value | The engine |
| --- | --- |
| `auto` (the default) | Announces each connection once |
| `never` | Says nothing |

Use `never` for a server whose own description does not help the model.

The engine announces a connection after someone authorizes it, and never
before. A request waits for each connection that it names to answer, so a notice
cannot announce a server that the engine has not reached. The engine does not
announce a connection that fails, and announces it later if it recovers.

## When a connection does not answer

A fetch that fails for the last time does not stop the turn. The engine goes on
without that connection's tools, because only the agent knows whether it can
work without them.

The model is told, in the prompt and in each place the absence would otherwise
read as nothing being there:

```json
{ "mcp_server": "sentry", "unavailable": true, "reason": "unreachable" }
```

`reason` is `unreachable`, or `needs_authorization` when the connection refused
the credential. The remote's own error is never in the notice; it goes to the
log, where an operator reads it.

`tool_search` names the connection too, because a search that answered nothing
would say there is nothing to find. `call_tool` names it beside a tool it cannot
place. The engine does not say the tool belongs to that connection: a fetch that
failed left no tool names behind.

`tool_sync_failure` on the connection, or `mcp_tool_sync_failure` on the agent,
chooses whether the engine says anything.

```toml title="substructure.toml"
[agent.support]
mcp_tool_sync_failure = "warn"
mcp = ["mcp.sentry", { id = "mcp.linear", tool_sync_failure = "silent" }]
```

| Value | The engine |
| --- | --- |
| `warn` (the default) | Names the connection wherever its tools are missing |
| `silent` | Says nothing |

Use `silent` for a connection the agent does not need. The agent's value is the
default for each of its connections, including each plugin's; a connection or a
plugin overrides it with its own.

A credential that a call rejects is a different case. That connection answered,
so it keeps its tools and the engine does not name it here. The model reads it
from the call that failed.

Each branch announces separately. A fork that never held a connection announces
that connection when it gets one.

## Filter the tools

A connection can offer a hundred tools. A model chooses worse as the list grows,
and worst between tools that look alike. Take fewer tools.

```typescript
type McpTools = {
    include?: string[]          // globs over the tool's name on the connection
    exclude?: string[]
    read_only?: boolean
    non_destructive?: boolean
    idempotent?: boolean
    defer?: boolean
}
```

The first five keys say which tools the agent may reach. `defer` says how those
tools reach the model, and the next section covers it.

The engine applies the capability keys, then `include`, then `exclude`. Each one
can only remove tools.

The globs match the tool's name on the connection, not the prefixed name that
the model sees.

The capability keys read the connection's MCP annotations. A tool with no
annotation does not pass them. Annotations are hints from the server. Use them
to take fewer tools, not as a security boundary.

## Approve a call

A filter says which tools an agent may reach. `approve` says which of them stop
and ask a person first.

```toml title="substructure.toml"
[agent.support]
mcp = [{ id = "mcp.sentry", approve = "destructive" }]
```

| `approve` | Asks about |
| --- | --- |
| `never` (the default) | Nothing. |
| `destructive` | Each tool the connection marks `destructiveHint`. |
| `always` | Every call on the connection. |

The setting belongs to the agent and the connection together, like
`auth_failure`. One connection can serve both an agent that a person watches and
an agent that runs on a schedule, and only the first has a person to ask.

`destructive` asks about the tools that a connection marks as destructive, and
about no others. A tool that the connection says nothing about is not one of
them. Silence is not a claim either way, and a setting that treated silence as a
claim would ask about every tool of a server that publishes no annotations.

So `destructive` is only as good as what a server says about itself, and a
server owes you no annotations at all. If the annotations are absent, wrong, or
not yours to trust, `always` asks about every call and depends on no
annotation.

A model that asks for a destructive call gets no tool result until a person
answers. The engine records the message, runs nothing, and
[interrupts](./100-interrupts.md).

```jsonc
{
  "reason": "`sentry__delete_issue` needs approval before it runs",
  "payload": {
    "message": "Run `sentry__delete_issue`?\n\n```\n{\n  \"issue\": \"PROJ-42\"\n}\n```",
    "tool_call_id": "call_a1",
    "metadata": {
      "type": "tool.approval",
      "tool": "sentry__delete_issue",
      "arguments": { "issue": "PROJ-42" },
      "remaining": 0,
      "options": [
        { "label": "Run it", "value": { "approved": true }, "style": "primary" },
        { "label": "Decline", "value": { "approved": false }, "style": "danger" }
      ]
    }
  }
}
```

The arguments are there because the tool's name says what would happen, and only
the arguments say what it would happen to. For a deferred tool, they are the
tool's own — not the `call_tool` wrapper's.

In Slack the two options are buttons in the thread. Anywhere else, resume the
interrupt yourself.

```jsonc
{ "type": "interrupt.resume", "interrupt_id": "mcp-approve:<tool call id>", "payload": { "approved": true } }
```

`true` runs the call. Anything else declines it: the model reads that a person
declined, and answers from that instead of retrying. A payload that the engine
does not recognize also declines, because a held call is exactly the call that
nobody wants run by accident.

## One question per call

A model can ask for several calls at once. The engine asks about each call that
needs approval on its own, so a person can run one and decline the next.

The questions come in turn — the answer to one raises the next — because a
branch holds one open question at a time. `metadata.remaining` counts the calls
behind this one.

The engine runs nothing from that message until a person answers every question,
including the calls that nobody is asked about. A call that the engine
dispatched while a question was open could settle first, and the engine would
prompt the model again while the held call is still unanswered.

An answered call runs, or the engine records a refusal as its result. Either
way, the engine prompts the model again only after every call it made has an
answer, so the model reads what ran and what did not together.

The [`no-code-mcp-approval`](../examples/no-code-mcp-approval) example is this
in one file, with an MCP server to point it at.

An agent whose worker writes its own decisions decides this itself: the approval
is a proposal, like every other, and a worker that writes its own answer to
`llm.finished` never sees it. See [Tool calls](./60-tools.md).

The engine cannot ask a question where nobody is watching. A session that has no
channel to show the question stops until someone resumes it by ID, so use
`never` for an agent that runs on a schedule.

## Defer a connection

A filter is one answer to a large connection. Search is the other. Set `defer`
and the model searches for a tool instead of reading a list of tools.

```toml title="substructure.toml"
[agent.support]
mcp = [{ id = "mcp.aws", tools = { defer = true } }]
```

`defer_tools` defers every tool of an agent. A connection overrides it.

```toml title="substructure.toml"
[agent.support]
defer_tools = true
mcp = [
  "mcp.aws",                                        # deferred, from the agent
  { id = "mcp.sentry", tools = { defer = false } }, # this one is listed
]
```

| `defer` | The request carries |
| --- | --- |
| `false` (the default) | Each tool the filter kept. |
| `true` | None of them. |

`defer` sets the flag on each tool that the filter kept. The agent then gets
`tool_search` and `call_tool`. A search gives each tool the same name that the
model calls directly, such as `aws__s3_list`.

The engine answers a search from the tools that it read when it opened each
connection. The search does not reach the network.

An answer gives no list of connections. Each tool name carries its connector,
such as `aws__s3_list`, and an answer says how many tools it searched.

An agent can mix the two. A connection that does not defer puts its own tools in
the request, beside `tool_search` and `call_tool`. The filter still applies to a
connection that does defer: a search does not show a tool that the filter
removed, and `call_tool` refuses to run one.

Use search when a connection has more tools than an agent needs at one time.
Keep the default when the agent uses most of the tools each session.

Deferral is a property of a tool, and not of MCP: a tool that your worker
declares sets `defer` on its own definition, and the same two tools find it and
run it.
See [Deferred tools](./65-deferred-tools.md).

A third answer is [Sub-agents](./80-sub-agents.md): give the connection to a
child agent, and the parent pays one tool.

## Names

The engine prefixes a connection's tools with its ID, such as `sentry__search`.
Set `prefix_tools = false` to use their own names.

The engine resolves each name against everything else the model can call. If two
names match, it drops one.

- A tool that you declared, or a sub-agent ID, keeps its name.
- If two connectors offer the same tool name, both lose it.

The engine reports every name it drops.

## When the engine fetches tools

The engine fetches a connection's tool list once per session, the first time a
config names it. It never refreshes the list during a session, so the model
never calls a tool that has disappeared.

The turn waits while a fetch runs. If the fetch fails for a reason that is not
the credential, the turn runs without those tools.

## When a credential stops working

An access token does not outlive a long session. The engine renews a token
before it expires, and renews it again when a server refuses it, because a
server can revoke a token before the expiry time that it gave. The engine then
tries the call again.

The engine repairs a registration that the server has forgotten the same way.
None of this reaches you.

If that fails, a person must act. There are three cases.

| The engine found | A person must |
| --- | --- |
| Nothing stored | Authorize the connection. |
| A grant the server refuses | Authorize it again. |
| A static token the server refuses | Set a new one with `subs auth`. |

The session then stops and asks. In Slack the bot posts the question in the
thread with a link, and a `Retry` button that fetches the tools again after a
person has authorized it. See [Slack](./130-slack.md#authorize-a-connection).

Stopping is the default because the alternative is worse: an agent that answers
without the tools it no longer has reads exactly like an agent that used them.

An agent with nobody to ask says so instead. Set `auth_failure` on the entry
that names the connection.

```toml title="substructure.toml"
[agent.support]
mcp = ["mcp.sentry"]                                       # stops and asks

[agent.digest]
mcp = [{ id = "mcp.sentry", auth_failure = "degrade" }]    # runs without it
```

Use `degrade` for an agent that runs on a schedule, or anywhere no person is
watching. The turn continues, the engine offers none of the connection's tools,
and the tool error tells the model that the connection needs authorizing, so the
model reports the gap instead of answering without it.

A session that has no channel to show a question in degrades whatever this
setting says. A question that nobody can see is a session that stops for good.

A worker can request the fetch itself with the `connector.sync` action, which is
what the `Retry` button proposes. See [Protocol](./230-protocol.md#actions).

## Next

- [Interrupts](./100-interrupts.md): the pause that an approval creates.
- [Tool calls](./60-tools.md): tools your worker runs.
- [Deferred tools](./65-deferred-tools.md): what `defer` turns on.
- [Plugins](./45-plugins.md): a directory that brings its own MCP servers.
- [Agents](./30-agents.md): the section that names a connection.
- [Sub-agents](./80-sub-agents.md): put a large connector behind a child agent.
