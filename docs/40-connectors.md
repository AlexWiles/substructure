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

```toml title="subs.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"

[mcp.github]
url = "https://api.githubcopilot.com/mcp/"
auth = "token"
prefix_tools = false
```

The file holds names and references. A token written in the file is a parse
error.

Most connections need nothing but a URL. The engine asks the server how it
authenticates. A server that issues a challenge points at its OAuth metadata. A
server that wants no credential answers without a challenge.

Set `auth` for the one case that asking cannot answer, a static token, or to
override a server whose discovery is broken.

| `auth` | Means | What `subs auth` does |
| --- | --- | --- |
| unset (default) | Ask the server. | Asks, then opens a browser or says what to write. |
| `"token"` | A static token you hold. Nothing on the wire announces this. | Takes the token. |
| `"oauth"` | Use OAuth, whatever the server advertises. | Opens a browser for consent. |
| `"none"` | Send nothing, whatever the server says. | Says there is nothing to do. |

## Authorize a connection

```sh
subs auth mcp.sentry
subs auth mcp.github
subs list
```

One command covers every kind. `subs auth` reads what the connection declares
and does what that needs: consent in a browser, or a token you type at a prompt
or pipe in. A token never appears in the command line.

```sh
subs auth mcp.github
gh auth token | subs auth mcp.github
```

`subs revoke <path>` empties a connection's credentials, every holder's, but
keeps the declaration. Deleting the `[mcp.<id>]` section instead removes the
connection and its credentials permanently.

The credential belongs to the path. Declare one server twice to connect two
accounts, then authorize each path on its own.

```toml title="subs.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"       # subs auth mcp.sentry

[mcp.sentry2]
url = "https://mcp.sentry.dev/mcp"       # subs auth mcp.sentry2
```

An agent names them by path, and sees `sentry__` and `sentry2__` tools.

## Scopes

`scopes` is the access that the engine asks consent for. If you declare no
scopes, the connection asks for everything the server advertises. That list is
the server's maximum, not its recommendation. Sentry advertises writing to
projects, teams, and events, although reading issues needs none of that access.

```toml title="subs.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"
scopes = ["org:read"]
```

## Register an OAuth client by hand

Most servers hand out a client identity when asked, either from a metadata
document or by dynamic registration. A few do neither, Google and GitHub among
them, and need you to register a client yourself.

Register the client with that server, then name the variables that hold its ID
and its secret. The file names the variables and never holds the secret. `subs
apply` strips both variables before the document reaches a deployment, because a
client belongs to whoever registered it against their own redirect URI.

```toml title="subs.toml"
[mcp.gmail]
url = "https://gmailmcp.googleapis.com/mcp/v1"
credential = "user"
scopes = ["https://www.googleapis.com/auth/gmail.modify"]
client_id_env = "GMAIL_CLIENT_ID"
client_secret_env = "GMAIL_CLIENT_SECRET"
```

The redirect URI you register depends on what runs the flow. An engine uses
`<base_url>/mcp/callback`, one address that every connection shares. The CLI
binds a fresh loopback port on every run, so register the CLI as a native or
desktop client, which does not match on the port. If a server issues no client
and the file names none, `subs auth` prints the URI that it bound.

## Where the credential is stored

The file decides.

| The file has | The credential goes |
| --- | --- |
| No `[remote]` | Into that environment's `db`, with the sessions that use it. |
| A `[remote]` | Into the deployment. It never reaches your machine. |

**A local database holds credentials. Add `*.db*` to `.gitignore`.**

With a `[remote]`, setting up a connection takes two steps.

- **Declaring** records the ID and the URL. `subs apply` declares every
  `[mcp.<id>]` in the file.
- **Authorizing** is the consent. `subs auth` does this.

A declared connection reaches nothing until it holds a credential.

If you change `auth` on a connection that already holds a credential, the engine
empties it. A credential the engine obtained the other way is not what the file
now says to send. Authorize the connection again.

A server that wants a header other than `Authorization: Bearer` names it, and
only under `auth = "token"`.

```toml title="subs.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"
auth = "token"
header = "sentry-bearer"
```

Some deployments allow only the URLs in their own catalog. The error lists them.

## Give a connection to an agent

An agent names a connection by its path, which is where the connection is
declared. A path on its own gives the agent every tool that the connection
offers. To take fewer, use the table form.

```toml title="subs.toml"
[agent.support]
mcp = ["mcp.sentry"]

[agent.triage]
mcp = [{ id = "mcp.sentry", tools = { read_only = true } }]
```

The filter belongs to the agent. These two agents share one connection and one
credential, and see different tools.

A plugin's server has a path like any other, so an agent can take one of them
without the rest of the bundle.

```toml title="subs.toml"
[agent.searcher]
mcp = ["plugin.docs.mcp.code"]
```

Name a server once. Granting the same server through `mcp` and through
`plugins = ["docs"]` gives it two policies, and the engine reports that as an
error instead of picking one.

A worker declares the same thing in the config it returns. On the wire the key
is `path` and it holds the full connection path.

```javascript
if (trigger.type === "session.start") {
    return {
        agent: {
            model: "claude-haiku-4-5",
            mcp: [{ path: "mcp.sentry", tools: { read_only: true } }]
        }
    };
}
```

The model now sees `sentry__search_issues` beside your own tools. When it calls
one, the engine runs it. Your worker still sees the call: `tool.finished`
arrives with the result.

## Tell the model a connection exists

The model cannot see a connection. It sees tools, and it cannot see a deferred
tool at all. So the engine tells the model that the connection exists, once per
connection, on the first request that can carry the notice.

```json
{ "mcp_server": "mcp.sentry", "tools": 12, "about": "…" }
```

`about` is what the server said it is for. A server that says nothing is
announced without it.

The engine puts the notice in the first place it can use.

| Place | When |
| --- | --- |
| The system prompt | While the engine has sent no request on this branch. The notice costs nothing there, because no cache exists yet. |
| The last user message | After that. An earlier system prompt must not change, because a change to it drops the cache. |
| A message of its own | When the turn ends on anything but a user message. |

The order is fixed and is not a setting. `mcp_announce` on the agent sets
whether the engine announces at all.

```toml title="subs.toml"
[agent.support]
mcp = ["mcp.sentry"]
mcp_announce = "never"
```

| Value | The engine |
| --- | --- |
| `auto` (the default) | Announces each connection once |
| `never` | Says nothing |

Use `never` for a server whose own description does not help the model.

The engine announces a connection after someone authorizes it, and never before.
A request waits for each connection that it names to answer, so a notice cannot
announce a server that the engine has not reached. The engine does not announce
a connection that fails, and announces it later if it recovers.

## Connection failures

A fetch that fails for the last time does not stop the turn. The engine goes on
without that connection's tools, because only the agent knows whether it can
work without them.

The engine tells the model, in the prompt and in each place where the absence
would otherwise read as nothing being there.

```json
{ "mcp_server": "mcp.sentry", "unavailable": true, "reason": "unreachable" }
```

`reason` is `unreachable`, or `needs_authorization` when the connection refused
the credential. The remote's own error is never in the notice. It goes to the
log, where an operator reads it.

`tool_search` names the connection too, because a search that answered nothing
would say there is nothing to find. `call_tool` names it beside a tool it cannot
place. The engine does not say the tool belongs to that connection, because a
fetch that failed left no tool names behind.

`tool_sync_failure` on the connection, or `mcp_tool_sync_failure` on the agent,
sets whether the engine says anything.

```toml title="subs.toml"
[agent.support]
mcp_tool_sync_failure = "warn"
mcp = ["mcp.sentry", { id = "mcp.linear", tool_sync_failure = "silent" }]
```

| Value | The engine |
| --- | --- |
| `warn` (the default) | Names the connection wherever its tools are missing |
| `silent` | Says nothing |

Use `silent` for a connection the agent does not need. The agent's value is the
default for each of its connections, including each plugin's. A connection or a
plugin overrides it with its own.

A credential that a call rejects is a different case. That connection answered,
so it keeps its tools and the engine does not name it here. The model reads the
problem from the call that failed.

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

The first five keys set which tools the agent can reach. `defer` sets how those
tools reach the model, and [Defer a connection](#defer-a-connection) covers it.

The engine applies the capability keys, then `include`, then `exclude`. Each one
can only remove tools.

The globs match the tool's name on the connection, not the prefixed name that
the model sees.

The capability keys read the connection's MCP annotations. A tool with no
annotation does not pass them. Annotations are hints from the server. Use them
to take fewer tools, not as a security boundary.

## Ask a person before a call runs

A filter sets which tools an agent can reach. `approve` sets which of them stop
and ask a person first.

```toml title="subs.toml"
[agent.support]
mcp = [{ id = "mcp.sentry", approve = "destructive" }]
```

| `approve` | Asks about |
| --- | --- |
| `never` (the default) | Nothing. |
| `destructive` | Each tool the connection marks `destructiveHint`. |
| `always` | Every call on the connection. |

The setting belongs to the agent and the connection together. One connection can
serve both an agent that a person watches and an agent that runs on a schedule,
and only the first has a person to ask.

`destructive` asks about the tools that a connection marks as destructive, and
about no others. A tool that carries no annotation is not one of them.

So `destructive` is only as good as the annotations a server publishes, and a
server does not have to publish any. If the annotations are absent, wrong, or
not yours to trust, use `always`. It asks about every call and reads no
annotation.

A model that asks for a destructive call gets no tool result until a person
answers. The engine records the message, runs nothing, and
[interrupts](./100-interrupts.md).

```jsonc
{
  "reason": "`sentry__delete_issue` needs approval before it runs",
  "payload": {
    "message": "Run `sentry__delete_issue`?\n\n```\n{\n  \"issue\": \"PROJ-42\"\n}\n```",
    "toolCallId": "call_a1",
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

The payload carries the arguments because the tool's name says what would
happen, and only the arguments say what it would happen to. For a deferred tool,
these are the tool's own arguments, not the `call_tool` wrapper's.

In Slack the two options are buttons in the thread. Anywhere else, resume the
interrupt yourself.

```jsonc
{ "type": "interrupt.resume", "interrupt_id": "mcp-approve:<tool call id>", "payload": { "approved": true } }
```

`true` runs the call. Anything else declines it. The model reads that a person
declined and answers from that instead of retrying. A payload that the engine
does not recognize also declines, so a malformed answer never runs a held
call.

### Approve several calls

A model can ask for several calls at once. The engine asks about each call that
needs approval on its own, so a person can run one and decline the next.

The questions come one at a time, because a branch holds one open question at a
time. Answering one raises the next. `metadata.remaining` counts the calls
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

An agent whose worker writes its own decisions decides this itself. The approval
is a proposal, like every other, and a worker that writes its own answer to
`llm.finished` never sees it. See [Tool calls](./60-tools.md).

The engine cannot ask a question where nobody is watching. A session that has no
channel to show the question stops until someone resumes it by ID, so use
`never` for an agent that runs on a schedule.

## Defer a connection

A filter is one answer to a large connection. Search is the other. Set `defer`
and the model searches for a tool instead of reading a list of tools.

```toml title="subs.toml"
[agent.support]
mcp = [{ id = "mcp.aws", tools = { defer = true } }]
```

`defer_tools` defers every tool of an agent. A connection overrides it.

```toml title="subs.toml"
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

Deferral is a property of a tool, not of MCP. A tool that your worker declares
sets `defer` on its own definition, and the same two tools find it and run it.
See [Deferred tools](./65-deferred-tools.md).

A third answer is [Subagents](./80-subagents.md): give the connection to a
child agent, and the parent pays one tool.

## Tool names

The engine prefixes a connection's tools with its ID, such as `sentry__search`.
Set `prefix_tools = false` to use their own names.

The engine resolves each name against everything else the model can call. If two
names match, it drops one.

- A tool that you declared, or a subagent ID, keeps its name.
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

Stopping is the default because an agent that answers without its tools looks
the same as an agent that used them. Nobody can tell that the answer is
incomplete.

An agent with nobody to ask says so instead. Set `auth_failure` on the entry
that names the connection.

```toml title="subs.toml"
[agent.support]
mcp = ["mcp.sentry"]                                       # stops and asks

[agent.digest]
mcp = [{ id = "mcp.sentry", auth_failure = "degrade" }]    # runs without it
```

Use `degrade` for an agent that runs on a schedule, or anywhere no person is
watching. The turn continues, the engine offers none of the connection's tools,
and the tool error tells the model that the connection needs authorizing, so the
model reports the gap instead of answering without it.

A session that has no channel to show a question degrades whatever this setting
says, because nobody would ever see the question and the session would never
resume.

A worker can request the fetch itself with the `connector.sync` action, which is
what the `Retry` button proposes. See [Protocol](./230-protocol.md#actions).

## Next steps

- [Interrupts](./100-interrupts.md): the pause that an approval creates.
- [Tool calls](./60-tools.md): tools your worker runs.
- [Deferred tools](./65-deferred-tools.md): what `defer` turns on.
- [Plugins](./45-plugins.md): a directory that brings its own MCP servers.
- [Agents](./30-agents.md): the section that names a connection.
- [Subagents](./80-subagents.md): put a large connector behind a child agent.
