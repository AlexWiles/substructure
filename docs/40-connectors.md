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
auth = "token"
prefix_tools = false
```

The file holds names and references. A `token` written in the file is a parse
error.

Most connections need nothing but a URL. The engine asks the server how it
authenticates: a server that challenges points at its OAuth metadata, and one
that does not want a credential answers without challenging.

Write `auth` for the one thing asking cannot answer — a static token — or to
override a server whose discovery is broken.

| `auth` | Means | Fill it with |
| --- | --- | --- |
| unset (default) | Ask the server. | `subs mcp login <id>`, if it wants one. |
| `"token"` | A static token you hold. Nothing on the wire announces this. | `subs mcp set-token <id>` |
| `"oauth"` | Use OAuth, whatever the server advertises. | `subs mcp login <id>` |
| `"none"` | Send nothing, whatever the server says. | Nothing. |

## Authorize it

```sh
subs mcp login sentry
subs mcp set-token github
subs mcp list
```

A token is typed at a prompt, piped in, or read from a variable — the same three
ways `subs llm set-key` takes a key. It never appears in the command line.

```sh
subs mcp set-token github --env GITHUB_TOKEN
gh auth token | subs mcp set-token github
```

`subs mcp logout <id>` empties a connection's credentials — every holder's —
without undeclaring it. Deleting the `[mcp.<id>]` section disconnects it for
good, and the credentials go with it.

The credential belongs to the id. Declare one server twice to connect two
accounts. Authorize each id on its own.

```toml title="substructure.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"       # subs mcp login sentry

[mcp.sentry2]
url = "https://mcp.sentry.dev/mcp"       # subs mcp login sentry2
```

An agent with both sees `sentry__` and `sentry2__` tools.

## What it asks for

`scopes` is the access consent is asked for. Declared nothing, a connection asks
for what the server advertises — which is that server's maximum, not its
recommendation. Sentry advertises writing to projects, teams and events where
reading issues needs none of it.

```toml title="substructure.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"
scopes = ["org:read"]
```

## Servers that issue no client

Most servers hand out a client identity when asked, either from a metadata
document or by dynamic registration. A few do neither — Google and GitHub among
them — and want one registered by hand.

Register it with that server, then name where the halves are kept. The file
names variables and never holds a secret, and `subs apply` strips these before
the document reaches a deployment: a client belongs to whoever registered it
against their own redirect URI.

```toml title="substructure.toml"
[mcp.gmail]
url = "https://gmailmcp.googleapis.com/mcp/v1"
credential = "user"
scopes = ["https://www.googleapis.com/auth/gmail.modify"]
client_id_env = "GMAIL_CLIENT_ID"
client_secret_env = "GMAIL_CLIENT_SECRET"
```

The redirect URI to register differs by what runs the flow. An engine's is
`<base_url>/mcp/callback`, one address that every connection shares. The CLI
binds a fresh loopback port on every run, so register it as a native or desktop
client, where the port is not matched. Where a server issues no client and the
file names none, `subs mcp login` prints the URI it bound.

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

`subs mcp login` and `subs mcp set-token` each do both. A declared connection
reaches nothing until it holds a credential.

Changing `auth` on a connection that already holds one empties it: a credential
obtained the other way is not what the file now says to send. Authorize it again.

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

## Announcing a connection

The model cannot see a connection. It sees tools, and a deferred tool is not
even that. So the engine tells it: once per connection, on the first request
that can carry the notice.

```json
{ "mcp_server": "sentry", "tools": 12, "about": "…" }
```

`about` is what the server said it is for. A server that says nothing is
announced without it.

The engine takes the first place it can use:

| Place | When |
| --- | --- |
| The system prompt | While no request of this branch has been sent. It is free there, because no cache exists yet. |
| The last user message | After that. An earlier system prompt must not change, because a change to it drops the cache. |
| A message of its own | When the turn ends on anything but a user message. |

The order is fixed, so it is not a setting. `announce_mcp` on the agent chooses
whether the engine announces at all.

```toml title="substructure.toml"
[agent.support]
mcp = ["sentry"]
announce_mcp = "never"
```

| Value | The engine |
| --- | --- |
| `auto` (the default) | Announces each connection once |
| `never` | Says nothing |

Use `never` for a server whose own words help nobody.

A connection is announced after it is authorized, and never before. A request
waits for each connection it names to answer, so a notice cannot claim a server
the engine has not reached. A connection that fails is not announced, and is
announced later if it recovers.

A branch announces on its own. A fork that never held a connection announces it
when it gets one.

## Filtering

A connection can offer a hundred tools. A model chooses worse as the list grows,
and worst between tools that look alike. Take fewer.

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

The first five keys say which tools the agent may reach. `defer` says how they
reach the model, and it is the next section.

The engine applies the capability keys, then `include`, then `exclude`. Each one
can only remove tools.

The globs match the tool's name on the connection, not the prefixed name the
model sees.

The capability keys read the connection's MCP annotations. A tool with no
annotation fails them. Annotations are hints from the server. Use them to take
fewer tools, not as a security boundary.

## Approving a call

A filter says which tools an agent may reach. `approve` says which of them stop
and ask a person first.

```toml title="substructure.toml"
[agent.support]
mcp = [{ id = "sentry", approve = "destructive" }]
```

| `approve` | Asks about |
| --- | --- |
| `never` (the default) | Nothing. |
| `destructive` | Each tool the connection marks `destructiveHint`. |
| `always` | Every call on the connection. |

The setting belongs to the pair, like `auth_failure`. One connection serves an
agent a person watches and an agent that runs on a schedule, and only the first
can be asked.

`destructive` asks about the tools a connection says destroy something, and
about no others. A tool it says nothing about is not one of them: silence is
not a claim either way, and a setting that read it as one would ask about every
tool of a server that annotates nothing.

So `destructive` is only as good as what a server says about itself, and a
server owes you no annotation at all. Where they are absent, wrong, or not
yours to trust, `always` asks about every call and depends on nothing.

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

The arguments are there because the tool's name says what would happen and only
the arguments say what it would happen to. For a deferred tool, they are the
tool's own — not the `call_tool` wrapper's.

In Slack the two options are buttons in the thread. Anywhere else, resume it.

```jsonc
{ "type": "interrupt.resume", "interrupt_id": "mcp-approve:<tool call id>", "payload": { "approved": true } }
```

`true` runs the call. Anything else declines it: the model reads that a person
declined, and answers from that instead of retrying. A payload nobody
recognizes declines, because the call this holds is the one nobody wanted run by
accident.

## One question per call

A model can ask for several calls at once. Each one that needs approval is asked
about on its own, so a person can run one and decline the next.

The questions come in turn — the answer to one raises the next — because a
branch holds one open question at a time. `metadata.remaining` counts the calls
behind this one.

Nothing of that message runs until every question is answered, including the
calls nobody asks about. A call dispatched while a question is open could settle
first, and the model would be prompted again with the held call unanswered.

An answered call runs, or gets a refusal recorded as its result. Either way the
model is prompted again only once every call it made has an answer, so it reads
what ran and what did not in one go.

The [`no-code-mcp-approval`](../examples/no-code-mcp-approval) example is this
in one file, with an MCP server to point it at.

An agent whose worker authors its decisions decides this itself: the approval is
a proposal, like every other, and a worker that writes its own answer to
`llm.finished` never sees it. See [Tool calls](./60-tools.md).

Nobody is asked where nobody is watching. A session with no channel to show the
question in stops until someone resumes it by id, so use `never` for an agent
that runs on a schedule.

## Deferring a connection

Filtering is one answer to a large connection. Search is the other. Set `defer`
and the model searches for a tool instead of reading a list of them.

```toml title="substructure.toml"
[agent.support]
mcp = [{ id = "aws", tools = { defer = true } }]
```

`defer_tools` sets it for every tool of an agent. A connection overrides it.

```toml title="substructure.toml"
[agent.support]
defer_tools = true
mcp = [
  "aws",                                         # deferred, from the agent
  { id = "sentry", tools = { defer = false } },  # this one is listed
]
```

| `defer` | The request carries |
| --- | --- |
| `false` (the default) | Each tool the filter kept. |
| `true` | None of them. |

`defer` sets the flag on each tool the filter kept. The agent then gets
`tool_search` and `call_tool`. A search gives each tool the same name the model
calls directly, such as `aws__s3_list`.

The engine answers a search from the tools it read when it opened each
connection. It does not reach the network.

An answer gives no list of connections. Each tool name carries its connector,
such as `aws__s3_list`, and an answer says how many tools it searched.

An agent can mix. A connection that does not defer puts its own tools in the
request, beside the two. The filter still applies to one that does: a search
does not show a tool the filter removed, and `call_tool` refuses one.

Use search when a connection has more tools than an agent needs at one time.
Keep the default when the agent uses most of the tools each session.

Deferral is a property of a tool, and not of MCP: a tool your worker declares
sets `defer` on its own definition, and the same two tools find it and run it.
See [Deferred tools](./65-deferred-tools.md).

A third answer is [Sub-agents](./80-sub-agents.md): give the connection to a
child agent, and the parent pays one tool.

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

The turn waits while a fetch runs. If the fetch fails for a reason that is not
the credential, the turn runs without those tools.

## When a credential stops working

An access token outlives no long session. The engine renews one before it
expires, and renews it again when a server refuses it, because a server can
revoke a token before the expiry it gave. It then tries the call again.

A registration the server has forgotten is repaired the same way. None of this
reaches you.

What a person must do is left when that fails, and it is one of three things.

| The engine found | A person must |
| --- | --- |
| Nothing stored | Authorize the connection. |
| A grant the server refuses | Authorize it again. |
| A static token the server refuses | Set a new one with `subs mcp set-token`. |

The session then stops and asks. In Slack the bot posts the question in the
thread with a link, and a `Retry` button that fetches the tools again once a
person has authorized it. See [Slack](./130-slack.md#authorizing-a-connection).

Stopping is the default because the alternative is worse: an agent that answers
from tools it no longer has reads exactly like one that used them.

An agent with nobody to ask says so instead. Set `auth_failure` on the entry
that names the connection.

```toml title="substructure.toml"
[agent.support]
mcp = ["sentry"]                                       # stops and asks

[agent.digest]
mcp = [{ id = "sentry", auth_failure = "degrade" }]    # runs without it
```

Use `degrade` for an agent that runs on a schedule, or anywhere no person is
watching. The turn goes on, the connection's tools are not offered, and the
tool error tells the model the connection needs authorizing, so it reports the
gap rather than answering around it.

A session no channel can show a question in degrades whatever this says. A
question nobody can see is a session that stops for good.

A worker can author the fetch itself with the `connector.sync` action, which is
what the `Retry` button proposes. See [Protocol](./230-protocol.md#actions).

## Next

- [Interrupts](./100-interrupts.md): the pause an approval is.
- [Tool calls](./60-tools.md): tools your worker runs.
- [Deferred tools](./65-deferred-tools.md): what `defer` turns on.
- [Agents](./30-agents.md): the section that names a connection.
- [Sub-agents](./80-sub-agents.md): put a large connector behind a child agent.
