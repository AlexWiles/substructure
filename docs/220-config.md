---
title: Config
group: Reference
---

`substructure.toml` declares one project. Every key is here.

The CLI reads the file from the working directory, or from the path you pass
with `-c`. It does not search parent directories.

An unknown key is a parse error.

## A full file

```toml title="substructure.toml"
name = "support-bot"
db = "substructure.db"
log = "info"

[llm.claude]
type = "anthropic"

[agent.support]
llm = "claude"
model = "claude-sonnet-4-5"
system = "You are a support agent."
mcp = ["mcp.sentry"]
sub_agents = ["researcher"]

[agent.researcher]
description = "Finds and reads sources."
llm = "claude"
model = "claude-haiku-4-5"

[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"

[slack]
dm = "support"
mentions = "support"

[run]
agent = "support"
output = "pretty"

[serve]
host = "127.0.0.1"
port = 8080

[remote]
url = "https://api.substructure.ai"
org = "org_01hx…"
project = "proj_01hx…"
```

## Two roles

A file has two roles. It can have one or both.

| Role | Keys | Commands |
| --- | --- | --- |
| An engine you run | `db`, `log`, `[run]`, `[serve]` | `subs run`, `subs serve` |
| A deployment you administer | `[remote]` | `subs apply`, `subs keys`, `subs sessions` |

The project itself stays the same for both roles: `name`, `[llm.<id>]`,
`[agent.<id>]`, `[mcp.<id>]`, `[plugin.<id>]`, and `[slack]`.

A second environment is a second file. `subs apply -c substructure.staging.toml`
deploys a separate project.

## Top level

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | string | none | The project's name. `subs apply` creates the project from it and renames it when it changes. |
| `db` | path | the file's name with a `.db` suffix | The SQLite file holding events, sessions, and connector credentials. A relative path resolves against the file. |
| `log` | string | `error` for `run`, `info` for `serve` | A `RUST_LOG` filter. `$RUST_LOG` wins over it. |

`substructure.toml` uses `substructure.db`. `subs.staging.toml` uses
`subs.staging.db`. Two files in one directory are two engines.

With no file at all, the engine uses
`~/.config/substructure/substructure.db`, beside `credentials.toml`, so
`subs chat` and `subs run` keep one set of sessions wherever you run them.

## `[llm.<id>]`

Where a model call runs. An agent names a block by its ID.

```toml
[llm.claude]
type = "anthropic"
api_key_env = "MY_ANTHROPIC_KEY"

[llm.byo]
type = "worker"
format = "anthropic"
```

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `type` | `anthropic`, `openai`, `openrouter`, `worker` | required | Who makes the call. |
| `api_key_env` | string | the vendor's own variable | The variable holding the key. For an engine you run. |
| `base_url` | url | the vendor's own | Where to send the call. |
| `format` | `openai`, `anthropic` | the engine's own shape | The wire shape of the `llm.execute` a worker answers. `type = "worker"` only. |
| `cache_ttl` | `5m`, `1h` (`anthropic`, `openrouter`); `in_memory`, `24h` (`openai`) | the vendor's own | How long the vendor holds a cached prompt prefix. |

There is no default block and no fallback. An agent names a block, or its calls
fail. See [LLMs](./70-llms.md).

`api_key_env` names a variable on your machine. `subs apply` removes it. A
deployment refuses a document that carries one.

## `[agent.<id>]`

An agent. The ID is what clients, channels, and parent agents route on.

```toml
[agent.support]
llm = "claude"
model = "claude-sonnet-4-5"
system = "You are a support agent."
mcp = ["mcp.sentry"]
sub_agents = ["researcher"]
tools = [{ name = "confirm", description = "Ask a person", handler = "client" }]
worker = "https://bot.example.com/agent"
signing_secret_env = "SUPPORT_SIGNING_SECRET"

[agent.support.retry]
tool = { max_attempts = 3 }
```

| Key | Type | Meaning |
| --- | --- | --- |
| `llm` | string | The `[llm.<id>]` block. Required when the section sets anything. |
| `model` | string | The model. Required when the section sets anything. |
| `system` | string | The system prompt. |
| `effort` | string | How hard the model thinks: `xhigh`, `high`, `medium`, `low`, `minimal`, or `none`. It sits on the agent because it pairs with the model. |
| `description` | string | What this agent does, shown to a parent that calls it. |
| `mcp` | list | Connections. An ID, or a table to take fewer tools, to put them behind a search, to stop before a call, or to go on without one that needs authorizing: `{ id, tools, approve, auth_failure }`. `tools` sets the filter and `defer`; see [Defer a connection](./40-connectors.md#defer-a-connection). `approve` is `never` (the default), `destructive`, or `always`; see [Approve a call](./40-connectors.md#approve-a-call). `auth_failure` is `interrupt` (the default) or `degrade`; see [Connectors](./40-connectors.md#when-a-credential-stops-working). |
| `defer_tools` | bool or table | Absent by default. Keeps every tool of this agent out of the request, whatever its source. `true` takes the defaults; a table sets them, with `strategy` for which tools find the deferred ones (`search`, the only value today) and `max_matches` for how many a search answers with (`5`). A tool or a connection overrides it with its own `defer`. See [Deferred tools](./65-deferred-tools.md). |
| `announce_mcp` | string | `auto` by default. Where the engine tells the model that a connection is available: `auto` or `never`. See [Announce a connection](./40-connectors.md#announce-a-connection). |
| `plugins` | list | Plugins this agent uses. An ID, or a table with the same knobs as an `mcp` entry, applied to the plugin's servers: `{ id, tools, approve, auth_failure }`. See [Plugins](./45-plugins.md). |
| `sub_agents` | list of IDs | Agents this one can call. |
| `tools` | list | Browser tools. Each needs `handler = "client"`. |
| `worker` | url | Where decisions go. Leave it off and the engine decides. |
| `signing_secret_env` | string | The variable holding the signing secret. For an engine you run. |
| `retry` | table | Timeouts and attempts, per kind. See [Retries](./210-retries.md). |

An agent that sets nothing needs a `worker`. An agent that sets anything needs
`llm` and `model`. See [Agents](./30-agents.md).

The tools your worker runs are worker code. They do not go in the file.

## `[mcp.<id>]`

An MCP server the engine connects to.

```toml
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"

[mcp.github]
url = "https://api.githubcopilot.com/mcp/"
auth = "token"
prefix_tools = false
```

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `url` | url | required | The server. |
| `auth` | string | ask the server | Override how it authenticates: `"token"`, `"oauth"`, or `"none"`. |
| `header` | string | `Authorization` | Header a static token rides in. Only under `auth = "token"`. |
| `credential` | string | `"shared"` | Whose credential the connection dials with: `"shared"` for one, `"user"` for one per person. |
| `scopes` | list | ask the server | The access to ask consent for. The server's own list is its maximum, not its recommendation. |
| `client_id_env` | string | — | Variable holding the OAuth client, for a server that issues none. Named, never written. |
| `client_secret_env` | string | — | The secret half. Only alongside `client_id_env`. |
| `prefix_tools` | bool | `true` | Show the model `<id>__<tool>` instead of the connection's own names. |

A `token` written in the file is a parse error. Fill a connection with
`subs auth <path>`. See [Connectors](./40-connectors.md).

## `[plugin.<id>]`

A plugin directory the CLI resolves and sends to the deployment. An agent names
a plugin by its ID.

```toml
[plugin.pdf]
path = "./plugins/pdf-tools"

[plugin.pdf.mcp.renderer]
auth = "none"
```

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `path` | path | required | The plugin directory. A relative path resolves against the file. |
| `mcp.<server>` | table | the plugin's own | What this deployment says about one of the plugin's servers. See [`[plugin.<id>.mcp.<server>]`](#pluginidmcpserver). |

The CLI resolves the directory to data — at startup for a local engine, at
`subs apply` for a deployment — so a session never reads plugin files. A
plugin's servers join the connection registry as `plugin.<id>.mcp.<server>`
and authorize like any connection. The model sees their tools under
`<plugin>_<server>__<tool>`. See [Plugins](./45-plugins.md).


### `[plugin.<id>.mcp.<server>]`

What this deployment says about one server the plugin declares, keyed by its
name in the plugin's `mcp.json`. Every key overrides the plugin's; one left out
keeps what the plugin shipped.

```toml
[plugin.reggu.mcp.code]
auth = "token"
url = "https://reggu-code.staging.example.com/mcp"
```

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `url` | url | the plugin's | Where this deployment reaches the server. |
| `auth` | string | ask the server | `"oauth"`, `"token"`, or `"none"`. `mcp.json` has no field for it. |
| `header` | string | `Authorization` | Header carrying a static token. Only under `auth = "token"`. |
| `credential` | string | `shared` | `shared` or `user`. |
| `scopes` | list | — | The access to ask consent for. |
| `client_id_env` | string | — | Variable holding the OAuth client. Named, never written. |
| `client_secret_env` | string | — | The secret half. |
| `prefix_tools` | bool | `true` | Show the model `<id>__<tool>`. |

Authorize it by its path: `subs auth plugin.reggu.mcp.code`.

## `[slack]`

Where the bot answers. Every key defaults to silence.

```toml
[slack]
dm = "support"
mentions = "support"

[slack.channel.C0ENGOPS]
agent = "oncall"

[slack.channel.C0RANDOM]
off = true
```

| Key | Type | Meaning |
| --- | --- | --- |
| `dm` | agent ID | Answers direct messages. |
| `mentions` | agent ID | Answers mentions in any channel `channel` does not name. |
| `channel.<id>.agent` | agent ID | Answers in that channel. |
| `channel.<id>.off` | bool | The bot stays out of that channel. |

Name a channel by ID, never by name. A `#name` is a parse error. See
[Slack](./130-slack.md).

## `[run]`

Defaults for `subs run`.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `agent` | agent ID | none | Which agent a bare `subs run` uses. |
| `output` | `ag-ui`, `jsonl`, `pretty` | `ag-ui` | How to print the turn. |

## `[serve]`

Defaults for `subs serve`.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `host` | string | `127.0.0.1` | The address to bind. |
| `port` | number | `8080` | The port. |
| `auth` | bool | `true` | Client and worker authentication. Set `false` only for a server nothing off this machine can reach. |

## `[remote]`

The deployment this file administers. That can be the hosted cloud, one you
host, or another person's `subs serve`.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `url` | url | `https://api.substructure.ai` | The API to talk to. |
| `org` | ID | none | Written by `subs link` or `subs apply`. |
| `project` | ID | none | Written by `subs apply` when it creates the project. |

`subs apply` writes the pin back into the file and keeps your comments. A second
apply changes nothing.

## Precedence

**flag > environment variable > `substructure.toml` > default**

Setting a value in the file still lets you override it on the command line.

The message, `--input`, and `--session` have no key in the file. They say what
one run does.

## Secrets

The file names secrets. It never holds them.

| Secret | How the file refers to it |
| --- | --- |
| Provider key | `api_key_env` on the LLM block, for an engine you run. `subs auth` for a deployment. |
| Signing secret | `signing_secret_env` on the agent, for an engine you run. The deployment creates its own. |
| Connector token | `subs auth <path>`. |
| Slack tokens | `$SLACK_APP_TOKEN` and `$SLACK_BOT_TOKEN`, or `subs slack connect`. |

`subs apply` strips `api_key_env` and `signing_secret_env` before it sends.

## Next

- [Agents](./30-agents.md): what an agent section declares.
- [Plugins](./45-plugins.md): what a `[plugin.<id>]` directory holds.
- [CLI](./260-cli.md): the commands that read this file.
- [Cloud](./170-cloud.md): applying it to a deployment.
- [Protocol](./230-protocol.md): the same types on the wire.
