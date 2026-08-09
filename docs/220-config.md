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
mcp = ["sentry"]
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

What the project is stays the same for both roles: `name`, `[llm.<id>]`,
`[agent.<id>]`, `[mcp.<id>]`, and `[slack]`.

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

## `[llm.<id>]`

Where a model call runs. An agent names a block by its id.

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

There is no default block and no fallback. An agent names a block, or its calls
fail. See [LLMs](./70-llms.md).

`api_key_env` names a variable on your machine. `subs apply` removes it. A
deployment refuses a document that carries one.

## `[agent.<id>]`

An agent. The id is what clients, channels, and parent agents route on.

```toml
[agent.support]
llm = "claude"
model = "claude-sonnet-4-5"
system = "You are a support agent."
mcp = ["sentry"]
sub_agents = ["researcher"]
tools = [{ name = "confirm", description = "Ask the human", handler = "client" }]
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
| `description` | string | What this agent does, shown to a parent that calls it. |
| `mcp` | list | Connections. An id, or a table to take fewer tools, to put them behind a search, or to go on without one that needs authorizing: `{ id, tools, auth_failure }`. `tools` sets the filter and `discovery`; see [Tool discovery](./40-connectors.md#tool-discovery). `auth_failure` is `interrupt` (the default) or `degrade`; see [Connectors](./40-connectors.md#when-a-credential-stops-working). |
| `tool_discovery` | string | The default for each connection: `all` (the default) or `search`. A connection overrides it in `tools.discovery`. See [Tool discovery](./40-connectors.md#tool-discovery). |
| `sub_agents` | list of ids | Agents this one can call. |
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
| `prefix_tools` | bool | `true` | Show the model `<id>__<tool>` instead of the connection's own names. |

A `token` written in the file is a parse error. Fill a connection with `subs mcp
login <id>` or `subs mcp set-token <id>`. See [Connectors](./40-connectors.md).

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
| `dm` | agent id | Answers direct messages. |
| `mentions` | agent id | Answers mentions in any channel `channel` does not name. |
| `channel.<id>.agent` | agent id | Answers in that channel. |
| `channel.<id>.off` | bool | The bot stays out of that channel. |

Name a channel by id, never by name. A `#name` is a parse error. See
[Slack](./130-slack.md).

## `[run]`

Defaults for `subs run`.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `agent` | agent id | none | Which agent a bare `subs run` uses. |
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
| `org` | id | none | Written by `subs link` or `subs apply`. |
| `project` | id | none | Written by `subs apply` when it creates the project. |

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
| Provider key | `api_key_env` on the LLM block, for an engine you run. `subs llm set-key` for a deployment. |
| Signing secret | `signing_secret_env` on the agent, for an engine you run. The deployment creates its own. |
| Connector token | `subs mcp login`, or `subs mcp set-token`. |
| Slack tokens | `$SLACK_APP_TOKEN` and `$SLACK_BOT_TOKEN`, or `subs slack connect`. |

`subs apply` strips `api_key_env` and `signing_secret_env` before it sends.

## Next

- [Agents](./30-agents.md): what an agent section declares.
- [CLI](./260-cli.md): the commands that read this file.
- [Cloud](./170-cloud.md): applying it to a deployment.
- [Protocol](./230-protocol.md): the same types on the wire.
