---
title: Self-hosting
group: Running it
---

The engine's source is on GitHub. Run it yourself and you hold every credential.

A self-hosted engine serves the same APIs as the cloud. The same `subs.toml`
describes both.

## Run the server

```sh
subs serve --host 0.0.0.0 --port 8080
```

Or put it in the file.

```toml title="subs.toml"
[serve]
host = "0.0.0.0"
port = 8080
auth = true
```

At startup the engine logs each declared agent and whether the engine or a
worker decides for it.

## Where each credential goes

| Credential | Where it goes |
| --- | --- |
| Provider keys | The environment. `api_key_env` on each `[llm.<id>]` block names the variable. |
| Signing secrets | The environment. `signing_secret_env` on each `[agent.<id>]` names the variable. |
| Connector credentials | The `db` file, written by `subs auth`. |
| Slack tokens | The environment. `$SLACK_APP_TOKEN_<AGENT>` and `$SLACK_BOT_TOKEN_<AGENT>`. |
| Client token secret | `$CLIENT_TOKEN_HS256_SECRET`. |

## Storage

`db` names the SQLite file the engine writes to.

```toml title="subs.toml"
db = "/var/lib/substructure/engine.db"
```

The database holds the event log, the sessions, the connector credentials, and
the attachment bytes. Back it up. See [Durability](./200-durability.md).

## Authentication

`auth = true` is the default. The engine then requires a client JWT on
`/api/client` and an API key on `/api/machine`.

Set `$CLIENT_TOKEN_HS256_SECRET` to the secret your backend signs client tokens
with. See [Authentication](./190-auth.md).

`--no-auth` turns both off. Use it only for a server nothing off the machine can
reach.

## Worker signing

Name the variable holding each agent's signing secret.

```toml title="subs.toml"
[agent.triage]
llm = "claude"
model = "claude-sonnet-4-5"
worker = "https://triage.internal/agent"
signing_secret_env = "TRIAGE_SIGNING_SECRET"
```

An agent that names no variable gets unsigned requests. Set the same secret
where the worker runs. See [Workers](./50-workers.md#verify-the-signature).

## Let people authorize MCP connections from a link

Set `public_url` to the HTTPS address a browser reaches this engine at. The
engine then mints authorize URLs and hosts the OAuth callback, so a prompt that
asks someone to connect a service carries a link they can click.

```toml title="subs.toml"
[serve]
host = "0.0.0.0"
port = 8080
public_url = "https://engine.example.com"
```

Without it, an operator has to run `subs auth <path>` on the machine where the
engine runs. See [Connectors](./40-connectors.md).

## Slack

An engine you run talks to Slack over
[Socket Mode](https://docs.slack.dev/apis/events-api/using-socket-mode/), which
is an outbound WebSocket. You need no public URL and no inbound firewall rule.

Declare an app for each agent that should have one.

```toml title="subs.toml"
[agent.support.slack]
name = "Support"
```

Then ask for the app.

```sh
subs auth agent.support.slack
```

It prints a manifest for that agent. Paste it into
[api.slack.com/apps](https://api.slack.com/apps), under Create New App, From a
manifest. The manifest matches what the agent declares, so an agent set to
`answers = "dm"` asks Slack for less than one that also answers in channels.

Install the app, then set the two variables the command names.

```sh
export SLACK_APP_TOKEN_SUPPORT=xapp-...
export SLACK_BOT_TOKEN_SUPPORT=xoxb-...
subs serve
```

The app token is under Basic Information, App-Level Tokens. Create one with the
`connections:write` scope. The bot token is under OAuth & Permissions.

The variables are named after the agent. Uppercase the agent ID and replace
anything that is not a letter or a digit with an underscore.

`subs serve` opens one connection per declared app and names any variable it
cannot find. `subs doctor` lists them without starting the server.

Repeat for each agent. Two agents need two Slack apps and two pairs of
variables.

See [Slack](./130-slack.md) for where the bot answers and what it can do.

## Administer a self-hosted deployment

`[remote]` names any server that speaks `/api/v1`. Point it at your own.

```toml title="subs.prod.toml"
[remote]
url = "https://engine.internal"
```

```sh
subs login -c subs.prod.toml
subs apply -c subs.prod.toml
```

The CLI stores credentials per server, so you can be logged in to your
deployment and the hosted cloud at once.

## Embed the engine

A Rust crate can embed the engine and drive it directly. A worker then becomes a
callback instead of an HTTP endpoint. See the crate docs in the repository.

## Next steps

- [Config](./220-config.md): every key the file holds.
- [Authentication](./190-auth.md): callers, tokens, and identity.
- [Durability](./200-durability.md): what the store holds.
- [REST API](./250-api.md): what the server exposes.
