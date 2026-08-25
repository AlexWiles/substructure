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
| Slack tokens | The environment. `$SLACK_APP_TOKEN` and `$SLACK_BOT_TOKEN`. |
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

A self-hosted engine talks to Slack over
[Socket Mode](https://docs.slack.dev/apis/events-api/using-socket-mode/), with a
Slack app you own. Socket Mode is an outbound WebSocket, so you need no public
URL.

Create a Slack app with this manifest.

```yaml
display_information:
  name: substructure.ai
features:
  bot_user:
    display_name: substructure.ai
    always_online: true
  agent_view: {}
  app_home:
    messages_tab_enabled: true
    messages_tab_read_only_enabled: false
oauth_config:
  scopes:
    bot:
      - im:history
      - app_mentions:read
      - channels:history
      - chat:write
      - assistant:write
      - files:read
      - files:write
  pkce_enabled: false
settings:
  event_subscriptions:
    bot_events:
      - app_mention
      - message.im
  interactivity:
    is_enabled: true
  org_deploy_enabled: false
  socket_mode_enabled: true
  token_rotation_enabled: false
  is_mcp_enabled: false
```

`agent_view` turns on the Agents tab in app settings. That adds
`assistant:write`, which the bot needs to stream a turn's progress.

`app_home` opens the Messages tab. Without it Slack says "Sending messages to
this app has been turned off" and a person cannot DM the bot, whatever the
scopes say.

Then get the app token and the bot token, and run the server.

```sh
export SLACK_APP_TOKEN=xapp-...
export SLACK_BOT_TOKEN=xoxb-...
subs serve --slack-agent my-agent
```

`--slack-agent` names the agent that answers DMs and any channel the file does
not name. Put the routing in the file instead.

```toml title="subs.toml"
[slack]
dm = "my-agent"
mentions = "my-agent"
```

See [Slack](./130-slack.md) for the routing rules.

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
