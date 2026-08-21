---
title: Local development
group: Running it
---

Run the engine on your machine. Iterate on an agent before it goes live. The
[local quick start](./15-quick-start-local.md) is the five-minute version of
the first two sections.

Everything works locally: Slack, MCP connections, workers, and sub-agents. The
engine keeps its state in a SQLite file beside your config.

## Set up a file

Write `substructure.toml` in your project root.

```toml title="substructure.toml"
name = "oncall-bot"

[llm.openrouter]
type = "openrouter"

[agent.oncall]
llm = "openrouter"
model = "anthropic/claude-sonnet-4-5"
system = "You are the on-call assistant."

[serve]
port = 8080
```

`[serve]` runs the engine here. Name a `[remote]` instead to point the CLI at a
deployment. See [Config](./220-config.md) for every section.

## Run one turn

```sh
export OPENROUTER_API_KEY=sk-or-...
subs run --agent oncall -o pretty "hi"
```

The reply streams to your terminal. `-o pretty` shows the turn as text. The
default, `ag-ui`, streams protocol events.

Put the flags in the file so you stop repeating them.

```toml title="substructure.toml"
[run]
agent = "oncall"
output = "pretty"
```

```sh
subs run "hi"
```

## Continue a session

After every run the CLI prints the command to continue it: the one you typed,
with the session pinned and the message replaced by a placeholder at the end.

```
continue this session with:
  subs run --session <session-id> '...'
```

```sh
subs run --session <session-id> "what was my first question?"
```

The agent remembers. The engine saves the whole session in `substructure.db`.
Stop everything, come back tomorrow, and the session continues.

## Read what happened

```sh
subs sessions list
subs sessions events <session-id>
```

These read `substructure.db` directly, so they work with nothing running. A
file naming a `[remote]` asks the deployment instead; `--db <path>` reads a
file whatever the config says.

`-o pretty` replays the session as text instead of printing its events.

```sh
subs sessions events <session-id> -o pretty
```

With `subs serve` running beside it, `--stream` follows the file live.

```sh
subs sessions events <session-id> --stream
```

## Run a server

```sh
subs serve --no-auth
```

This serves the [REST API](./250-api.md) and the [AG-UI](./140-ag-ui.md)
endpoints on `127.0.0.1:8080`. Point a frontend at it.

`--no-auth` turns off client and worker authentication. Use it only for a server
that nothing off this machine can reach.

## Develop a worker

Run your worker and point an agent at it.

```toml title="substructure.toml"
[agent.oncall]
llm = "openrouter"
model = "anthropic/claude-sonnet-4-5"
worker = "http://localhost:4444"
```

```sh
node server.mjs
subs run "hi"
```

Every decision now POSTs to your code. See [Workers](./50-workers.md).

A local engine signs decisions only when the agent names
`signing_secret_env`. Leave it off while you develop.

## Connect Slack locally

A local engine uses a Slack app you own, over Socket Mode. See
[Self-hosting](./180-self-hosting.md#slack).

```sh
export SLACK_APP_TOKEN=xapp-...
export SLACK_BOT_TOKEN=xoxb-...
subs serve --no-auth --slack-agent my-agent
```

## Connect MCP servers locally

```sh
subs mcp login sentry
```

The engine on this machine runs the OAuth flow. The credential goes into that
environment's `db`.

**That database now holds credentials. Add `*.db*` to `.gitignore`.**

## Two environments

A second environment is a second file. Each one gets its own database.

```sh
subs run -c substructure.dev.toml "hi"
subs serve -c substructure.dev.toml
```

`db` defaults to the file's name. `substructure.toml` uses `substructure.db`.
`substructure.dev.toml` uses `substructure.dev.db`.

## Develop against a cloud project

One file can do both. Keep the engine keys and a `[remote]` section together.

```sh
subs serve                       # run it here
subs apply                       # deploy the same declaration
```

`subs serve` is the engine, so it runs here whatever the file names. Every other
command follows the `[remote]`: with one, `subs run` sends the turn to the
deployment. Keep a second file with no `[remote]` to run turns here.

`subs run` and `subs serve` read `api_key_env` and `signing_secret_env`. `subs
apply` strips them.

## Logs

```toml title="substructure.toml"
log = "info"
```

`log` takes `RUST_LOG` syntax: a level on its own, or per-target directives such
as `substructure_core=debug,warn`. `$RUST_LOG` wins over it. Without it, `subs
run` shows errors and `subs serve` shows info.

## Next

- [Workers](./50-workers.md): the code that the engine calls.
- [CLI](./260-cli.md): every command and flag.
- [Self-hosting](./180-self-hosting.md): run the engine for other people.
- [Cloud](./170-cloud.md): deploy the same file.
