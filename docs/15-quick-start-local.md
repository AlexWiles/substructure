---
title: Quick start (local)
group: Getting started
---

Run an agent on your machine. You write one config file and run two commands.
You do not need an account.

## Install the CLI

```sh
curl -fsSL https://subs.dev/cli.sh | bash
```

The CLI is also the engine. Nothing on this page signs in or leaves this
machine.

## Run an agent

Create a `subs.toml`.

```toml title="subs.toml"
name = "example"

[llm.openrouter]
type = "openrouter"

[agent.teammate]
llm = "openrouter"
model = "deepseek/deepseek-v4-flash-0731"
system = "You are a helpful teammate."
```

`[llm.openrouter]` sets which provider to call. `[agent.teammate]` sets who
calls it. The file names no `[remote]`, so every command acts on this machine.

Set your provider key.

```sh
export OPENROUTER_API_KEY=sk-or-...
```

The engine reads the key from your environment. The file holds no keys. It can
name a variable with `api_key_env`.

Talk to the agent.

```sh
subs chat teammate -c subs.toml
```

Chat holds one session open and streams the reply as it is written, so the next
question can depend on the last one. `Ctrl-D` ends the chat and prints the
command to pick the session back up.

```text
continue this session with:
  subs chat teammate --session 01a02417-7d46-7441-8090-23b20d0f980f
```

The engine saves the whole session in `~/.config/subs/subs.db`, so you can stop
everything and continue the session tomorrow.

```sh
subs sessions list
subs sessions events <session-id> -o pretty
```

To send one message and exit, use `subs run teammate "what is broken?"`. At a
terminal it prints text. Piped into another program it streams AG-UI protocol
events. `-o` overrides the choice.

## Run the engine as a server

Add these sections to `subs.toml`.

```toml title="subs.toml"
[serve]
port = 9999
auth = false

[remote]
url = "http://localhost:9999"
```

Start the server.

```sh
subs serve -c subs.toml
```

In another terminal, the same chat command now talks to it.

```sh
subs chat teammate -c subs.toml
```

`serve` runs the same engine as an HTTP server, with the REST and AG-UI
endpoints a frontend needs. `[remote]` points the client at it, so the turn runs
in the server process and the session lands in its database.

## Check what is left to do

```sh
subs doctor
```

Doctor lists the variables that are empty and the connections that hold no
credential.

## Next steps

Add one thing at a time.

- [Local development](./160-local-development.md): workers, Slack, MCP, and a
  second environment.
- [Workers](./50-workers.md): run your own code at every step of the loop.
- [Quick start (cloud)](./10-quick-start.md): the same file, hosted, in Slack.
- [How it works](./20-how-it-works.md): the terms the rest of the docs use.
