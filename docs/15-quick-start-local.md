---
title: Quick start (local)
group: Getting started
---

Run an agent on your machine. One file, two commands, no account.

## Install the CLI

```sh
curl -fsSL https://subs.dev/cli.sh | bash
```

The CLI is the engine too. Nothing on this page signs in or leaves this machine.

## Describe the agent

Write `subs.toml` in your project root.

```toml title="subs.toml"
name = "oncall-bot"

[llm.openrouter]
type = "openrouter"

[agent.oncall]
llm = "openrouter"
model = "anthropic/claude-sonnet-4-5"
system = "You are the on-call assistant."
```

`[llm.openrouter]` says which provider to call. `[agent.oncall]` says who calls
it. The file names no `[remote]`, so every command acts here.

## Give it a key

```sh
export OPENROUTER_API_KEY=sk-or-...
```

The engine takes the key from your environment. The file holds no keys — it
names variables at most, with `api_key_env`.

## Talk to it

```sh
subs run oncall "what is broken?"
```

The reply streams to your terminal as text. Piped into another program it
streams AG-UI protocol events instead; `-o` says which, whatever the venue.

## Continue the session

After every run the CLI prints the command to continue it: the one you typed,
with the session pinned and the message replaced by a placeholder at the end.

```
continue this session with:
  subs run oncall --session <session-id> '...'
```

```sh
subs run oncall --session <session-id> "what did I just ask?"
```

The agent remembers. The engine saves the whole session in
`~/.config/subs/subs.db`, so you can stop everything and pick
the session up tomorrow.

```sh
subs sessions list
subs sessions events <session-id> -o pretty
```

## What you have

An agent that runs on this machine, on your key, against a SQLite file that you
can delete. The engine calls the model, saves every step, and replays it on
demand. You wrote one file and no code.

`subs serve --no-auth` runs the same engine as an HTTP server, with the REST and
AG-UI endpoints a frontend needs.

Run `subs doctor` at any time to see what the project still needs.

## Next

Add one thing at a time.

- [Local development](./160-local-development.md): workers, Slack, MCP, and a
  second environment.
- [Workers](./50-workers.md): run your own code at every step of the loop.
- [Quick start (cloud)](./10-quick-start.md): the same file, hosted, in Slack.
- [How it works](./20-how-it-works.md): the words the rest of the docs use.
