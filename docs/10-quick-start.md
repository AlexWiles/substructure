---
title: Quick start (cloud)
group: Getting started
---

Put an agent in Slack, on the hosted cloud. One file, five commands, no server
of your own.

To run the engine on your machine instead, with no account and no deployment,
start with the [local quick start](./15-quick-start-local.md).

## Install the CLI

```sh
npm install -g @substructure.ai/cli
subs login
```

`subs login` authenticates in your browser. It stores a token under
`~/.config/substructure`.

## Describe the agent

Write `substructure.toml` in your project root.

```toml title="substructure.toml"
name = "oncall-bot"

[llm.openrouter]
type = "openrouter"

[agent.oncall]
llm = "openrouter"
model = "anthropic/claude-sonnet-4-5"
system = "You are the on-call assistant."

[slack]
dm = "oncall"
mentions = "oncall"

[remote]
url = "https://api.substructure.ai"
```

`[llm.openrouter]` says which provider to call. `[agent.oncall]` says who calls
it. `[slack]` says where the agent answers. `[remote]` says the file describes a
deployment, so every command on this page acts on the cloud. The file holds no
keys.

## Create the project

```sh
subs apply
```

Apply creates the project and writes its ID back into the file. Run it again
after every change to the file.

## Add your LLM key

```sh
subs llm set-key openrouter
```

The command reads the key from stdin. Calls run on your key.

## Connect Slack

```sh
subs slack connect
```

This opens Slack's consent page. The token goes to the deployment.

## Talk to it

Mention the bot in a channel and it answers in the thread. Send it a DM and it
answers every message.

The thread is the session. Later mentions continue the conversation.

## What you have

An agent that answers in Slack, hosted. The engine calls the model, saves every
step, and streams the reply into the thread. Your machine runs nothing. You
wrote one file and no code.

Run `subs doctor` at any time to see what the project still needs.

## Next

Add one thing at a time.

- [Connectors](./40-connectors.md): give the agent the tools of Sentry, GitHub,
  or any MCP server.
- [Workers](./50-workers.md): run your own code at every step of the loop.
- [Quick start (local)](./15-quick-start-local.md): run the same file on your
  machine.
- [How it works](./20-how-it-works.md): the words the rest of the docs use.
