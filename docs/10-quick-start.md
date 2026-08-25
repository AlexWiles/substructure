---
title: Quick start (cloud)
group: Getting started
---

Put an agent in Slack, hosted on substructure.ai. You write one config file and
run five commands. You do not run a server.

To run the engine on your own machine, with no account and no deployment, start
with the [local quick start](./15-quick-start-local.md).

## Install the CLI

```sh
curl -fsSL https://subs.dev/cli.sh | bash
subs login
```

The script verifies the release checksum and installs to `~/.local/bin`. Set
`SUBS_INSTALL_DIR` to install elsewhere and `SUBS_VERSION` to pin a release. To
install from npm instead: `npm install -g @substructure.ai/cli`.

`subs login` authenticates in your browser and stores a token under
`~/.config/subs`.

## Write the config file

Create `subs.toml` in your project root.

```toml title="subs.toml"
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

`[llm.openrouter]` sets which provider to call. `[agent.oncall]` sets who calls
it. `[slack]` sets where the agent answers. `[remote]` makes the file describe a
deployment, so every command on this page acts on the cloud. The file holds no
keys.

## Create the project

```sh
subs apply
```

Apply creates the project and writes its ID back into the file. Run it again
after every change to the file.

## Upload your LLM key

```sh
subs auth llm.openrouter
```

The command reads the key from stdin. Model calls run on your key.

## Connect Slack

```sh
subs slack connect
```

This opens Slack's consent page. The token goes to the deployment.

## Talk to the agent

Mention the bot in a channel and it answers in the thread. Send it a DM and it
answers every message.

The thread is the session. Later mentions continue the conversation.

## Check what is left to do

```sh
subs doctor
```

Doctor lists every setup step that nobody has finished, and the command that
finishes it.

## Next steps

Add one thing at a time.

- [Connectors](./40-connectors.md): give the agent the tools of Sentry, GitHub,
  or any MCP server.
- [Workers](./50-workers.md): run your own code at every step of the loop.
- [Quick start (local)](./15-quick-start-local.md): run the same file on your
  machine.
- [How it works](./20-how-it-works.md): the terms the rest of the docs use.
