# no-code-slack-bot-channels

One Slack bot, a different agent in each channel. There is no worker and no code.

A channel names an agent, not a prompt and not a tool list, because the agent
already is that bundle. Point a channel at another agent and its prompt, model,
and tools change together.

```toml
[slack]
dm = "support"                   # direct messages
mentions = "support"             # @mentions, in any channel not named below

[slack.channel.C01ABCD2EFG]      # #eng-oncall
agent = "oncall"

[slack.channel.C03HIJK4LMN]      # #random
off = true
```

Leave `mentions` out and the table becomes an allowlist. Invite the bot
anywhere, and it answers only in the channels the file names.

## Find a channel id

Name a channel by id, never by name. A rename moves a name to another channel.
The id does not move, and a `#name` here is a parse error.

An id starts with `C` and looks like `C01ABCD2EFG`. Three ways to read one:

- Open the channel, click its name, and scroll to the bottom of the **About**
  tab. The id is there with a button that copies it.
- Take the last segment of the channel link, `…/archives/C01ABCD2EFG`. **Copy
  link** on the channel in the sidebar gives you the same thing.
- Call [`conversations.list`](https://docs.slack.dev/reference/methods/conversations.list/)
  to read them all at once.

The engine checks every `agent` in this table against the file when it reads it,
so a typo fails at startup instead of becoming a bot that never answers.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Put your own ids in the two `[slack.channel.…]` sections. Then deploy the file
and give it a key:

```sh
subs login
subs apply
subs llm set-key openrouter
```

Install the bot into your workspace:

```sh
subs slack connect
```

Invite the bot to both channels. Mention it in each one, and read who answers.
