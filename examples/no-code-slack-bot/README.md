# no-code-slack-bot

A Slack bot that is one file. There is no worker and no code.

The bot answers DMs and mentions. A thread is a session, so the next mention in
that thread continues the conversation.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Deploy the file and give it a key:

```sh
subs login
subs apply
subs llm set-key openrouter
```

Install the bot into your workspace:

```sh
subs slack connect
```

That opens Slack's consent page. The token goes to the deployment, not to this
machine. Now mention the bot in a channel, or send it a DM.

## Where it answers

Each key is a way a message reaches the bot. A channel reaches it only by
mention. A DM never needs one. Each key defaults to silence, so the bot answers
only where you tell it to:

```toml
[slack]
dm = "oncall"              # direct messages
mentions = "oncall"        # @mentions, in any channel
```

To give one channel its own agent, see
[no-code-slack-bot-channels](../no-code-slack-bot-channels).
