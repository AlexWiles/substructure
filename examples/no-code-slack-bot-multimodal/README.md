# no-code-slack-bot-multimodal

A multi-modal Slack agent with just a config file.

## Create the Slack app

Create a [Slack app](https://api.slack.com/apps) from this manifest.
```yaml
display_information:
  name: assistant
features:
  bot_user:
    display_name: assistant
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

Then collect the two tokens:

- **App token** (`xapp-…`): app settings → Basic Information → App-Level
  Tokens → Generate, with the `connections:write` scope.
- **Bot token** (`xoxb-…`): Install App → install to your workspace, then copy
  the Bot User OAuth Token.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

From this directory, with the tokens and an [OpenRouter](https://openrouter.ai)
key in the environment:

```sh
export SLACK_APP_TOKEN=xapp-...
export SLACK_BOT_TOKEN=xoxb-...
export OPENROUTER_API_KEY=sk-or-...

subs serve
```

Socket Mode connects outward, so no public URL is needed. DM the bot a
screenshot and ask about it, or mention it in a channel with an image attached.
A thread is a session: a later mention in the thread still sees the image.
