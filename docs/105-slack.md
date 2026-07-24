---
title: Slack
group: Frontends
---

The engine serves a Slack bot over [Socket
Mode](https://docs.slack.dev/apis/events-api/using-socket-mode/) — an outbound
WebSocket, no public URL. Mention the bot in a channel and it answers in the
thread — the thread is the session, so follow-up mentions continue the
conversation. DM the bot and every message is answered, no mention needed —
same model: a top-level DM message starts a thread, and that thread is the
session.

## Setup

Create a Slack app with a manifest like:

```yaml
display_information:
  name: substructure.ai
features:
  bot_user:
    display_name: substructure.ai
    always_online: true
oauth_config:
  scopes:
    bot:
      - im:history
      - app_mentions:read
      - channels:history
      - chat:write
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

And get the app token and a bot token.

## Run

```sh
export SLACK_APP_TOKEN=xapp-...
export SLACK_BOT_TOKEN=xoxb-...
subs serve --dev --worker-url http://localhost:4444 --slack-agent my-agent
```

`--slack-agent` names the agent that handles mentions and enables the channel.

## Mapping

| Slack | Engine |
| --- | --- |
| Thread (channel or DM) | Session `slack:{channel}:{thread_ts}` |
| Mention / DM message | One turn (`client.append`: the unseen thread delta) |
| Reply | The turn result, posted to the thread |

The bot posts once, when the turn completes — no token streaming. An
interrupt posts `Paused: {reason}`, or a button prompt (below). Every
conversation is a thread: a top-level DM message (or unthreaded mention)
starts one at its own ts, and follow-ups inside the thread continue that
session.

## Interrupt prompts

An interrupt whose payload carries a `message` renders it (with buttons)
instead of `Paused: {reason}` — how a worker asks a human something from
Slack (tool approval, a clarifying choice). The payload *is* an
[AG-UI Interrupt](https://docs.ag-ui.com/concepts/interrupts): spec fields
top-level (the categorical routing hint goes in the interrupt's `reason` —
`tool_call`, `input_required`, `confirmation`, or a namespaced custom), and
everything convention-specific under `metadata`, the spec's extension pocket:

```json
{
  "message": "Run `send_email`?",
  "toolCallId": "tc_1",
  "expiresAt": "2026-07-24T18:00:00Z",
  "metadata": {
    "options": [
      { "label": "Approve", "value": { "decision": "approve" }, "style": "primary" },
      { "label": "Deny", "value": { "decision": "deny" }, "style": "danger" }
    ]
  }
}
```

`message` is mrkdwn; `metadata.options` become buttons (`style` `primary` or
`danger`); `expiresAt` renders as a footnote. `metadata` is delivered to
AG-UI clients verbatim (so they see the options too) and is the place for
anything else the worker wants on record — e.g. a `pending` stash of the
held call. It is client-visible by definition: anything private belongs in
worker state, not the payload. An interrupt without a `message` posts
`Paused: {reason}`. A click resumes the interrupt with the AG-UI resume
shape plus a provenance stamp — the same shape AG-UI clients produce, so a
worker reads one resolution everywhere:

```json
{ "status": "resolved",
  "payload": { "decision": "approve" },
  "responder": { "channel": "slack", "user": "U…", "label": "Approve" } }
```

The inner `payload` is the chosen option's `value` verbatim (read from the
recorded interrupt, not the wire: a click can't smuggle a value). The worker
decides what it means; it should treat anything but an explicit expected
resolution as its safe default, since a resume can also arrive from the API
with any payload. Both shapes are typed in the protocol schema
(`InterruptPayload`, `InterruptResolution`), so generated worker types
include them.
Once resumed — by a click, the API, or a timeout — the prompt message loses
its buttons and shows the outcome. Prompt posts are stamped with their
interrupt id, so redeliveries dedupe and never join the transcript. Messages
sent while a prompt is pending are not lost: they ride the thread delta into
the next turn. See the
[`node-hono-tool-approval`](../examples/node-hono-tool-approval) example for
the worker side.

## Thread context

The session's recorded messages are the cursor: each mention or DM message
fetches only the thread past the highest recorded `slack:{ts}` id
(`conversations.replies` with `oldest`) and appends that delta
(`client.append`) — composed against the
session at delivery, so a message that arrives while a turn is running lands
after its reply instead of forking. Fetched messages record under
`slack:{ts}`, so redeliveries reconcile instead of duplicating. Users
appear as `<@U…>: text` user messages. The bot stamps each reply's Slack
metadata with the engine ids behind it (message, session, turn), so a fetch
maps its own replies back to their recorded assistant nodes — skipped when
already on the path, rebuilt in place when not: a lost database recovers the
conversation from its thread. Chatter between mentions reaches the agent at
the next mention. If the fetch fails (e.g. missing `channels:history` or
`im:history`), the message alone is appended with a note that context may be
missing.

Replies are durable: a checkpointed processor watches the event log and posts
completions, so a turn in flight across an engine restart still answers. An
event is acked only once its turn is recorded — Slack redelivers unacked
events, and the turn id dedupes the replay. Before posting, the processor
checks the thread for the turn's stamped reply, so a crash between post and
checkpoint doesn't answer twice.

## Next

- [Conversations](./70-conversations.md): the session behind a thread.
- [AG-UI](./100-ag-ui.md): the browser-chat channel.
