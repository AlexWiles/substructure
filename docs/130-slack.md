---
title: Slack
group: Frontends
---

Mention the bot in a channel and it answers in the thread. DM it and it answers
every message.

The thread is the session. Later mentions continue the conversation.

## Set it up

Two steps. Say who answers, then connect the workspace.

```toml title="substructure.toml"
[slack]
dm = "support"
mentions = "support"
```

```sh
subs apply
subs slack connect
```

`subs slack connect` opens Slack's consent page and installs the app for you.
The token goes to the deployment. The command waits for the workspace, then
prints its name.

You need both steps. A connected workspace with no `[slack]` is a bot that never
answers.

A Slack app installs once per workspace. Run the command again and it refreshes
the token.

An engine you run needs a Slack app you own. See
[Self-hosting](./180-self-hosting.md#slack).

## Where the bot answers

```toml title="substructure.toml"
[slack]
dm = "support"               # direct messages
mentions = "support"         # @mentions, in any channel not named below

[slack.channel.C0ENGOPS]     # eng-oncall
agent = "oncall"

[slack.channel.C0RANDOM]     # random
off = true
```

| Where | Answered by |
| --- | --- |
| A DM | `dm` |
| A channel with a `[slack.channel.<id>]` | its own `agent`, or nobody when `off` |
| A mention in any other channel | `mentions` |

Each key defaults to off. The bot answers only where you tell it to.

Leave `mentions` off and the channel table becomes an allowlist. Invite the bot
anywhere and it answers only in the channels you named.

DMs have their own key. An allowlist does not stop them.

A channel names an agent, not a prompt or a tool list.
[`[agent.<id>]`](./30-agents.md) already holds those. Point a channel at another
agent and you change its prompt, its model, and its tools together.

Name a channel by **id**, never by name. Get the id from the channel's About
tab, from the channel link (`…/C0ENGOPS`), or from `conversations.list`. A
`#name` is a parse error. The engine checks every `agent` against the file's
sections at startup.

## How Slack maps to the engine

| Slack | Engine |
| --- | --- |
| A thread, in a channel or a DM | Session `slack:{channel}:{thread_ts}` |
| A mention or a DM message | One turn |
| A task card | A tool call or a sub-agent run |
| A reply | The turn's result |

A turn is one Slack message. The message opens with the first task card. More
cards stream in as the turn makes tool calls. The result completes the message.

A thread shows one open message at a time. A queued turn waits for the turn
ahead of it.

Each tool call and sub-agent run shows as a
[task card](https://docs.slack.dev/reference/block-kit/blocks/task-card-block/).
Cards are collapsed by default. Once the turn finishes, each card shows its
arguments and its result.

## Interrupt prompts

An interrupt with a `message` in its payload posts that message with buttons.
This is how a worker asks a person something from Slack.

The payload is an [AG-UI Interrupt](https://docs.ag-ui.com/concepts/interrupts).
Put the spec's fields at the top level. Put everything else under `metadata`.

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

`message` is mrkdwn. Each entry in `metadata.options` becomes a button.
`expiresAt` shows as a footnote. An interrupt with no `message` posts
`Paused: {reason}`.

The engine delivers `metadata` to AG-UI clients unchanged. Every client can read
it, so keep anything private in worker state.

Put the routing hint in the interrupt's `reason`: `tool_call`,
`input_required`, `confirmation`, or your own namespaced value.

### What a click does

A click resumes the interrupt with the AG-UI resume shape.

```json
{ "status": "resolved",
  "payload": { "decision": "approve" },
  "responder": { "channel": "slack", "user": "U…", "label": "Approve" } }
```

The inner `payload` is the chosen option's `value`. The engine reads it from the
recorded interrupt, so a click cannot send its own value.

Every click is a `client.action` decision. For a prompt button, the engine
proposes the resolution above as an `interrupt.resolve` action. A worker that
returns the proposal needs no code.

A worker that wants its own rules answers the decision itself. It can check who
clicked or refuse the resolve. Slack delivers a click more than once, so record
the clicks you have handled in worker state.

When someone resumes the interrupt, the prompt loses its buttons and shows the
result. Work after the resume goes into a new message.

For the worker side, see the
[`node-hono-tool-approval`](../examples/node-hono-tool-approval) example.

## Customizing what the bot shows

Every Slack message goes through the decision loop. A decision response can
carry a `channels.slack` value. Without it, the bot shows its default. With it,
your value wins.

```json
{
  "messages": [ ... ],
  "actions": [ ... ],
  "channels": {
    "slack": {
      "status": "is researching…",
      "view": { "text": "fallback text", "blocks": [ ... ] },
      "update": { "channel": "C…", "ts": "123.456", "text": "…", "blocks": [ ... ] }
    }
  }
}
```

| Key | Sets |
| --- | --- |
| `status` | The thread's working indicator. |
| `view` | The whole state of the turn's message. It streams during the turn and becomes the final message. |
| `prompt` | The message an interrupt posts, when the same decision raises one. |
| `update` | Rewrites one message, by channel and ts. |

On every decision the engine proposes the default `view`. A worker that
customizes changes the proposed view instead of building a new one.

Blocks with buttons pass through unchanged. Give each button an `action_id` and
a `value`, and the click comes back as a `client.action` decision with both.

A worker that replaces a prompt's blocks must keep the `interrupt_id` and
`option` on its buttons, or the click cannot find its interrupt.

This is how you build a custom bot on the default behavior. Add buttons to a
`turn.finished` view. When someone clicks one, days later, answer the
`client.action` with a note and an `llm.call`. The work opens a new turn.

## Thread context

The bot reads what it missed. Messages sent between mentions reach the agent at
the next mention. Users appear as `<@U…>: text`.

The bot needs `channels:history` and `im:history` to do this. Without them it
appends the new message alone, with a note that context may be missing.

## Pointing a session at a thread

A session belongs to Slack because of the `slack_channel` and `slack_thread_ts`
owner metadata. The shape of the session id does not matter.

An API client can set the same metadata on its first submit. That session's
turns then go into the thread like any other.

A click on a reply goes back to the session that posted it. Each reply carries
its session id.

## Next

- [Interrupts](./100-interrupts.md): the pause behind a prompt.
- [Conversations](./120-conversations.md): the session behind a thread.
- [Self-hosting](./180-self-hosting.md#slack): run the bot with your own Slack app.
