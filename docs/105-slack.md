---
title: Slack
group: Frontends
---

The engine serves a Slack bot over [Socket
Mode](https://docs.slack.dev/apis/events-api/using-socket-mode/). Socket Mode is
an outbound WebSocket, so you need no public URL.

Mention the bot in a channel and it answers in the thread. The thread is the
session, so later mentions continue the conversation. In a DM, the bot answers
every message, with no mention. A DM message at the top level starts a thread,
and that thread is the session.

## Connect a workspace

The bot runs where its token is, and the file says which.

If the file names a [`[remote]`](./170-cloud.md), the deployment installs its own
Slack app for you:

```sh
subs slack connect
```

That opens Slack's consent page. The token goes to the deployment, not to this
machine. The command waits for the workspace, then prints its name.

A Slack app installs once per workspace. Run the command again on a workspace
the org already has, and it refreshes that token and adds nothing. The command
says so instead of reporting a new workspace. A workspace belongs to one org,
and an org can have several.

The workspace is only one half. `[slack]` below routes a message to an agent,
and `subs apply` sends it to the deployment. A connected workspace with no
`[slack]` is a bot that never answers.

You can do the two steps in either order. Routing applied before the workspace
exists does nothing until the workspace arrives, and connecting is what starts
it. So `subs apply` and then `subs slack connect` is enough. You do not need a
second apply.

An engine you run yourself has no app to install for you. `subs serve` talks to
Slack over Socket Mode with a Slack app you own. So `subs slack connect` says so
and points to the steps below. A deployment that installs no Slack app says the
same.

## Setup

These are the Socket Mode steps, for an engine you run.

Create a Slack app with a manifest like this one:

```yaml
display_information:
  name: substructure.ai
features:
  bot_user:
    display_name: substructure.ai
    always_online: true
  agent_view: {}
oauth_config:
  scopes:
    bot:
      - im:history
      - app_mentions:read
      - channels:history
      - chat:write
      - assistant:write
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

Then get the app token and a bot token. `agent_view` turns on the Agents
feature, the **Agents** tab in app settings. That adds `assistant:write`, which
the bot needs to stream a turn's progress.

## Run

```sh
export SLACK_APP_TOKEN=xapp-...
export SLACK_BOT_TOKEN=xoxb-...
subs serve --no-auth --slack-agent my-agent
```

`--slack-agent` names the agent that answers DMs and any channel the file does
not name. Put it in [`substructure.toml`](./160-cli.md#environments) to leave it
off the command line:

```toml
[slack]
dm = "my-agent"
mentions = "my-agent"
```

The tokens stay in the environment either way. The file names a secret. It never
holds one.

## Where the bot answers

There are three separate settings, because they have different answers and reach
different people:

```toml
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

The two top-level keys are named after how a message reaches the bot. A channel
reaches it only by `@`-mention. A DM never needs one.

Each key defaults to off, so the bot answers only where you tell it to.

A channel names an *agent*. It does not name a prompt or a tool list, because
[`[agent.<id>]`](./160-cli.md#agentid) already holds those. Point a channel at
another agent, and you change its system prompt, its model, and its tools
together. Those tools belong to that agent. They are not a request the model can
refuse.

Leave `mentions` off and the table becomes an allowlist. You can invite the bot
anywhere, and it answers only in the channels named here:

```toml
[slack.channel.C0ENGOPS]
agent = "oncall"
```

DMs have their own setting. An allowlist does not stop them, and `dm` does not
open any channel. Nobody invites a bot to a DM, so the two are separate keys.

Name a channel by **id**, never by name. A rename moves a name to another
channel. The id does not move. Get the id from the channel's **About** tab, from
the channel link (`…/C0ENGOPS`), or from `conversations.list`. A `#name` here is
a parse error, not a rule that matches nothing. The engine checks every `agent`
against the file's `[agent.<id>]` sections when it reads the file, so a typo
fails at startup. It does not become a bot that never answers.

## Mapping

| Slack | Engine |
| --- | --- |
| A thread, in a channel or a DM | Session `slack:{channel}:{thread_ts}` |
| A mention or a DM message | One turn. `client.append` carries the part of the thread the session has not seen. |
| A task card | A tool call or a sub-agent run |
| A reply | The turn's result. It closes the turn's message. |

A turn is one Slack message. The message opens with the first task card. More
cards stream in as the turn makes tool calls. The result completes the message.

Before there is a card to show, the thread shows Slack's own status indicator
instead, through
[`assistant.threads.setStatus`](https://docs.slack.dev/reference/methods/assistant.threads.setStatus/).
It reads `substructure.ai is thinking…`. It is not a message, and the bot sets it
the same way in a DM and in a channel. It belongs to the turn: it goes up when
the turn starts and comes down when the turn ends. So a message that is still
waiting never says the thread is working. Slack removes a status after two
minutes, so the bot sets it again while the turn continues. A message queued
behind a running turn has no turn yet, so nothing marks it until the turn ahead
of it ends.

An interrupt closes the message early. `Paused: {reason}`, or a button prompt
described below, lands in the message with the cards it interrupted. Work after
the resume opens a new message. A thread shows one open message at a time. A
queued turn waits for the turn ahead of it to end before it opens its own
message. It does not stream beside it.

Every conversation is a thread. A DM message at the top level, or a mention
outside a thread, starts one at its own ts. Later messages in that thread
continue that session.

## Activity

Each tool call and sub-agent run is a
[task card](https://docs.slack.dev/reference/block-kit/blocks/task-card-block/).
A card shows its name, state, and duration. Cards are collapsed by default, so a
long run stays one line until someone expands it. Each card is keyed by the
engine's call id, so a repeated event updates a card instead of adding one. What
the model says before a call streams as text above that call's card. A response
with no call left is the turn's answer, and the reply already carries it.

While the turn runs, a card shows only its name, state, and duration, because
Slack limits a streaming task chunk to 256 characters. When the turn finishes,
the bot rebuilds the message from blocks. There, a card's `details` and `output`
are rich text, so each call shows its arguments and its result.

A Slack message holds 50 blocks and 40,000 characters. A card costs one block
plus what it carries. A turn larger than that drops its oldest calls, one at a
time, until the rest fit. It puts the dropped calls on one line, such as
`… 253 earlier steps`. Every card that stays keeps everything it carried. About
twenty long calls use up the characters well before fifty calls use up the
blocks. The streamed message carried all of them. Only the rebuilt message has a
limit.

Cards go through the decision loop. On each decision the engine proposes the
message view, derived from the turn's event log. A repeated event derives the
same view. The view from the completed decision is what streams. The bot compares
that view against what it already sent and appends only the changes. Progress is
best effort. If Slack refuses an append, the bot stops streaming and posts the
turn's reply as its own message. After a restart, the bot rebuilds the view from
the log and continues. None of this changes the reply, which still arrives
exactly once.

The bot appends at most once per second across all sessions, so a burst of tool
calls goes out together on the next tick.

## Interrupt prompts

If an interrupt's payload has a `message`, the bot shows that message with
buttons instead of `Paused: {reason}`. This is how a worker asks a person
something from Slack, such as a tool approval or a choice.

The payload is an [AG-UI Interrupt](https://docs.ag-ui.com/concepts/interrupts).
Put the spec's fields at the top level. Put the routing hint in the interrupt's
`reason`: `tool_call`, `input_required`, `confirmation`, or your own namespaced
value. Put everything else under `metadata`, the spec's extension field:

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

`message` is mrkdwn. Each entry in `metadata.options` becomes a button, with
`style` set to `primary` or `danger`. `expiresAt` shows as a footnote. The engine
delivers `metadata` to AG-UI clients unchanged, so they see the options too. Use
it for anything else the worker wants on record, such as a `pending` copy of the
held call. Every client can read it, so keep anything private in worker state,
not in the payload. An interrupt with no `message` posts `Paused: {reason}`.

A click resumes the interrupt with the AG-UI resume shape and a record of who
clicked. AG-UI clients produce the same shape, so a worker reads one shape
everywhere:

```json
{ "status": "resolved",
  "payload": { "decision": "approve" },
  "responder": { "channel": "slack", "user": "U…", "label": "Approve" } }
```

The inner `payload` is the chosen option's `value`, unchanged. The engine reads
it from the recorded interrupt, not from the wire, so a click cannot send its own
value. The worker decides what it means. A resume can also arrive from the API
with any payload, so treat anything other than a resolution you expect as your
safe default. The protocol schema types both shapes, as `InterruptPayload` and
`InterruptResolution`, so generated worker types include them.

Every click is a `client.action` decision. The action's args carry the click
unchanged: `action_id`, `value`, the user, and where the message is. For a click
on a prompt button, the engine proposes the resolution above as an
`interrupt.resolve` action that carries the recorded option value. So a worker
that returns the proposal, and an engine-hosted agent, behave as described with
no code. A worker that wants its own rules answers the decision itself. It can
check who clicked, refuse the resolve, or do something else.

A click on a prompt that is no longer open proposes a message update that marks
it `(no longer active)`. A click on a button the worker created gets no proposal,
because only the worker knows what that button means. The worker's answer can
start work, which opens a turn, update the message, or only write state. Slack
sends interaction payloads again, so the worker records the clicks it has handled
in its own state.

A prompt closes the turn's streaming message instead of holding the stream open.
An approval can wait for a long time, and a stream cannot. So the prompt sits
under the cards that led to it, and the work after the resume streams into a new
message.

When someone resumes the interrupt, by a click, through the API, or by a timeout,
the prompt loses its buttons and shows the result. The edit removes the buttons
and adds the result. It leaves every other block, including the cards. Each
prompt post carries its interrupt id, so a repeated delivery does not post twice
and never joins the transcript. Messages sent while a prompt is open are not
lost. They arrive with the thread delta on the next turn. For the worker side,
see the [`node-hono-tool-approval`](../examples/node-hono-tool-approval) example.

## Customizing what the bot shows

Every Slack message is written through the decision loop. A decision response can
carry a `channels.slack` value. Without it, the bot shows its default, which is
everything above. With it, the value wins:

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

- `status` sets the thread's working indicator.
- `view` is the whole state you want for the turn's message. During the turn it
  streams: task cards are set by `task_id`, and other blocks stream as text. On
  `turn.finished` it becomes the final message. The thread keeps the last view
  before the turn ends.
- `prompt` is the message an interrupt posts, when the same decision raises one.
  It sets the buttons and the layout. Without it, the bot shows the interrupt
  payload in the default shape.
- `update` rewrites one message. Use it to clear an old prompt, or to edit a
  button's message in place.

On every decision of a Slack session, the engine proposes the default `view`: the
cards, the answer, and on `turn.finished` how long the turn took. So a worker
that returns the proposal posts the default, and a worker that customizes changes
the proposed view instead of building a new one. The engine never reads these
values. Only the Slack transport reads this key.

Blocks with buttons pass through unchanged. Give each button an `action_id` and a
`value`, and the click comes back as a `client.action` decision with both. The
prompt buttons the bot draws itself carry an `interrupt_id` and an `option`. A
worker that replaces a prompt's blocks must keep them, or the click cannot find
its interrupt.

This is how you build a custom bot on top of the default behavior. Add "skills"
buttons to a `turn.finished` view. When someone clicks one, days later on an idle
session, answer the `client.action` with a transcript note and an `llm.call`. The
work opens a new turn, streams, and ends with new buttons.

## Pointing a session at a thread

A session belongs to Slack because of the address on its owner: the
`slack_channel` and `slack_thread_ts` owner metadata. The shape of the session id
does not matter. The bot writes both on every session it creates. An API client
can set the same metadata on its first submit, and that session's turns then go
into the thread like any other. Each reply the bot posts carries its session id,
so a click on that message goes back to the session that posted it, not to the
thread's own session.

## Thread context

The session's recorded messages are the cursor. For each mention or DM message,
the bot fetches only the part of the thread after the highest recorded
`slack:{ts}` id, using `conversations.replies` with `oldest`. It appends that
part with `client.append`. The engine composes the append against the session
when it is delivered, so a message that arrives during a turn lands after that
turn's reply. It does not branch.

The bot records fetched messages under `slack:{ts}`, so a repeated delivery
matches the existing message instead of adding one. Users appear as `<@U…>: text`
user messages. The bot skips its own posts that carry no stamp, which are streams
still in flight, so progress never reads as conversation.

The bot writes the engine ids on each reply's Slack metadata: the message, the
session, and the turn. So a fetch maps its own replies back to the recorded
assistant messages. It skips a reply already on the path and rebuilds one that is
not. This means a lost database can recover the conversation from its thread.
Messages between mentions reach the agent at the next mention. If the fetch fails,
because `channels:history` or `im:history` is missing for example, the bot
appends the message alone with a note that context may be missing.

Replies survive a restart. A processor with a checkpoint watches the event log
and posts each completion, so a turn that was running when the engine restarted
still answers. The bot acks an event only after its turn is recorded. Slack sends
unacked events again, and the turn id removes the duplicate. Before it posts, the
processor checks the thread for the turn's stamped reply, so a crash between the
post and the checkpoint does not answer twice.

## Reuse: webhooks and several workspaces

The bot's behavior is in `SlackBot`, resolved for each workspace. Socket Mode is
a thin transport over it. A crate that embeds the engine can run the same bot
over the Events API instead. Do three things:

1. Implement `WorkspaceResolver`, which maps a team to a bot token, a tenant, and
   an agent. Use one tenant per install, because `slack:{channel}:{ts}` ids are
   unique only within a workspace.
2. Mount `webhook_router`. It verifies signatures on `/events` and
   `/interactions`, and answers `url_verification`.
3. Call `SlackBot::start`.

Both transports parse their deliveries into the same payloads and give them to
the same bot, so the behavior is the same. Run one `SlackBot` per process,
because the outbound processor keeps one named checkpoint.

## Next

- [Conversations](./70-conversations.md): the session behind a thread.
- [AG-UI](./100-ag-ui.md): the browser-chat channel.
