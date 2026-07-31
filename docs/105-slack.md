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
  agent_view: {}
oauth_config:
  scopes:
    bot:
      - im:history
      - app_mentions:read
      - channels:history
      - chat:write
      - assistant:write
      - reactions:write
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

And get the app token and a bot token. `agent_view` enables the Agents
feature (the **Agents** tab in app settings), which adds `assistant:write` —
what the bot needs to stream a turn's progress.

## Run

```sh
export SLACK_APP_TOKEN=xapp-...
export SLACK_BOT_TOKEN=xoxb-...
subs serve --no-auth --slack-agent my-agent
```

`--slack-agent` names the agent that handles mentions and enables the channel.
Pin it in [`substructure.toml`](./160-cli.md#environments) to leave it off the
command line:

```toml
[slack]
agent = "my-agent"
```

The tokens stay in the environment either way: the file names secrets, it never
holds them.

## Channels

`agent` is the default. Where one channel differs, name it:

```toml
[slack]
agent = "support"            # answers wherever nothing below applies

[slack.channel.C0ENGOPS]     # eng-oncall
agent = "oncall"

[slack.channel.C0RANDOM]     # random
off = true
```

A channel names an *agent*, not a prompt or a tool list, because
[`[agent.<id>]`](./160-cli.md#agentid) already is that bundle: pointing a
channel at another agent gives it another system prompt, model, and tool set
at once, and the tools are that agent's own rather than a request the model may
decline.

Leaving the default off makes the table an allowlist — the bot answers in the
channels named here and nowhere else:

```toml
[slack.channel.C0ENGOPS]
agent = "oncall"
```

A DM is a channel (`D…`) and resolves the same way, so an allowlist with no
default silences DMs too. Name a default to keep them.

Channels are keyed by **id**, never by name: a rename re-points a name, and the
id does not move. Get one from the channel's **About** tab, from the channel
link (`…/C0ENGOPS`), or from `conversations.list`. A `#name` here is a parse
error rather than a rule that matches nothing. Every `agent` is checked against
the file's `[agent.<id>]` sections when it is read, so a typo fails at startup
instead of as a bot that never answers.

## Mapping

| Slack | Engine |
| --- | --- |
| Thread (channel or DM) | Session `slack:{channel}:{thread_ts}` |
| Mention / DM message | One turn (`client.append`: the unseen thread delta) |
| Task card | A tool call or sub-agent run |
| Reply | The turn result, closing the turn's message |

A turn is one message: it opens on the first task card, its tool calls stream
in as more cards while it runs, and the result finalizes it. Until there is a
card to show, the thread carries Slack's own status indicator instead
([`assistant.threads.setStatus`](https://docs.slack.dev/reference/methods/assistant.threads.setStatus/)) —
`substructure.ai is thinking…`, chrome rather than a message, and the same
call in a DM as in a channel. It belongs to the turn, start to end: it goes
up when the turn starts and comes back down when the turn ends, so a message
still waiting its turn never claims the thread is working. Slack drops a
status after two minutes, so a turn still thinking has it re-set.
Arrival is the reaction's job instead: the message that asked carries a 👀
from the moment the bot takes it — so a mention queued behind a running turn
is acknowledged before it is answered — removed when its turn completes.
An interrupt closes the message early —
`Paused: {reason}`, or a button prompt (below), lands in the message with
the cards it interrupted, and work after the resume opens a fresh one. A
thread shows one open message at a time: a queued turn waits for the turn
before it to settle before it opens its own, rather than streaming beside
it. Every conversation is a thread: a top-level DM message (or unthreaded
mention) starts one at its own ts, and follow-ups inside the thread continue
that session.

## Activity

Each tool call and sub-agent run is a
[task card](https://docs.slack.dev/reference/block-kit/blocks/task-card-block/)
carrying its name, state and duration — collapsed by default, so a long run
stays one line until expanded. Cards are keyed by the engine's call id, so a
redelivered event sets a card rather than adding one. What the model says on
its way to a call streams as its own text ahead of that call's card; a
response with nothing left to call is the turn's answer, which the reply
carries already.

While the turn runs a card carries only its name, state and duration —
Slack caps a streaming task chunk at 256 characters. When the turn finishes
the message is rebuilt out of blocks, where a card's `details` and `output`
are rich text: each call arrives with its arguments and its result.

A message holds 50 blocks and 40,000 characters, and a card costs one block
plus whatever it carries. A turn bigger than that gives up its oldest calls
one at a time until the rest fit, and stands them on a single line —
`… 253 earlier steps`. Every card that survives keeps everything it carried.
Twenty-odd chatty calls exhaust the characters long before fifty calls
exhaust the blocks; the streamed message carried all of them either way,
only the rebuild has a budget to keep.

Cards are derived from the event log, never accumulated: each append folds
the turn's events and sends only what changed. Progress is best-effort — an
append Slack refuses abandons the stream, and the turn's reply posts as its
own message instead. The same fallback covers a restart mid-turn: a stream
carries no metadata until `chat.stopStream`, so an interrupted process can't
re-attach to one. The reply itself is unaffected, and still lands exactly
once.

Appends are paced at one per second across all sessions, so a burst of tool
calls coalesces into the next tick.

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
A prompt closes the turn's streaming message rather than holding a stream
open behind it: an approval may wait indefinitely, and a stream may not.
So the prompt sits under the cards that led to it, and the work that follows
the resume streams into a new message.
Once resumed — by a click, the API, or a timeout — the prompt loses its
buttons and shows the outcome, cards and all: the edit drops the buttons
and adds the outcome, and leaves every other block alone. Prompt posts are stamped with their
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
appear as `<@U…>: text` user messages; the bot's own unstamped posts (a
stream still in flight) are skipped, so progress never reads as
conversation. The bot stamps each reply's Slack
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

## Reuse: webhooks and multiple workspaces

The bot's behavior lives in `SlackBot`, resolved per workspace — Socket Mode
is one thin transport over it. An embedding crate can run the same bot over
the Events API instead: implement `WorkspaceResolver` (team → bot token,
tenant, agent — one tenant per install, since `slack:{channel}:{ts}` ids are
only unique per workspace), mount `webhook_router` (signature-verified
`/events` and `/interactions`, `url_verification` answered), and call
`SlackBot::start`. Behavior is identical by construction: both transports
parse deliveries down to the same payloads and hand them to the same bot.
Run exactly one `SlackBot` per process — the outbound processor keeps a
single named checkpoint.

## Next

- [Conversations](./70-conversations.md): the session behind a thread.
- [AG-UI](./100-ag-ui.md): the browser-chat channel.
