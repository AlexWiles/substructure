# node-hono-slack-skills

A Slack bot whose every reply ends with a row of skill buttons. Click one and
the thread gains that [skill](https://agentskills.io) — its instructions and its
tools — and the bot answers again with it.

The bot's messages are the engine's own. The worker takes the proposed Slack
view and appends its buttons, so the task cards, the answer, and the timing all
stay.

A skill's instructions go on the end of the transcript, not into the system
prompt. The prompt stays byte-identical for the life of the thread, so turning a
skill on adds to the cached prefix instead of breaking it.

```
skills/
├── changelog-entry/
│   └── SKILL.md      # instructions only
├── commit-messages/
│   └── SKILL.md
└── unit-conversion/
    ├── SKILL.md
    └── tools.mjs     # the function it unlocks
```

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker**:

```sh
npm install
node server.mjs
```

**2. Serve the bot.** You need a Slack app you own, with Interactivity on — see
[Self-hosting](../../docs/180-self-hosting.md#slack) for its manifest.

```sh
export SLACK_APP_TOKEN=xapp-... SLACK_BOT_TOKEN=xoxb-... OPENROUTER_API_KEY=sk-or-...
subs serve -c substructure.toml
```

DM the bot, or mention it in a channel. It answers in the thread, and the reply
carries **changelog-entry · commit-messages · unit-conversion**. Ask it to
convert 42 km to miles, then click **unit-conversion**: the answer comes back
with the `convert_units` tool card, and the button now reads **✓
unit-conversion**. Every later turn in that thread keeps the skill.

## How it works

| Decision | What the worker does |
| --- | --- |
| `session.start` | Sets the agent, with the tools the thread's skills unlock. |
| `turn.finished` | Appends an `actions` block to the proposed `channels.slack.view`. |
| `client.action` | A click. Appends the skill, rewrites the clicked message, and opens a turn with `llm.call`. |
| `tool.execute` | Runs a tool an active skill unlocked. |

A click is a `client.action` named after the button's `action_id`, so
`skill:unit-conversion` says which button it was. The click's `args` carry the
message it came from, which the worker rewrites through `channels.slack.update`
to move the check mark.

Nothing about a click expires. Click a skill on a week-old reply and the thread
picks it up and answers again.

## Add a skill

Drop a `skills/<name>/SKILL.md` with `name` + `description` frontmatter, and the
button appears. For tools, add a `tools.mjs` that default-exports
`[{ name, description, input, exec }]`.

## Related

- [node-hono-skills](../node-hono-skills): the model loads a skill itself, with
  a tool, instead of a person choosing one.
- [node-hono-tool-approval](../node-hono-tool-approval): buttons from an
  interrupt, which the engine resolves for you.
