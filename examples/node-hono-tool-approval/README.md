# node-hono-tool-approval

Human-in-the-loop tool approval, served with [Hono](https://hono.dev). The
`send_email` tool doesn't run when the model calls it: the worker raises an
interrupt carrying an approval `prompt` (message + options), the session
parks, and the resolution decides — approve runs the held call, deny hands
the model a "Denied" tool result it can react to.

In Slack the prompt posts as buttons and a click resumes the interrupt. On
the CLI the same resolution is an `interrupt.resume` input.

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

**2. Drive a session with the CLI.** Reuse one `--session` across turns.

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run -c substructure.toml --agent my-agent "email bob@example.com a hello"
```

The run parks on the prompt and prints the interrupt id:

```
⚠ interrupt approve:tc_... [tool_call]: Run `send_email`? ...
```

Resolve it — the AG-UI resume shape, with the inner `payload` a chosen
option's value, verbatim:

```sh
subs run -c substructure.toml \
    --session approval-demo \
    --input '{"type":"interrupt.resume","interrupt_id":"approve:tc_...","payload":{"status":"resolved","payload":{"decision":"approve"}}}'
```

The held call runs and the turn finishes. Send `{"decision":"deny"}` instead
and the model gets `Denied by the user.` as the tool result.

## Slack

Serve the same worker behind the Slack channel
(see [the Slack docs](../../docs/105-slack.md); the app needs Interactivity
enabled for buttons):

```sh
export SLACK_APP_TOKEN=xapp-... SLACK_BOT_TOKEN=xoxb-... ANTHROPIC_API_KEY=sk-ant-...
subs serve -c substructure.toml
```

Ask the bot to send an email: the thread gets the prompt with
**Approve / Deny** buttons, and the click's resolution (who, which option)
is stamped back onto the message.
