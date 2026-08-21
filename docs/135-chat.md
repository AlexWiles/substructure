---
title: Chat
group: Frontends
---

`subs chat` is an agent session in your terminal. It is a channel, like
[Slack](./130-slack.md): a line you type becomes a client message, and the
event stream becomes text.

```sh
subs chat
```

```console
substructure · assistant · 01a02417-7d46-7441-8090-23b20d0f980f

> what is the capital of Portugal?
The capital of Portugal is Lisbon.

> how far is it from Madrid?
About 500 km.

>
```

`subs run` sends one message and exits, so a second message needs `--session`
and a second process. Chat holds the session, so the next question can depend
on the last one.

## Where the turn runs

The file decides, as it does for every other command. A file that names no
`[remote]` describes an engine here, so chat starts one in the process and the
session lands in this machine's database. A file that names one describes a
deployment, so chat streams the turn from there.

```sh
subs chat                          # the file decides
subs chat --url http://localhost:8080   # a deployment, for this chat
```

`--url` also points at a `subs serve` you are running, which is how to chat
with the engine that answers your Slack workspace.

See [CLI](./260-cli.md#where-a-command-acts).

## Which agent

Chat drives the agent `[run].agent` names — the same question `subs run`
answers, so one file serves both. `--agent <id>` picks another for one chat.

```toml
[run]
agent = "assistant"
```

## The session

A chat with no `--session` opens a new one and prints its id. `Ctrl-D` ends
the chat and prints how to pick it back up.

```console
continue this session with:
  subs chat --session 01a02417-7d46-7441-8090-23b20d0f980f
```

The session is durable, so it outlives the process: `subs sessions list` shows
it, `subs run --session` adds a turn to it, and a Slack thread on the same
engine is the same kind of thing. See [Conversations](./120-conversations.md).

`↑` walks the lines you have typed, in this chat and in earlier ones. The
history is beside your credentials, in
`$XDG_CONFIG_HOME/substructure/chat_history`.

## Interrupt prompts

A turn that stops to ask something renders the question where you are, and
your answer resumes it. No id is typed.

An interrupt whose `metadata.options` names answers renders them as a picker —
the same list Slack draws as buttons. `↑`/`↓` pick, `Enter` answers, and the
turn carries on in the same chat.

````console
? Run `issues__delete_issue`?

```
{
  "id": "7"
}
```  ›
❯ Run it
  Decline
````

An interrupt that offers no options takes typed text, delivered as the
resolution payload whole.

A question waits where a Slack prompt's buttons wait: in the session, not in
the process. `Ctrl-C` at a question ends the chat and leaves it parked, and a
chat that opens a parked session — `--session`, after any exit — asks the
question first, before reading a line.

What chat resolves is stamped `channel: "cli"`, so the session records which
channel answered and which option was picked, the same way Slack's does. See
[Interrupts](./100-interrupts.md).

## What it does not do

The first cut renders a turn and answers what it parks on. It has no slash
commands, it cannot send a message while a turn is running, and `Ctrl-C` at
the prompt ends the chat rather than the turn.

A client tool ends the turn with nothing to settle it, so chat says so and
stops rather than offering an input it cannot send:

```console
⧗ get_weather is waiting on a client-side result, which this chat cannot settle.
```

Use `subs run --input '{"type":"tool.result",…}'` for those. See
[Client tools](./150-client-tools.md).

## Related

- [`no-code-chat`](../examples/no-code-chat): a chat agent that is one file.
- [Slack](./130-slack.md): the same session, in a thread.
- [CLI](./260-cli.md): every command and flag.
