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
assistant
  model   claude-sonnet-5
  llm     claude (anthropic)
  session 01a02417-7d46-7441-8090-23b20d0f980f

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

A line ending in `\` is not finished, so `Enter` opens the next one and the
message carries the newline. `Alt-Enter` and `Ctrl-J` do the same.

```console
> summarise this: \
  the first point \
  the second point
```

`Shift-Enter` is not one of them, and cannot be: a terminal sends the same
byte for `Enter` and `Shift-Enter`, so nothing downstream can tell them apart.
Terminals that let you map a key can send `Alt-Enter`'s bytes — `ESC` then
`CR` — for it:

```
kitty      map shift+enter send_text all \x1b\r
wezterm    { key = "Enter", mods = "SHIFT", action = act.SendString("\x1b\r") }
alacritty  { key = "Return", mods = "Shift", chars = "\u001b\r" }
iTerm2     Keys → Key Bindings → Shift-Enter → Send Hex Code → 0x1b 0x0d
```

## While a turn runs

A turn is quiet between the model call and its first token, and again while a
tool runs. A line below the transcript names the step in progress and counts
the time it takes.

```console
⠙ fetch_url (7s)
```

It names one call, counts a batch of them (`2 tools`), and counts the attempts
when the engine tries the same call again. The line is drawn on stderr and
erased before anything else is written, so `subs run` piped into another
program still writes only the turn.

A call is written once, when it is answered: the status line is what says it
is running, so the transcript keeps one line per call.

```console
● get_current_time (180ms)
  2026-08-22T09:50:33.010Z
↻ fetch_url {"url":"https://example.com"} (attempt 1, 2.1s)
  503 Service Unavailable
● fetch_url {"url":"https://example.com"} (attempt 2, 1.4s)
  {"status":"ok"}
```

A call on a connection is named by the server and the tool, as each of them
says it is called:

```console
● deepwiki Ask a question  {"q":"login page"} (1.2s)
```

`●` answered, `✗` failed, `↻` failed and will be tried again. A call that took
less than a moment shows no time. A result that is only text reads as that
text.

A result longer than the screen is cut short, and the rest is counted:

```console
  … +182 lines
```

## What the answer looks like

The answer is markdown, and it reads as markdown: headings, `code`, **bold**,
lists, quotes, and links carry their styling rather than their markers. A line
is styled once it is whole and a fenced block once it closes, so the transcript
is written once and never rewritten.

Styling is bold, dim, italic, and underline, and nothing else. No colour is
set, so the transcript reads on a light background as well as a dark one and
there is no theme to keep right.

A model told to answer in another dialect is read as CommonMark all the same —
Slack's `*bold*` is CommonMark's *italic*, so an agent whose system prompt asks
for Slack mrkdwn reads differently here than it does in Slack.

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
commands, and it cannot send a message while a turn is running.

Nothing here stops a turn the engine has started. `Ctrl-C` during a turn stops
watching it and prints how to pick the session back up; the turn runs to its
end, and the next chat on that session reads what it wrote.

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
