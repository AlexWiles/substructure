# no-code-chat

A conversation that stays open, in the terminal. The agent is one file. There
is no worker and no code.

`subs run` sends one message and exits, so the next message needs `--session`
and a second process. `subs chat` holds the session, so the next question can
depend on the last one.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

This file names no `[remote]`, so the turn runs on this machine, on your own
key:

```sh
export OPENROUTER_API_KEY=sk-or-...
subs chat assistant
```

```text
substructure · assistant · 01a02417-7d46-7441-8090-23b20d0f980f

> what is the capital of Portugal?
The capital of Portugal is **Lisbon** (Lisboa in Portuguese). …

> how far is it from Madrid?
Lisbon is approximately **505-520 kilometers (about 315-325 miles)** from
Madrid. …

>
```

The second question never says Portugal. It is the same session, so the agent
still has the first.

## Leaving and coming back

`Ctrl-D` ends the chat and prints the session it held:

```text
continue this session with:
  subs chat assistant --session 01a02417-7d46-7441-8090-23b20d0f980f
```

The session is in the database, not in the process, so the answer survives the
exit:

```text
> which two cities have we been discussing?
The two cities we have been discussing are **Lisbon** … and **Madrid** …
```

`↑` walks what you typed before, across chats as well as within one.

## Against a deployment

Add a `[remote]`, and the same command chats with the deployment instead. The
file decides where a turn runs, as it does for every other command.

```toml
[remote]
url = "https://api.substructure.ai"
```

```sh
subs login
subs apply
subs llm set-key openrouter
subs chat assistant
```

The session lives on the deployment now, so `subs sessions list` shows it and
the key stays there rather than in your environment.

## Interrupts

A turn that stops to ask something shows the question as a picker, and your
answer resumes it — no id is typed. See
[no-code-mcp-approval](../no-code-mcp-approval#in-a-chat).

## Next

- [no-code-basic](../no-code-basic): the same agent, one message at a time.
- [Chat](../../docs/135-chat.md): what the channel renders, and what it does not.
- [Slack](../../docs/130-slack.md): the same session, in a thread.
