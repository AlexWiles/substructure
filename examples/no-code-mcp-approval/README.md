# no-code-mcp-approval

An agent that stops and asks a person before a connection's destructive tools
run. The agent is one file. There is no worker and no code.

`approve` on the entry that names the connection says which of its calls wait.

```toml
[agent.ops]
mcp = [{ id = "issues", approve = "destructive" }]
```

`mcp-server.mjs` is an MCP server standing in for a real one: an issue tracker
with a tool that reads and a tool that destroys. What matters is its
annotations, because they are what `destructive` reads.

```jsonc
{ "name": "search_issues", "annotations": { "readOnlyHint": true,  "destructiveHint": false } }
{ "name": "delete_issue", "annotations": { "readOnlyHint": false, "destructiveHint": true  } }
```

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Start the connection and give the engine a key. The connection is on this
machine, so the turn runs here too — there is no `[remote]` in this file.

```sh
node mcp-server.mjs &
export OPENROUTER_API_KEY=sk-or-...
```

A read runs as it always has:

```sh
subs run ops "which issues mention the login page?"
```

```text
● issues__search_issues {"q":"login page"}
  #7: the login page is blank
```

A delete stops:

```sh
subs run ops "delete issue 7"
```

````text
⚠ interrupt mcp-approve:toolu_01Vnn… [`issues__delete_issue` needs approval before it runs]: Run `issues__delete_issue`?

```
{
  "id": "7"
}
```
````

Nothing was dialled. The arguments are in the question because the tool's name
says what would happen and only they say what it would happen to.

## Answering

Resume the interrupt by id, with the session the run printed.

```sh
subs run ops --session <session> --input '{"type":"interrupt.resume","interrupt_id":"mcp-approve:<tool call id>","payload":{"approved":true}}'
```

`true` runs the call. Anything else declines it, and the model reads that a
person declined instead of a result:

```text
Issue #7 has been deleted. Issue #9 could not be deleted—the deletion was declined.
```

In Slack the question is a message in the thread with `Run it` and `Decline`
buttons, and no id is typed. See [Slack](../../docs/130-slack.md#interrupt-prompts).

## In a chat

`subs chat` holds the session open, so the question arrives where you are and
no id is typed here either. The options are the same ones Slack draws as
buttons.

```sh
subs chat ops
```

````console
> delete issue 7
? Run `issues__delete_issue`?

```
{
  "id": "7"
}
```  ›
❯ Run it
  Decline
````

`↑`/`↓` pick and `Enter` answers. `Run it` runs the call and the turn carries
on in the same chat:

```console
✔ Run `issues__delete_issue`? · Run it
● issues__delete_issue {"id":"7"}
  deleted issue #7
Done. Issue 7 deleted.

>
```

`Decline` dials nothing, and the model reads the decline instead of a result:

```console
✔ Run `issues__delete_issue`? · Decline
The deletion of issue 7 was declined. …
```

`Ctrl-C` at the question leaves it parked in the session, and
`subs chat ops --session <id>` asks it again.

See [Chat](../../docs/135-chat.md).

## Two calls at once

Ask for both and the model sends two calls in one message.

```sh
subs run ops "delete issues 7 and 9"
```

Each is asked about on its own, so one can run and the next be declined. The
question says how many are behind it:

````text
⚠ interrupt … Run `issues__delete_issue`?

```
{
  "id": "7"
}
```

One more call waits behind it.
````

The questions come in turn: the answer to one raises the next. Nothing of that
message runs until every one is answered, including the calls nobody asks
about — a call dispatched while a question is open could settle first, and the
model would be prompted again with the held call unanswered.

## Settings

| `approve` | Asks about |
| --- | --- |
| `never` (the default) | Nothing. |
| `destructive` | Each tool the connection marks `destructiveHint`. |
| `always` | Every call on the connection. |

`destructive` asks about what a server says destroys something, and nothing
else. Delete the `annotations` from `delete_issue` in `mcp-server.mjs` and run
the delete again — it goes straight through, because the server no longer says
it destroys anything.

That is the limit of the setting: it is the server's word about itself, and a
server owes you no annotation at all. Use `always` where the answer must not
depend on that.

```toml
[agent.ops]
mcp = [{ id = "issues", approve = "always" }]
```

The setting belongs to the pair, not to the connection. One connection and one
credential can serve an agent a person watches and an agent that runs on a
schedule, and only the first can be asked.

```toml
[agent.ops]
llm = "openrouter"
model = "anthropic/claude-haiku-4.5"
mcp = [{ id = "issues", approve = "destructive" }]   # a person is watching; it asks

[agent.digest]
llm = "openrouter"
model = "anthropic/claude-haiku-4.5"
mcp = ["issues"]                                     # runs nightly; nothing asks
```

A session with no channel to show the question in stops until someone resumes
it by id, so leave `approve` off for an agent that runs unattended.

See [Approving a call](../../docs/40-connectors.md#approving-a-call).

## Next

- [node-hono-tool-approval](../node-hono-tool-approval): the same pause for a
  tool your own worker runs, authored by the worker.
- [no-code-mcp](../no-code-mcp): a connection with nothing to approve.
