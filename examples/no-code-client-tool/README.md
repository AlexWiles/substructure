# no-code-client-tool

A tool the file declares and the client runs. There is no worker and no code.

The tool is `handler = "client"`, so the engine hands the call to whoever is
connected — a browser, an app, or the CLI — and waits. This is the only handler
a file can declare, because the other handlers are worker code.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Deploy the file and give it a key:

```sh
subs login
subs apply
subs llm set-key openrouter
```

Ask something that depends on where you are. The model calls `get_location`, and
the turn yields with the call pending:

```sh
subs run "recommend a coffee shop near me"
```

Pretty output prints the pending call with its id, and the command to continue
the session. Take that command and answer the call:

```sh
--input '{"type":"tool.result","id":"<toolCallId>","result":"Lisbon"}'
```

The turn resumes where it stopped. A call can wait as long as it needs to.

## Run it here instead

Delete `[remote]` and the turn runs on this machine, on your own key:

```sh
export OPENROUTER_API_KEY=sk-or-...
subs run -c substructure.toml --agent assistant "recommend a coffee shop near me"
```
