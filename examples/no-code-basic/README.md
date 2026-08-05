# no-code-basic

An agent that is one file. There is no worker and no code.

The engine proposes each step of a turn. This agent has no `worker`, so the
engine accepts its own proposal.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

One terminal.

```sh
export OPENROUTER_API_KEY=sk-or-...
subs run -c substructure.toml --agent assistant "hi"
```

## Deploy

```sh
subs login
subs apply
subs llm set-key openrouter
```

The agent now runs on the deployment `[remote]` names, on the key you uploaded.
The file never holds a key. It only names one.

## Outgrow the file

Add a `worker` URL to the agent, and the engine POSTs every decision to your
code instead. See [node-hono-basic](../node-hono-basic).
