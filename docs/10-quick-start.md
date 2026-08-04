---
title: Quick start
group: Getting started
---

Declare an agent, talk to it, give it a tool, pick the conversation back up,
add a teammate — then, when you outgrow the file, attach a worker. One file to
start, no dependencies.

## 1. Declare an agent

```sh
npm install -g @substructure.ai/cli
subs init
```

That writes a `substructure.toml`. The part that matters is four lines:

```toml title="substructure.toml"
[llm.claude]
type = "anthropic"

[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
```

`[llm.claude]` says what to call and `[agent.assistant]` says who is calling
it. The key never goes in the file — the engine reads it from the environment.

## 2. Send a message

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run --agent assistant -o pretty "hi"
```

The reply streams to your terminal. There is no worker yet: the engine derives
the next step of every turn and, because this agent has no `worker` URL,
accepts its own proposal.

`-o pretty` renders the turn for reading; the default `ag-ui` streams protocol
events instead, which is what a frontend wants. Add a `[run]` section to stop
repeating either flag:

```toml title="substructure.toml"
[run]
agent = "assistant"   # what a bare `subs run` drives
output = "pretty"
```

## 3. Add a tool

Tools the model calls come from an MCP connection or from your own worker code.
Declare a connection and point the agent at it:

```toml title="substructure.toml"
[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
mcp = ["files"]

[mcp.files]
url = "https://mcp.example.com/mcp"
```

```sh
subs mcp login files
subs run "what files are here?"
```

The engine resolves the connection's tools, offers them to the model, and runs
the calls itself. See [Connections](./85-connectors.md) for filtering which
tools the model sees.

## 4. Continue the conversation

Each run so far started a new session. After every run, the CLI prints the
command that continues it:

```
continue this session with:
  subs run --agent assistant --session <session-id> '...'
```

Run it with a follow-up question:

```sh
subs run --session <session-id> "what was my first question?"
```

The agent remembers. The whole session is persisted in `substructure.db`:
messages, tool calls, results. Stop everything, come back tomorrow, and it
still resumes.

## 5. Add a teammate

A second agent is a second section. It can run on its own model — a cheaper one
for the narrow job:

```toml title="substructure.toml"
[llm.claude]
type = "anthropic"

[llm.cheap]
type = "anthropic"

[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
system = "Delegate poems to the poet, then answer."
sub_agents = ["poet"]

[agent.poet]
description = "Writes a haiku on any topic."
llm = "cheap"
model = "claude-haiku-4-5"
system = "You are a poet. Respond with a single haiku."
```

The sub-agent appears to the model as a tool:

```sh
subs run "have the poet write a haiku about tuesday"
```

The poet runs in its own session, and its cost and token usage roll up to the
parent.

## 6. Add a worker

Everything so far is configuration. A **worker** is code: an HTTP service the
engine asks before each step of a turn, so you can override any decision the
engine would have made — approve a tool call, rewrite a prompt, branch on your
own state.

Attaching one is a single line on the agent that needs it:

```toml title="substructure.toml"
[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
worker = "http://localhost:4444"
```

The starter worker accepts every proposal, which is exactly what the engine was
already doing:

```javascript title="server.mjs"
// A complete worker, served with Node's built-in http server.
import { createServer } from "node:http";

function decide({ proposed }) {
    // The declared agent arrives as the `session.start` proposal, and every
    // other proposal is the step the engine would have taken. Echo them all,
    // then start overriding the ones you care about.
    return proposed;
}

const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        const decision = decide(JSON.parse(body));
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(decision));
    });
});

server.listen(4444, () =>
    console.log("worker listening on http://localhost:4444"));
```

```sh
node server.mjs      # in one terminal
subs run "hi"   # in another
```

Behaviour is unchanged — and nothing from steps 1–5 is discarded. Now override
one trigger at a time. A tool your worker runs, for example:

```javascript title="server.mjs"
const tools = [
    {
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: () => new Date().toISOString()
    }
];

function decide({ trigger, proposed }) {
    // Add our tools to the declared config, keeping its `llm` and `model`.
    if (trigger.type === "session.start") {
        return {
            agent: {
                ...proposed.agent,
                tools: tools.map(({ name, description }) => ({ name, description }))
            }
        };
    }

    // Run our tool when the model calls it.
    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        return { actions: [{ type: "tool.result", result: tool.exec() }] };
    }

    return proposed;
}
```

```sh
subs run "what time is it?"
```

The model calls the tool, your worker runs it, and the engine folds the result
back into the conversation.

Only the agents with a `worker` URL are pushed to one; the rest stay
engine-hosted, in the same app and the same file.

## Next

- [Quick start (cloud)](./15-quick-start-cloud.md): run this app against the hosted engine.
- [Core concepts](./20-concepts.md): sessions, turns, triggers, decisions.
- [Tool calls](./30-tools.md): schemas, validation, and where tools run.
- [Sub-agents](./80-sub-agents.md): delegation, child sessions, cost roll-up.
- [Durability](./110-durability.md): what's saved, and how runs recover.
- More agents in [`examples/`](../examples), in Node, Python, and Go.
