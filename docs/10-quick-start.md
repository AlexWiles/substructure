---
title: Quick start
group: Getting started
---

Declare an agent, talk to it, give it a tool, continue the conversation, and add
a second agent. Then attach a worker when you outgrow the file. You start with
one file and no dependencies.

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

`[llm.claude]` says what to call. `[agent.assistant]` says who calls it. The key
never goes in the file. The engine reads it from the environment.

## 2. Send a message

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run --agent assistant -o pretty "hi"
```

The reply streams to your terminal. There is no worker yet. The engine plans
each step of the turn. This agent has no `worker` URL, so the engine accepts its
own plan.

`-o pretty` shows the turn as text to read. The default, `ag-ui`, streams
protocol events, which is what a frontend needs. Add a `[run]` section so you do
not repeat these flags:

```toml title="substructure.toml"
[run]
agent = "assistant"   # which agent a bare `subs run` uses
output = "pretty"
```

## 3. Add a tool

The tools a model calls come from an MCP connection or from your worker code.
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

The engine reads the connection's tools, gives them to the model, and runs the
calls itself. See [Connectors](./85-connectors.md) to select which tools the
model sees.

## 4. Continue the conversation

Each run so far started a new session. After every run, the CLI prints the
command to continue it:

```
continue this session with:
  subs run --agent assistant --session <session-id> '...'
```

Run it with a follow-up question:

```sh
subs run --session <session-id> "what was my first question?"
```

The agent remembers. The engine saves the whole session in `substructure.db`:
messages, tool calls, and results. Stop everything, come back tomorrow, and the
session continues.

## 5. Add a second agent

A second agent is a second section. It can use its own model. Use a cheaper
model for a small job:

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

The model sees the sub-agent as a tool:

```sh
subs run "have the poet write a haiku about tuesday"
```

The poet runs in its own session. Its cost and token use are added to the
parent.

## 6. Add a worker

Everything above is configuration. A **worker** is code. It is an HTTP service
that the engine asks before each step of a turn. Use it to change any decision
the engine would make: approve a tool call, rewrite a prompt, or branch on your
own state.

Add one line to the agent that needs it:

```toml title="substructure.toml"
[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
worker = "http://localhost:4444"
```

This first worker accepts every proposal. That is what the engine already did:

```javascript title="server.mjs"
// A complete worker, served with Node's built-in http server.
import { createServer } from "node:http";

function decide({ proposed }) {
    // The engine proposes each step. Return the proposal to accept it.
    // Change the ones you care about.
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

The behavior does not change, and steps 1 to 5 still apply. Now change one
trigger at a time. For example, add a tool that your worker runs:

```javascript title="server.mjs"
const tools = [
    {
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: () => new Date().toISOString()
    }
];

function decide({ trigger, proposed }) {
    // Add our tools to the config from the file. Keep its `llm` and `model`.
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

The model calls the tool, your worker runs it, and the engine adds the result to
the conversation.

Only the agents with a `worker` URL use a worker. The others stay with the
engine, in the same app and the same file.

## Next

- [Quick start (cloud)](./15-quick-start-cloud.md): run this app on the hosted engine.
- [Core concepts](./20-concepts.md): sessions, turns, triggers, and decisions.
- [Tool calls](./30-tools.md): schemas, validation, and where tools run.
- [Sub-agents](./80-sub-agents.md): child sessions and cost.
- [Durability](./110-durability.md): what is saved, and how a run recovers.
- More agents in [`examples/`](../examples), in Node, Python, and Go.
