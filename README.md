# substructure.ai
Build production-ready AI agents in any language with no SDK

[![cli](https://img.shields.io/npm/v/@substructure.ai/cli?label=cli)](https://www.npmjs.com/package/@substructure.ai/cli)

> Pre-1.0: APIs and the wire protocol may change between releases.

Substructure is an open-source engine that runs the agent loop over HTTP. It
saves every session and streams AG-UI events to your frontend. Your code stays
on your infrastructure. Everything is HTTP, so you can write your agent in any
language. There is no SDK to install.

## Why use it

- **In an existing codebase**: Your agent is an HTTP endpoint. Tools are
  functions in your codebase. They already have your database and your
  permissions.
- **In a new project**: The engine handles sessions, history, streaming, and the
  client API. You write the agent and its tools. That is most of the backend.
- **Long-running work**: Tools can be background jobs. A request for human
  approval can wait for days.
- **Durability**: Every step is saved before it runs. A deploy, a crash, or a
  client reconnect loses nothing.

## See it in action

Install the CLI and declare an agent.

```sh
npm install -g @substructure.ai/cli
subs init
```

That writes a `substructure.toml`. The part that matters is four lines:

```toml
[llm.claude]
type = "anthropic"

[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
```

Send it a message:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run --agent assistant -o pretty "hi"
```

There is no worker yet. The engine plans each step of the turn. This agent has
no `worker` URL, so the engine accepts its own plan.

**Add a worker when you outgrow the file.** A worker is an HTTP endpoint. The
engine asks it before each step, so you can change any decision the engine
would make. Point one agent at it:

```toml
[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
worker = "http://localhost:4444"
```

```javascript
// A complete worker, served with Node's built-in http server. No dependencies.
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
node server.mjs                    # one terminal
subs run --agent assistant "hi"    # another
```

The run command starts an engine. The engine runs the turn and streams the reply
to the terminal.

Next:

- The [quick start](./docs/10-quick-start.md) continues from here: add a tool,
  continue the conversation, add a teammate, then attach a worker.
- The same agent in [Python](./examples/python-fast-api-basic), or in
  [Go with tools and generated types](./examples/go-chat-with-tools).
- A [full chat UI](./examples/node-hono-assistant-ui) talking to the engine
  over AG-UI.

## Features

### Write only the logic you care about

At each step of the loop, the engine tells your code what it plans to do next.
Your code can accept the plan or do something else. A working agent is a few
lines. To add your own behavior, change the plan.

Docs: [Core concepts](./docs/20-concepts.md)

Examples: [Node](./examples/node-hono-basic), [Python](./examples/python-fast-api-basic)

### Any language

Your agent is an HTTP endpoint. There is no SDK to install. You can generate typed bindings from the published JSON schema.

Docs: [Typed bindings](./docs/40-typed-bindings.md)

Examples: [Go](./examples/go-chat-with-tools), [Python](./examples/python-fast-api-pydantic-chat-with-tools), [TypeScript](./examples/node-hono-typescript-chat-with-tools), [Elixir](./examples/elixir-plug-chat-with-tools)

### Crash recovery

Every step is saved before it runs. If the engine or your service stops during a run, the run continues from where it stopped. If you submit the same message twice, the turn runs once. Clients can reconnect.

Docs: [Durability](./docs/110-durability.md)

### Waiting for humans

An agent can stop and wait for a person to answer or approve, then continue from where it stopped. A waiting agent uses no compute.

Docs: [Interrupts](./docs/140-interrupts.md)

### Tools that take hours

A tool does not have to answer immediately. Your service can accept the call, do the work on its own schedule, and report the result later. The run waits, then continues when the answer arrives.

Docs: [Deferred tools](./docs/130-deferred-tools.md)

### Full chat support

The engine has history, editing, regeneration, and branching. You do not build a chat backend. A user can edit an earlier message and go in a new direction. The original thread stays.

Docs: [Conversations](./docs/70-conversations.md)

Examples: [Node](./examples/node-hono-plan-mode), [Python](./examples/python-fast-api-plan-mode)

### Works with existing chat UIs

The engine streams AG-UI events, so frontends like assistant-ui and CopilotKit connect to it directly.

Docs: [AG-UI](./docs/100-ag-ui.md)

Examples: [assistant-ui](./examples/node-hono-assistant-ui), [CopilotKit](./examples/node-hono-copilotkit)

### Tools that run in the browser

A tool can run in the user's browser instead of on your server. The run waits for the browser to answer, then continues.

Docs: [Client-side tools](./docs/90-client-tools.md)

Examples: [Node](./examples/node-hono-client-tool)

### Agent state without a database

The engine stores your agent's state with the conversation. Your code gets the state on every request, and can write changes back.

Docs: [Agent state](./docs/60-state.md)

Examples: [Node](./examples/node-hono-plan-mode), [Python](./examples/python-fast-api-plan-mode)

### Sub-agents

An agent can give work to other agents. Each one runs in its own session. Their cost and token use are added to the parent.

Docs: [Sub-agents](./docs/80-sub-agents.md)

Examples: [Node](./examples/node-hono-subagent), [Python](./examples/python-fast-api-subagent)

### Any LLM

The engine can call Anthropic, OpenAI, or OpenRouter with your keys. Or your worker can make the calls itself, through your own gateway or a provider the engine does not know.

Docs: [LLMs](./docs/50-llms.md)

Examples: [Node + Anthropic](./examples/node-hono-anthropic), [Node + OpenAI](./examples/node-hono-openai), [Node + OpenRouter](./examples/node-hono-openrouter), [Python + Anthropic](./examples/python-fast-api-anthropic), [Python + OpenAI](./examples/python-fast-api-openai)

### Retries and timeouts

Set a timeout and a retry policy on any tool or LLM call. The engine applies them, and keeps applying them after a restart.

Docs: [Retries and timeouts](./docs/120-retries.md)

### Validated tool calls

Give a tool an input and output schema. The engine checks every call against it. Bad calls go back to the model to correct. They do not reach your code.

Docs: [Tool calls](./docs/30-tools.md)

Examples: [Node](./examples/node-hono-tools), [Python](./examples/python-fast-api-tools), [Go](./examples/go-chat-with-tools)

### MCP servers

Your worker can connect to an MCP server, give its tools to the model, and send each call back to it. The engine does not need to know about MCP.

Examples: [Node](./examples/node-hono-mcp), [Python](./examples/python-fast-api-mcp), [Go](./examples/go-mcp)

## The parts

- **Server:** The engine that runs the agent loop, written in Rust. Run it on your machine, embed it in your process, or use the hosted version at [https://app.substructure.ai](https://app.substructure.ai). It runs the loop, saves each step, retries failures, makes LLM calls, streams events, and supervises sub-agents.
- **Workers:** Your agent code. It receives a trigger and returns actions. It runs in your codebase with your dependencies. It can be an HTTP endpoint, or a callback if you embed the engine.
- **Clients:** They send work and stream events back, from your backend or from the browser. The engine also serves an **AG-UI** endpoint, so any AG-UI chat frontend can connect and stream.
- **CLI:** Use it to set up, watch, and debug from the terminal. It also starts a local server.

## Install

The CLI is available at:
```sh
npm i -g @substructure.ai/cli
```

## Docs
Full documentation in [`docs/`](./docs).
