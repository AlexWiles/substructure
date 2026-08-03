# substructure.ai
Build production-ready AI agents in any language with no SDK

[![cli](https://img.shields.io/npm/v/@substructure.ai/cli?label=cli)](https://www.npmjs.com/package/@substructure.ai/cli)

> Pre-1.0: APIs and the wire protocol may change between releases.

Substructure is an open-source engine that drives the agent loop over HTTP. It keeps sessions durable and streams AG-UI events to your frontend. Your code stays on your infrastructure. It's just HTTP, so the engine is language agnostic. You don't even need an SDK.

## Why use it

- **In an existing codebase**: Your agent is just an HTTP endpoint. Tools
  are functions in your codebase, so they already have your database and
  your permissions.
- **In a new project**: The engine already handles sessions, history,
  streaming, and the client API. You write the agent and its tools, and
  that's most of the backend.
- **Long-running work**: First class support for async tools & async interrupt
  continuation. So tools can be background jobs and human approval can wait for
  days if needed.
- **Durability**: Every step is saved before it runs, so a deploy, a crash or a
  client reconnect doesn't lose anything.

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

There is no worker yet — the engine derives every step of the turn and, because
this agent has no `worker` URL, takes its own proposal.

**Add a worker when you outgrow the file.** A worker is an HTTP endpoint the
engine asks before each step, so you can override any decision it would have
made. Point one agent at it:

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
node server.mjs                    # one terminal
subs run --agent assistant "hi"    # another
```

The run command spins up an engine, which drives the turn and streams the response back to the terminal.

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
Your code can approve it or do something else. A working agent is a few lines.
Custom behavior is added by overriding the proposed decision.

Docs: [Core concepts](./docs/20-concepts.md)

Examples: [Node](./examples/node-hono-basic), [Python](./examples/python-fast-api-basic)

### Any language

Your agent is an HTTP endpoint. There is no SDK to install, and typed bindings can be generated from the published JSON schema.

Docs: [Typed bindings](./docs/40-typed-bindings.md)

Examples: [Go](./examples/go-chat-with-tools), [Python](./examples/python-fast-api-pydantic-chat-with-tools), [TypeScript](./examples/node-hono-typescript-chat-with-tools), [Elixir](./examples/elixir-plug-chat-with-tools)

### Crash recovery

Every step is saved before it runs. If the engine or your service dies mid-run, the run continues where it stopped. Submitting the same message twice won't run the turn twice. Client reconnects are handled.

Docs: [Durability](./docs/110-durability.md)

### Waiting for humans

An agent can stop and wait for a person to respond or approve then pick up where it left off. A waiting agent costs nothing to keep around.

Docs: [Interrupts](./docs/140-interrupts.md)

### Tools that take hours

A tool doesn't have to answer right away. Your service can acknowledge the call, do the work on its own schedule and report the result later. The run waits and resumes when the answer arrives.

Docs: [Deferred tools](./docs/130-deferred-tools.md)

### Full chat support

History, editing, regeneration, and branching are built into the engine. You don't build a chat backend. Users can edit an earlier message and go a new direction without losing the original thread.

Docs: [Conversations](./docs/70-conversations.md)

Examples: [Node](./examples/node-hono-plan-mode), [Python](./examples/python-fast-api-plan-mode)

### Works with existing chat UIs

The engine streams AG-UI events directly, so frontends like assistant-ui and CopilotKit connect to it.

Docs: [AG-UI](./docs/100-ag-ui.md)

Examples: [assistant-ui](./examples/node-hono-assistant-ui), [CopilotKit](./examples/node-hono-copilotkit)

### Tools that run in the browser

A tool can execute in the user's browser instead of on your server. The run pauses until the browser answers, then continues.

Docs: [Client-side tools](./docs/90-client-tools.md)

Examples: [Node](./examples/node-hono-client-tool)

### Agent state without a database

The engine stores your agent's working state next to the conversation. Your code gets it on every request and can write back changes.

Docs: [Agent state](./docs/60-state.md)

Examples: [Node](./examples/node-hono-plan-mode), [Python](./examples/python-fast-api-plan-mode)

### Sub-agents

An agent can hand work to other agents. Each one runs in its own session, and the cost and token usage roll up to the parent.

Docs: [Sub-agents](./docs/80-sub-agents.md)

Examples: [Node](./examples/node-hono-subagent), [Python](./examples/python-fast-api-subagent)

### Any LLM

The engine can call Anthropic, OpenAI, or OpenRouter with your keys or you can make the LLM calls yourself, through your own gateway or a provider the engine doesn't know about right inside your worker.

Docs: [LLMs](./docs/50-llms.md)

Examples: [Node + Anthropic](./examples/node-hono-anthropic), [Node + OpenAI](./examples/node-hono-openai), [Node + OpenRouter](./examples/node-hono-openrouter), [Python + Anthropic](./examples/python-fast-api-anthropic), [Python + OpenAI](./examples/python-fast-api-openai)

### Retries and timeouts

Set a timeout and retry policy on any tool or LLM call. The engine enforces them, including across restarts.

Docs: [Retries and timeouts](./docs/120-retries.md)

### Validated tool calls

Give a tool an input and output schema and the engine checks every call against it. Malformed calls go back to the model to fix itself instead of reaching your code.

Docs: [Tool calls](./docs/30-tools.md)

Examples: [Node](./examples/node-hono-tools), [Python](./examples/python-fast-api-tools), [Go](./examples/go-chat-with-tools)

### MCP servers

Your worker can connect to an MCP server, expose its tools to the model, and forward each call back to it. The engine never needs to know MCP exists.

Examples: [Node](./examples/node-hono-mcp), [Python](./examples/python-fast-api-mcp), [Go](./examples/go-mcp)

## The system pieces

- **Server:** The engine that drives the agent loop, written in Rust. It can be run locally on your machine, embedded in process, or as a cloud hosted version available at [https://app.substructure.ai](https://app.substructure.ai). The server drives the loop, handles durability, retries, llm calls (optionally), realtime streaming, subagent supervision and more.
- **Workers:** Your agent logic. Receives a decision trigger, returns actions. Runs in your codebase with your dependencies. Can be an HTTP endpoint for use with the cloud/local server, or a callback passed to embedded substructure.
- **Clients:** Submit work and stream events back, backend-to-backend or straight from the browser. The engine also serves a native **AG-UI** endpoint, so any AG-UI chat frontend connects and streams directly.
- **CLI:** Substructure comes with a CLI to help you provision, observe, and debug from the terminal. You can also start a local server.

## Install

The CLI is available at:
```sh
npm i -g @substructure.ai/cli
```

## Docs
Full documentation in [`docs/`](./docs).
