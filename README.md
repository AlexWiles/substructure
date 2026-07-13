# substructure.ai: Build production-ready AI agents in any language with no SDK

[![cli](https://img.shields.io/npm/v/@substructure.ai/cli?label=cli)](https://www.npmjs.com/package/@substructure.ai/cli)

> Substructure is under active development. APIs, CLI commands, and the wire protocol may change between releases for versions 0.1.x

Substructure is an open-source engine that drives the agent loop over HTTP. It keeps sessions durable and streams AG-UI events to your frontend. Your code stays on your infrastructure. It's just HTTP, so the engine is language agnostic. You don't even need an SDK.

## See it in action
Two terminals.

**1. Create and start the agent worker**:

Create a server.mjs with this content:

```javascript
// A complete chat agent served with Node's built-in http server. No dependencies.
import { createServer } from "node:http";

function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        // The engine will use this agent config to generate proposed actions.
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true
            }
        };
    }

    // Accept the engine's proposal for every other decision.
    return proposed;
}

const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        const decision = decide(JSON.parse(body));
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(decision ?? null));
    });
});

server.listen(4444, () =>
    console.log("worker listening on http://localhost:4444"));
```

Start the worker

```sh
node server.mjs
```

**2. Send a message with the CLI**

In another terminal.

Install the CLI.

```sh
npm install -g @substructure.ai/cli
```

Then send a message to the worker.

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "hi"}}'
```

## Features

- **Write only the logic you care about**: At each step of the loop, the engine
  tells your code what it plans to do next. Your code can approve it or do
  something else. A working agent is a few lines; custom behavior is added one
  decision at a time.
- **Any language**: Your agent is an HTTP endpoint. If it can receive and return
  JSON, it works. There is no SDK to install, and typed bindings can be generated
  from the published JSON schema.
- **Crash recovery**: Every step is saved before it runs. If the engine or your
  service dies mid-run, the run continues where it stopped. Submitting the same
  message twice won't run the turn twice.
- **Waiting for humans**: An agent can stop and wait for a person to respond or
  approve — for minutes or weeks — then pick up where it left off. A waiting agent
  costs nothing to keep around.
- **Retries and timeouts**: Set a timeout and retry policy on any tool or LLM
  call. The engine enforces them, including across restarts.
- **Sub-agents**: An agent can hand work to other agents. Each one runs in its own
  session, and the cost and token usage roll up to the parent.
- **Tools that run in the browser**: A tool can execute in the user's browser
  instead of on your server. The run pauses until the browser answers, then
  continues.
- **Full chat support**: History, editing, regeneration, and branching are
  built into the engine. You don't build a chat backend. Users can edit an
  earlier message and go a new direction without losing the original thread.
- **Works with existing chat UIs**: The engine streams AG-UI events directly, so
  frontends like assistant-ui and CopilotKit connect to it with no adapter or
  backend glue.
- **Any LLM**: The engine can call Anthropic, OpenAI, or OpenRouter with your
  keys or you can make the LLM calls yourself, through your own gateway or
  a provider the engine doesn't know about right inside your worker.
- **Agent state without a database**: The engine stores your agent's working
  state next to the conversation. Your code gets it on every request and can
  write back changes — per-conversation memory with nothing extra to set up.
- **Tools that take hours**: A tool doesn't have to answer right away. Your
  service can acknowledge the call, do the work on its own schedule — a queue,
  a batch job, another system — and report the result later. The run waits
  durably and picks up when the answer arrives.
- **Validated tool calls**: Give a tool an input and output schema and the
  engine checks every call against it. Malformed calls go back to the model to
  fix itself instead of reaching your code.

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
