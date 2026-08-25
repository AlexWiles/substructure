# substructure.ai
[substructure.ai](https://substructure.ai)

> Pre-1.0: APIs and the wire protocol might change between releases.

Subs is an agent harness for the cloud.

It runs an unprivileged agent loop with no system access. It uses MCP servers for tools. It can run locally or as a client and server.

Declare your agents in a config file. If you need full customization, point
the agent at an HTTP endpoint and implement a webhook that drives the agent loop.

Subs handles durability, retries, timeouts, MCP connection management, session state, session branching, AG-UI, Slack connections, LLM calls, Sub agents, interrupts and more.

Need a quick way to turn a sanbox into and MCP server? Check out [mcpd](https://github.com/substructureai/mcpd).

# Get started

## Intall the engine

```sh
curl -fsSL https://subs.dev/cli.sh | bash
```

## Run a local agent

Create a `subs.toml`

```toml
name = "example"

[llm.openrouter]
type = "openrouter"

[agent.buddy]
llm = "openrouter"
model = "deepseek/deepseek-v4-flash-0731"
system = "You are the a helpful buddy."
```

```sh
subs chat buddy -c subs.toml
```

## Use server/client mode

Add this to `subs.toml`

```toml
# server config
[serve]
port = 9999
auth = false

# remote client config
[remote]
url = "http://localhost:9999"
```

Start the server

```sh
subs serve -c subs.toml
```

In another terminal:

```sh
subs chat buddy -c subs.toml
```

## Put an agent in Slack using hosted substructure

```toml
name = "example"

[llm.openrouter]
type = "openrouter"

[agent.buddy]
llm = "openrouter"
model = "deepseek/deepseek-v4-flash-0731"
system = "You are the a helpful buddy."

[slack]
dm = "buddy"

[remote]
url = "https://api.substructure.ai"
```

```sh
subs apply
subs auth llm.openrouter
subs slack connect
```

Mention the bot in a channel and it answers in the thread. You wrote one file
and no code.

Full walkthrough in the [quick start](./docs/10-quick-start.md).

## Control the loop with your own code

Add a `worker` to any agent. The engine then asks your endpoint before each step
of a turn.

```toml
[agent.oncall]
llm = "openrouter"
model = "anthropic/claude-sonnet-4-5"
worker = "http://localhost:4444"
```

The engine proposes each step. Return the proposal to accept it. Change the ones
you care about.

```javascript
// A complete worker, on Node's built-in http server. No dependencies.
import { createServer } from "node:http";

function decide({ trigger, proposed }) {
    // Run the tool when the model calls it.
    if (trigger.type === "tool.execute" && trigger.name === "get_time") {
        return { actions: [{ type: "tool.result", result: new Date().toISOString() }] };
    }

    return proposed;
}

const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(decide(JSON.parse(body))));
    });
});

server.listen(4444);
```

Only the agents that name a worker use one. The rest stay with the engine, in
the same project and the same file.

Docs: [Workers](./docs/50-workers.md)

## Develop locally

Run the engine on your machine and iterate before you deploy. A file with no
`[remote]` runs the turn here, on your own key.

```sh
export OPENROUTER_API_KEY=sk-or-...
subs run oncall "what is broken?"
```

The reply streams to your terminal. `subs serve` runs the same engine as an HTTP
server, with the AG-UI and REST endpoints a frontend needs.

Slack, MCP connections, and workers all work locally.

Docs: [Local development](./docs/160-local-development.md)

## Why use it

- **In an existing codebase.** Your agent is an HTTP endpoint. Tools are
  functions you already have, with your database and your permissions.
- **In a new project.** The engine handles sessions, history, streaming, and the
  client API. You write the agent and its tools.
- **Long-running work.** A tool can be a background job. A request for approval
  can wait for days.
- **Durability.** Every step is saved before it runs. A deploy, a crash, or a
  reconnect loses nothing.

## Features

### Talk to your agents in Slack

Mention the bot or DM it. The thread is the session. Route different channels to
different agents.

Docs: [Slack](./docs/130-slack.md)

### Talk to them in your terminal

`subs chat` holds one session open, streams the reply as it is written, and
turns an approval prompt into a picker. The session is the same kind a Slack
thread is.

Docs: [Chat](./docs/135-chat.md)

Examples: [no-code-chat](./examples/no-code-chat)

### Write only the logic you care about

At each step the engine tells your code what it plans to do next. Accept the
plan or do something else. A working agent is a few lines.

Docs: [How it works](./docs/20-how-it-works.md)

Examples: [Node](./examples/node-hono-basic), [Python](./examples/python-fast-api-basic)

### Any language

Your agent is an HTTP endpoint. Generate typed bindings from the published JSON
schema.

Docs: [Typed bindings](./docs/270-typed-bindings.md)

Examples: [Go](./examples/go-chat-with-tools), [Python](./examples/python-fast-api-pydantic-chat-with-tools), [TypeScript](./examples/node-hono-typescript-chat-with-tools), [Elixir](./examples/elixir-plug-chat-with-tools)

### MCP connectors

Declare an MCP server and the engine handles the authorization, reads the tools
it offers, and runs every call. Your code never holds a token.

Docs: [Connectors](./docs/40-connectors.md)

Examples: [Node](./examples/node-hono-connectors)

### Plugins

Point an agent at an [agent-plugins](https://agent-plugins.org) directory and it
gets that plugin's skills and MCP servers.

Docs: [Plugins](./docs/45-plugins.md)

### Any LLM

The engine calls Anthropic, OpenAI, or OpenRouter with your key. Or your worker
makes the call and the engine never sees a key.

Docs: [LLMs](./docs/70-llms.md)

Examples: [Anthropic](./examples/node-hono-anthropic), [OpenAI](./examples/node-hono-openai), [OpenRouter](./examples/node-hono-openrouter)

### Crash recovery

Every step is saved before it runs. A run continues from where it stopped. The
same message submitted twice runs once.

Docs: [Durability](./docs/200-durability.md)

### Human approval

An agent can stop and wait for a person to approve, then continue. A waiting
agent uses no compute. In Slack this is a button.

Docs: [Interrupts](./docs/100-interrupts.md)

Examples: [Node](./examples/node-hono-tool-approval), [No code](./examples/no-code-mcp-approval)

### Tools that take hours

A tool does not have to answer immediately. Accept the call, do the work on your
own schedule, and report the result later.

Docs: [Async tools](./docs/110-async-tools.md)

### Full chat support

History, editing, regeneration, and branching belong to the engine. A user can
edit an earlier message and go a new direction. The original branch stays.

Docs: [Conversations](./docs/120-conversations.md)

Examples: [Node](./examples/node-hono-plan-mode), [Python](./examples/python-fast-api-plan-mode)

### Works with existing chat UIs

The engine streams AG-UI events, so assistant-ui and CopilotKit connect to it
directly.

Docs: [AG-UI](./docs/140-ag-ui.md)

Examples: [assistant-ui](./examples/node-hono-assistant-ui), [CopilotKit](./examples/node-hono-copilotkit)

### Tools that run in the browser

A tool can run in the user's browser instead of on your server. The run waits
for the browser, then continues.

Docs: [Client-side tools](./docs/150-client-tools.md)

Examples: [Node](./examples/node-hono-client-tool)

### Agent state without a database

The engine stores your agent's state with the conversation. Your code gets it on
every request and writes changes back.

Docs: [Agent state](./docs/90-state.md)

Examples: [Node](./examples/node-hono-plan-mode), [Python](./examples/python-fast-api-plan-mode)

### Sub-agents

An agent can give work to other agents. Each child runs in its own session. The
parent's totals include each child's cost and token use.

Docs: [Sub-agents](./docs/80-sub-agents.md)

Examples: [Node](./examples/node-hono-subagent), [Python](./examples/python-fast-api-subagent)

### Validated tool calls

Give a tool an input and output schema. The engine checks every call against it.

Docs: [Tool calls](./docs/60-tools.md)

Examples: [Node](./examples/node-hono-tools), [Python](./examples/python-fast-api-tools), [Go](./examples/go-chat-with-tools)

### Retries and timeouts

Set a policy on any tool or model call. The engine applies it, and keeps
applying it after a restart.

Docs: [Retries and timeouts](./docs/210-retries.md)

### Host it yourself

Run the engine on your own servers and hold every credential.

Docs: [Self-hosting](./docs/180-self-hosting.md)

## The parts

- **Engine.** Runs the agent loop, in Rust. It calls the model, runs tools,
  saves each step, retries failures, streams events, and supervises sub-agents.
  Use the hosted version at [app.substructure.ai](https://app.substructure.ai),
  run it from the CLI, or embed it in your process.
- **Workers.** Your agent code. It receives a trigger and returns actions. It
  runs in your codebase with your dependencies.
- **Clients.** They send work and stream events back, from your backend or from
  the browser. Slack and AG-UI are clients.
- **CLI.** Set up, deploy, watch, and debug from the terminal. It also runs the
  engine locally.

## Install

```sh
curl -fsSL https://subs.dev/cli.sh | bash
```

Verifies the release checksum and installs to `~/.local/bin`
(`SUBS_INSTALL_DIR` to move it, `SUBS_VERSION` to pin a tag). Or from npm:

```sh
npm i -g @substructure.ai/cli
```

## Docs

Full documentation in [`docs/`](./docs).

## License

[Functional Source License 1.1](./LICENSE), converting to Apache 2.0.
