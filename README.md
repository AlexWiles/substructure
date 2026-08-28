# substructure.ai
[substructure.ai](https://substructure.ai)

> Pre-1.0: APIs and the wire protocol can change between releases.

`subs` is an agent harness for the cloud.

It runs an unprivileged agent loop with no system access. It uses MCP servers
for tools. It runs locally or as a client and a server.

Declare your agents in a config file. To customize the loop, point an agent at
an HTTP endpoint and answer a webhook.

`subs` handles durability, retries, timeouts, MCP connection management, session state, session branching, AG-UI, Slack connection, LLM calls, subagents, interrupts and more.


To turn a sandbox into an MCP server, see
[mcpd](https://github.com/substructureai/mcpd).

# Get started

## Install the CLI

```sh
curl -fsSL https://subs.dev/cli.sh | bash
```

The CLI is also the engine.

## Run an agent on your machine

Create a `subs.toml`.

```toml title="subs.toml"
name = "example"

[llm.openrouter]
type = "openrouter"

[agent.teammate]
llm = "openrouter"
model = "deepseek/deepseek-v4-flash-0731"
system = "You are a helpful teammate."
```

Set your provider key and talk to the agent.

```sh
export OPENROUTER_API_KEY=sk-or-...
subs chat teammate -c subs.toml
```

## Run the engine as a server

Add a `[serve]` section and a `[remote]` that points at it.

```toml title="subs.toml"
[serve]
port = 9999
auth = false

[remote]
url = "http://localhost:9999"
```

Start the server.

```sh
subs serve -c subs.toml
```

In another terminal, the same chat command now talks to it.

```sh
subs chat teammate -c subs.toml
```

## Run the agent on substructure.ai

Point `[remote]` at the hosted engine instead of your own.

```toml title="subs.toml"
[remote]
url = "https://api.substructure.ai"
```

Create the project from the file, then upload your LLM key.

```sh
subs apply
subs auth llm.openrouter
```

The same chat command now runs the turn on the hosted engine.

```sh
subs chat teammate -c subs.toml
```

## Put the agent in Slack

Say which agent takes a DM and which one answers a mention.

```toml title="subs.toml"
[slack]
dm = "teammate"
mentions = "teammate"
```

Apply the file again, then connect your workspace.

```sh
subs apply
subs slack connect
```

Mention the bot in a channel and it answers in the thread.

## Connect the agent to an MCP server

Declare the server and give it to an agent. Every user of the agent shares one
credential.

```toml title="subs.toml"
[mcp.sentry]
url = "https://mcp.sentry.dev/mcp"

[agent.teammate]
llm = "openrouter"
model = "deepseek/deepseek-v4-flash-0731"
system = "You are a helpful teammate."
mcp = ["mcp.sentry"]
```

Authorize the connection.

```sh
subs auth mcp.sentry
```

## Give each user their own MCP connection

Set `credential = "user"` and each user connects their own account. A
user-scoped connection works only in a one-on-one chat between the agent and
that user.

```toml title="subs.toml"
[mcp.linear]
url = "https://mcp.linear.app/mcp"
credential = "user"

[agent.personal]
llm = "openrouter"
model = "deepseek/deepseek-v4-flash-0731"
system = "Help me with my Linear issues."
mcp = ["mcp.linear"]

[slack]
dm = "personal"
mentions = "teammate"
```

## Control the agent loop with a webhook

Point an agent at a URL and the engine sends every decision for that agent to
your code. This gives you full control of the loop, including how the agent
behaves in Slack.

```toml title="subs.toml"
[agent.teammate]
llm = "openrouter"
model = "deepseek/deepseek-v4-flash-0731"
system = "You are a helpful teammate."
mcp = ["mcp.sentry"]
worker = "https://example.com/agent"
```

Your endpoint reads the engine's proposal and returns it, changing only the
steps you care about. There is no SDK to install.

```typescript title="server.ts"
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import type { DecisionRequest, DecisionResponse } from "./protocol.ts";

function decide({ trigger, proposed }: DecisionRequest): DecisionResponse {
    if (trigger.type === "session.start") {
        return {
            agent: {
                ...proposed.agent,
                tools: [{ name: "current_time", description: "Get the current time" }]
            }
        };
    }

    // Run our tool when the model calls it.
    if (trigger.type === "tool.execute" && trigger.name === "current_time") {
        return { actions: [{ type: "tool.result", result: new Date().toISOString() }] };
    }

    // Accept the engine's proposal for everything else.
    return proposed;
}

const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 });
```

Only the agents that name a worker use one. The rest stay with the engine, in
the same project and the same file.

Full walkthrough in the [quick start](./docs/10-quick-start.md). Docs:
[Workers](./docs/50-workers.md), [Connectors](./docs/40-connectors.md),
[Slack](./docs/130-slack.md), [Local development](./docs/160-local-development.md)

## Features

### Slack

Mention the bot or DM it. The thread is the session. Route different channels to
different agents.

Docs: [Slack](./docs/130-slack.md)

### Terminal chat

`subs chat` holds one session open, streams the reply as it is written, and
turns an approval prompt into a picker. The session is the same kind a Slack
thread is.

Docs: [Chat](./docs/135-chat.md)

Examples: [no-code-chat](./examples/no-code-chat)

### Custom decisions

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

### Chat history, editing, and branching

History, editing, regeneration, and branching belong to the engine. A user can
edit an earlier message and go a new direction. The original branch stays.

Docs: [Conversations](./docs/120-conversations.md)

Examples: [Node](./examples/node-hono-plan-mode), [Python](./examples/python-fast-api-plan-mode)

### Existing chat UIs

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

### Subagents

An agent can give work to other agents. Each child runs in its own session. The
parent's totals include each child's cost and token use.

Docs: [Subagents](./docs/80-subagents.md)

Examples: [Node](./examples/node-hono-subagent), [Python](./examples/python-fast-api-subagent)

### Validated tool calls

Give a tool an input and output schema. The engine checks every call against it.

Docs: [Tool calls](./docs/60-tools.md)

Examples: [Node](./examples/node-hono-tools), [Python](./examples/python-fast-api-tools), [Go](./examples/go-chat-with-tools)

### Retries and timeouts

Set a policy on any tool or model call. The engine applies it, and keeps
applying it after a restart.

Docs: [Retries and timeouts](./docs/210-retries.md)

### Self-hosting

Run the engine on your own servers and hold every credential.

Docs: [Self-hosting](./docs/180-self-hosting.md)

## Parts of the system

- **Engine.** Runs the agent loop, in Rust. It calls the model, runs tools,
  saves each step, retries failures, streams events, and supervises subagents.
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

The script verifies the release checksum and installs to `~/.local/bin`. Set
`SUBS_INSTALL_DIR` to install elsewhere and `SUBS_VERSION` to pin a release. Or
install from npm:

```sh
npm i -g @substructure.ai/cli
```

## Docs

Full documentation in [`docs/`](./docs).

## License

[MIT](./LICENSE)
