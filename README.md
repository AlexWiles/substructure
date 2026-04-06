# substructure.ai

A durable runtime for AI agents. You own the agent logic and tool execution -- Substructure handles state persistence, retries, cost tracking, sub-agent orchestration, and client connections.

The runtime runs a decision loop: it sends your handler a trigger (user message, LLM response, tool result) and your handler returns actions (call an LLM, execute a tool, spawn a sub-agent, finish). Every decision and result is persisted. If something crashes, the runtime replays the event log and picks up where it left off.

## How it works

Three components:

- **Server** -- Orchestration layer written in Rust. Persists state, retries failures, tracks costs, manages sub-agents. Runs standalone or embeds in your process.
- **Workers** -- Your agent logic as an HTTP handler. Receives a decision trigger, returns actions. Runs in your codebase with your dependencies.
- **Clients** -- Submit work and stream events back. Works service-to-service from backends or real-time from browsers with JWT auth.

## Install

```sh
npm i -g @substructure.ai/cli
```

## Usage

Start the server:

```sh
substructure serve --worker-url http://localhost:4444
```

Define an agent with the middleware DSL. Each `.use()` adds a capability -- state, message history, tools, LLM routing:

```typescript
import { defineAgent, state, systemMessage, messageHistory, tools, llmLoop } from "@substructure.ai/sdk/agent";
import { Worker } from "@substructure.ai/sdk";

const agent = defineAgent("weather-agent")
  .use(state())
  .use(systemMessage("You are a helpful weather assistant."))
  .use(messageHistory())
  .use(tools({ getWeather }))
  .use(llmLoop({
    request: { model: "anthropic/claude-sonnet-4" },
    llm_client: "openrouter",
    retry: { timeout_secs: 120, max_retries: 3 },
  }));

const worker = new Worker([agent]);
Bun.serve({ port: 4444, fetch: worker.fetchHandler() });
```

Submit a message and stream events:

```typescript
import { BackendClient } from "@substructure.ai/sdk";

const client = new BackendClient({ url: "http://localhost:9000", apiKey });

const stream = client.submit({
  agentId: "weather-agent",
  payload: {
    type: "message",
    message: { role: "user", content: "What's the weather in SF?" },
  },
  auth: { tenant_id: "default", sub: "user-1" },
});

for await (const event of stream) {
  console.log(event.payload.type);
}

const result = await stream.result;
```

You can also skip the server and run everything in-process with the embedded runtime:

```typescript
import { Substructure } from "@substructure.ai/sdk";
import { EmbeddedRuntime } from "@substructure.ai/runtime";

const runtime = new EmbeddedRuntime({ db: "agent.db" });
const sub = new Substructure({ runtime });
sub.agent(agent);

const stream = sub.submit({
  agentId: "weather-agent",
  payload: {
    type: "message",
    message: { role: "user", content: "What's the weather in SF?" },
  },
  sessionId: crypto.randomUUID(),
  auth: { tenant_id: "default", sub: "user-1" },
  turnId: crypto.randomUUID(),
});

for await (const event of stream) {
  console.log(event.payload.type);
}
```

## Features

- **Durable execution** -- Every decision persisted to an event log. Survives crashes, restarts, and timeouts.
- **Middleware composition** -- State, history, tools, LLM routing, sub-agents are all middleware. Add what you need, replace what you don't.
- **Sub-agents** -- Agents can delegate to child agents. Failures isolate. Costs roll up.
- **Cost and token tracking** -- Per-turn, per-session, per-sub-agent.
- **Multi-tenant** -- Sessions scoped by tenant and user. JWT auth for browser clients.
- **Portable** -- Workers are HTTP handlers. Deploy anywhere: Cloudflare Workers, Fly.io, bare metal, wherever.

## Examples

| Example | Description |
| --- | --- |
| [receipt-extractor](examples/receipt-extractor) | Extract structured data from receipts using tools |
| [coding-agent](examples/coding-agent) | File I/O agent with shell access and HTTP streaming |
| [remote-agent](examples/remote-agent) | Multi-agent delegation with sub-agents |
| [client-action-request](examples/client-action-request) | Custom client actions beyond chat messages |
| [structured-output](examples/structured-output) | JSON schema validated outputs |
| [cloudflare-worker](examples/cloudflare-worker) | Deploy a worker to Cloudflare |
| [fly-deploy](examples/fly-deploy) | Deploy to Fly.io |
| [raw-handler](examples/raw-handler) | Low-level handler without middleware DSL |

## Packages

| Package | Description |
| --- | --- |
| `@substructure.ai/sdk` | TypeScript SDK -- client, worker, and agent middleware |
| `@substructure.ai/runtime` | Embedded Rust runtime via NAPI bindings |
| `@substructure.ai/cli` | CLI for running the server |

## Development

```bash
pnpm install
pnpm dev       # Run the dashboard
pnpm build     # Build all packages
pnpm test      # Run tests
```
