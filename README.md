# substructure.ai

A durable runtime for AI agents. You own the agent logic and tool execution. Substructure handles state persistence, retries, cost tracking, sub-agent orchestration, and client connections.

The runtime executes a decision loop: it sends your handler a trigger (user message, LLM response, tool result) and your handler returns actions (call an LLM, execute a tool, spawn a sub-agent, finish). Every decision and result is persisted. If something crashes, the runtime replays the event log and picks up where it left off.

## How it works

Three components:

- **Server** -- Orchestration layer written in Rust. Persists state, retries failures, tracks costs, manages sub-agents. Runs standalone or embeds in your process.
- **Workers** -- Your agent logic as an HTTP handler. Receives a decision trigger, returns actions. Runs in your codebase with your dependencies.
- **Clients** -- Submit work and stream events back. Works service-to-service from backends or real-time from browsers.

## Install

```sh
npm i -g @substructure.ai/cli
```

## Usage

Start the server in dev mode (auth disabled):

```sh
export OPENROUTER_API_KEY=sk-or-...   # https://openrouter.ai/keys
substructure start --dev --provider openrouter --port 9000 --worker-url http://localhost:4444
```

For production startup, drop `--dev` and set `CLIENT_TOKEN_ISSUER`, `CLIENT_TOKEN_AUDIENCE`, `CLIENT_TOKEN_HS256_SECRET`, `WORKER_API_KEY`, and `ADMIN_API_KEY`. See `substructure start --help`.

Define an agent with the middleware DSL. Each `.use()` adds a capability -- state, message history, tools, LLM routing:

```typescript
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const { agent } = sub;

const getWeather = agent.tool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
  execute: (args: string) => {
    const { city } = JSON.parse(args);
    return { city, temp_f: 62, condition: "sunny" };
  },
});

const weatherAgent = agent({ id: "weather-agent" })
  .use(agent.jsonState())
  .use(agent.systemMessage("You are a helpful weather assistant."))
  .use(agent.messageHistory())
  .use(agent.tools([getWeather]))
  .use(agent.llmLoop({
    request: { model: "anthropic/claude-sonnet-4-5" },
    llm_client: "openrouter",
  }));

Bun.serve({ port: 4444, fetch: sub.worker({ agents: [weatherAgent] }).fetchHandler() });
```

Submit a message and stream events:

```typescript
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const client = sub.backend.client({ url: "http://localhost:9000", apiKey: "your-api-key" });

const stream = client.submit({
  agentId: "weather-agent",
  payload: {
    type: "message",
    message: { role: "user", content: "What's the weather in SF?" },
  },
  identity: { id: "user-1" },
});

for await (const event of stream) {
  console.log(event.payload.type);
}

const result = await stream.result;
```

You can also skip the server and run everything in-process with the embedded runtime:

```typescript
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const instance = await sub.embedded({ agents: [weatherAgent], db: "agent.db" });

const stream = instance.submit({
  agentId: "weather-agent",
  payload: {
    type: "message",
    message: { role: "user", content: "What's the weather in SF?" },
  },
  identity: { tenant_id: "default", id: "user-1" },
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
