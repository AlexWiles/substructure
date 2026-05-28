# substructure.ai

Substructure is an open-source engine for building durable, long-running AI agents using just an HTTP endpoint hosted on your infrastructure, in your code.

Substructure drives the agentic loop, handling retries, sub-agent supervision, llm calls, real-time event streaming and more. Tool execution and agent decisions live in your codebase on your infrastructure.

## How it works

- **Server:** The engine that drives the agent loop, written in Rust. It can be run locally on your machine, embedded in process, or as a cloud hosted version available at [https://app.substructure.ai](https://app.substructure.ai). The server drives the loop, handles durability, retries, llm calls, realtime streaming, subagent supervision and more.
- **Workers:** Your agent logic. Receives a decision trigger, returns actions. Runs in your codebase with your dependencies. Can be an HTTP endpoint for use with the cloud/local server, or a callback passed to embedded substructure.
- **Clients:**  Submit work and stream events back. We have support for both backend-to-backend as well as browser based clients.
- **CLI:** Substructure comes with a CLI to help you provision, observe, and debug from the terminal. You can also start a local server.
- **SDK:** We provide a TypeScript SDK for building agents and setting up your worker with a just a few lines of code. It also includes server-to-server and browser clients.

## Why Substructure
- **Write agent logic, not agent infrastructure.** The event log, retries, timeouts, streaming, etc. are Substructure's job.
- **Add agents to the codebase you already have.** Workers are plain HTTP handlers. You can drop them into your app, deploy them to your infrastructure.
- **Ship to serverless.** Stateless workers means they can be deployed to any serverless platform. There are no long running processes.

## Install

The CLI is available at:
```sh
npm i -g @substructure.ai/cli
```

The SDK is available at:
```sh
npm i @substructure.ai/sdk
```


## A Quick Example

This walks through running an agent against [Substructure Cloud](https://app.substructure.ai). Three steps: define a worker, point the cloud at it, submit a turn.

**1. Define an agent and serve it as a worker.** Workers are plain HTTP handlers; deploy this anywhere with a public URL (Cloudflare, Vercel, Fly, your own infra). See [`examples/`](./examples) for full deployments.

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
  }));

const worker = sub.worker({ agents: [weatherAgent] });

export default {
  fetch: worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET }),
};
```

**2. Provision Substructure Cloud and point it at your deployed worker.**

```sh
substructure cloud login

substructure cloud link                                          # link this directory to an org & app

substructure cloud webhook set https://your-worker.example.com   # tell the substructure where to call

# Prints out the signing secret for the webhook. Copy into your worker's env as SIGNING_SECRET:
substructure cloud webhook secret

# Mint an API key for your client:
export SUBSTRUCTURE_API_KEY=$(substructure cloud keys create demo)
```

**3. Submit a turn from your client.**

```typescript
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const client = sub.backend.client({
  url: "https://api.substructure.ai",
  apiKey: process.env.SUBSTRUCTURE_API_KEY!,
});

const scope = await client.startTurn({
  agentId: "weather-agent",
  payload: {
    type: "message",
    message: { role: "user", content: "What's the weather in SF?" },
  },
  identity: { id: "user-1" },
});

const { data } = await client.turnResult(scope);
console.log(data);
```

## Docs

Full documentation in [`docs/`](./docs).

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
