# substructure.ai

Add durable, long running, fault tolerant AI agents to your product.

You own the agent logic and tool execution, Substructure handles all the other stuff like durability, retries, timeouts, budgets and observability.

## How it works

Substructure runs a decision loop. The runtime sends your handler a trigger (a user message, an LLM response, a tool result) and your handler decides what happens next: call an LLM, execute a tool, spawn a sub-agent, or finish. You return actions, and the runtime durably executes them.

Every decision and its result is persisted. If something crashes, the runtime replays the event log and picks up where it left off.

## Install

```sh
npm i -g @substructure.ai/cli
```

## Development

```bash
# Install dependencies
pnpm install

# Run the dashboard
pnpm dev

# Build all packages
pnpm build

# Run tests
pnpm test
```

