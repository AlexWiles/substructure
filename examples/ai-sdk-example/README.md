# ai-sdk-example

Runs an existing Vercel AI SDK agent on Substructure: construct a
`ToolLoopAgent` (same settings as the AI SDK's) and drop it into a handler chain.

```ts
import { ToolLoopAgent } from "@substructure.ai/sdk/ai";

const assistant = new ToolLoopAgent({
    model: openrouter("openai/gpt-5-nano"),
    instructions: "You are a concise assistant.",
    tools, // your existing AI SDK ToolSet (zod inputSchema, execute, toModelOutput, …)
});

const chatAgent = agent({ id: "ai-sdk-agent" }).use(assistant);
const worker = sub.worker({ agents: [chatAgent] });
```

Passing the agent to `.use()` composes `messageHistory` + `tools` + `llmLoop`
under the hood. The chain is a normal Substructure worker agent: it answers
decisions and is agnostic to whether an embedded or remote engine drives it.
Substructure
always owns the loop. Each LLM step runs one `streamText` call (your tools are
passed to the model without `execute`, so the model returns tool calls instead of
running them); Substructure executes the tools as durable steps and iterates.
Token deltas stream back through Substructure to any client (session SSE, AG-UI).

Swap providers by changing the `model` line.

## Run

In one terminal, start a local Substructure server pointed at this worker:

```sh
export OPENROUTER_API_KEY=sk-or-...
substructure start --dev --port 9000 --worker-url http://localhost:3030
```

In another terminal, start the worker:

```sh
export OPENROUTER_API_KEY=sk-or-...
pnpm install
pnpm start
```

In a third terminal, submit a turn and watch token deltas stream:

```sh
pnpm client
```
