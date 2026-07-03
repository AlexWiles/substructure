# ai-sdk-example

Runs an existing Vercel AI SDK agent on Substructure: call `aiSdkAgent` with the
same settings you'd give the AI SDK, wrap it in `agent({ name, decide })`, and
hand that to `worker([...])`.

```ts
import { agent, worker } from "@substructure.ai/sdk";
import { aiSdkAgent } from "@substructure.ai/sdk/adapters/ai";

const assistant = agent({
    name: "ai-sdk-agent",
    decide: aiSdkAgent({
        model: openrouter("openai/gpt-5-nano"),
        instructions: "You are a concise assistant.",
        tools, // your existing AI SDK ToolSet (zod inputSchema, execute, toModelOutput, …)
    }),
});

const handler = worker([assistant]).fetch({ signingSecret: process.env.SIGNING_SECRET });
```

`aiSdkAgent` returns a Substructure `decide` function (the loop) — wrap it in
`agent({ name, decide })` and pass that to `worker([...])`.
It's a normal Substructure worker agent: it answers
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
npm install
npm start
```

In a third terminal, submit a turn and watch token deltas stream:

```sh
npm run client
```
