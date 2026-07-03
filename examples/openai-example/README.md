# openai-example

Runs an existing OpenAI Agents SDK agent on Substructure: pass `openaiAgent`
settings directly (or wrap a `new Agent({...})`), wrap the result in
`agent({ name, decide })`, and hand that to `worker([...])`.

```ts
import { agent, worker } from "@substructure.ai/sdk";
import { openaiAgent } from "@substructure.ai/sdk/adapters/openai";

const assistant = agent({
    name: "openai-agent",
    decide: openaiAgent({
        model: "gpt-5-nano",
        instructions: "You are a concise assistant.",
        tools, // your existing @openai/agents tools (zod parameters, execute, …)
    }),
});

const handler = worker([assistant]).fetch({ signingSecret: process.env.SIGNING_SECRET });
```

You can also pass an Agents SDK `Agent` instead of settings, then wrap it in
`agent({ name, decide })`:

```ts
import { Agent } from "@openai/agents";

const assistant = agent({
    name: "openai-agent",
    decide: openaiAgent(
        new Agent({ name: "openai-agent", model: "gpt-5-nano", instructions, tools }),
    ),
});
```

`openaiAgent` returns a Substructure `decide` function (the loop) — wrap it in
`agent({ name, decide })` and pass that to `worker([...])`.
It's a normal Substructure worker agent: it answers
decisions and is agnostic to whether an embedded or remote engine drives it.
Substructure always owns the loop. Each LLM step runs one `responses.create`
call (your tools are passed to the model as definitions only, so the model
returns function calls instead of running them); Substructure executes the tools
as durable steps and iterates. Token deltas stream back through Substructure to
any client (session SSE, AG-UI).

## Run

In one terminal, start a local Substructure server pointed at this worker. The
worker makes the OpenAI calls itself, so the server needs no LLM provider:

```sh
substructure start --dev --port 9000 --worker-url http://localhost:3030
```

In another terminal, start the worker:

```sh
export OPENAI_API_KEY=sk-...
npm install
npm start
```

In a third terminal, submit a turn and watch token deltas stream:

```sh
npm run client
```
