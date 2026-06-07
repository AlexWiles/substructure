# openai-example

Runs an existing OpenAI Agents SDK agent on Substructure: wrap a `new Agent({...})`
(or pass `OpenAIAgent` settings directly) and drop it into a handler chain.

```ts
import { OpenAIAgent } from "@substructure.ai/sdk/adapters/openai";
import { Agent, tool } from "@openai/agents";

const assistant = new OpenAIAgent(
    new Agent({
        name: "Assistant",
        model: "gpt-5-nano",
        instructions: "You are a concise assistant.",
        tools, // your existing @openai/agents tools (zod parameters, execute, …)
    }),
);

const chatAgent = agent({ id: "openai-agent" }).use(assistant);
const worker = sub.worker({ agents: [chatAgent] });
```

You can also skip the Agents SDK `Agent` and pass settings directly:

```ts
const assistant = new OpenAIAgent({ model: "gpt-5-nano", instructions, tools });
```

Passing the agent to `.use()` composes `messageHistory` + `tools` + `llmLoop`
under the hood. The chain is a normal Substructure worker agent: it answers
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
pnpm install
pnpm start
```

In a third terminal, submit a turn and watch token deltas stream:

```sh
pnpm client
```
