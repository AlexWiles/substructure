# anthropic-example

Runs Claude on Substructure via the core `@anthropic-ai/sdk` Messages API.

The Anthropic SDK is a low-level client, not an agent framework — there's no
`Agent` type to wrap (unlike the OpenAI Agents and Vercel AI SDK adapters). So
this adapter exposes a single honest primitive: `anthropicGenerate`, a
generator you plug into `llmToolLoop`. You compose the loop yourself.

```ts
import { anthropicGenerate } from "@substructure.ai/sdk/adapters/anthropic";

const getWeather = sub.agent.tool({
    name: "getWeather",
    description: "Get the current weather for a city.",
    parameters: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
    },
    execute: (args) => {
        const { city } = JSON.parse(args);
        return `It is 22°C and sunny in ${city}.`;
    },
});

const chatAgent = sub
    .agent({ id: "anthropic-agent" })
    .use(sub.agent.messageHistory("You are a concise assistant."))
    .use(sub.agent.tools([getWeather]))
    .use(
        sub.agent.llmToolLoop({
            generator: anthropicGenerate({ model: "claude-haiku-4-5", max_tokens: 1024 }),
        }),
    );
```

Tools are declared the normal Substructure way, with `tool()` / `tools()` —
there's nothing Anthropic-specific about a tool definition. The `tools()`
middleware forwards each definition onto `request.tools`, which is where
`anthropicGenerate` reads the model-facing tool list, so you never pass tools to
the generator too.

Substructure always owns the loop. Each LLM step runs one `messages.stream` call
(your tools reach the model as definitions only, so it returns `tool_use` blocks
instead of running them); Substructure executes the tools as durable steps and
iterates. Anthropic has no `system` or `tool` message role — the
`messageHistory` instructions become the top-level `system` param and tool
results go back as `tool_result` blocks in a `user` message. Token deltas stream
back through Substructure to any client (session SSE, AG-UI).

## Run

In one terminal, start a local Substructure server pointed at this worker. The
worker makes the Anthropic calls itself, so the server needs no LLM provider:

```sh
substructure start --dev --port 9000 --worker-url http://localhost:3030
```

In another terminal, start the worker:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
pnpm install
pnpm start
```

In a third terminal, submit a turn and watch token deltas stream:

```sh
pnpm client
```
