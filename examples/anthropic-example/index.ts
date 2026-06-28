import Substructure from "@substructure.ai/sdk";
import { anthropicGenerate } from "@substructure.ai/sdk/adapters/anthropic";
import { serve } from "@hono/node-server";

const sub = new Substructure();

const getWeather = sub.agent.tool({
    name: "getWeather",
    description: "Get the current weather for a city.",
    parameters: {
        type: "object",
        properties: { city: { type: "string", description: "City name" } },
        required: ["city"],
    },
    execute: (args) => {
        const { city } = JSON.parse(args) as { city: string };
        return `It is 22°C and sunny in ${city}.`;
    },
});

const chatAgent = sub
    .agent({ id: "anthropic-agent" })
    .use(sub.agent.tools([getWeather]))
    .use(
        sub.agent.llm({
            generator: anthropicGenerate({ model: "claude-haiku-4-5", max_tokens: 1024 }),
            instructions: "You are a concise assistant.",
        }),
    );

const worker = sub.worker({ agents: [chatAgent] });

const handler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });

const port = Number(process.env.PORT ?? 3030);
serve({ fetch: handler, port });

console.log(`anthropic-example worker listening on http://localhost:${port}`);
