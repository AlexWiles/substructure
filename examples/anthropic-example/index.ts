import { agent, tool, toolLoop, worker } from "@substructure.ai/sdk";
import { anthropicGenerate } from "@substructure.ai/sdk/adapters/anthropic";
import { serve as honoServe } from "@hono/node-server";

const getWeather = tool({
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

const chatAgent = agent({
    name: "anthropic-agent",
    decide: toolLoop({
        llm: anthropicGenerate({ model: "claude-haiku-4-5", max_tokens: 1024 }),
        instructions: "You are a concise assistant.",
        tools: [getWeather],
    }),
});

const handler = worker([chatAgent]).fetch({ signingSecret: process.env.SIGNING_SECRET });

const port = Number(process.env.PORT ?? 3030);
honoServe({ fetch: handler, port });

console.log(`anthropic-example worker listening on http://localhost:${port}`);
