// Vercel serverless function: a single fetch-style handler that exposes
// the worker over HTTP. Deploys with `vercel deploy`; point a hosted
// Substructure backend at https://<your-deploy>/api/agent.
//
// Worker handlers are stateless per request: each invocation runs one
// decision and returns. The backend keeps the session + event log, so
// this function can scale to zero between turns.

import { agent, tool, toolLoop, worker } from "@substructure.ai/sdk";

const getWeather = tool({
    name: "get_weather",
    description: "Get current weather for a city",
    input: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
    },
    execute: (args) => {
        const { city } = JSON.parse(args);
        return JSON.stringify({ city, temp_f: 62, condition: "sunny" });
    },
});

const weatherAgent = agent({
    name: "weather",
    decide: toolLoop({
        llm: { model: "anthropic/claude-sonnet-4-6" },
        instructions: "Weather assistant. Be concise.",
        tools: [getWeather],
    }),
});

export default {
    fetch: worker([weatherAgent]).fetch({ signingSecret: process.env.SIGNING_SECRET }),
};
