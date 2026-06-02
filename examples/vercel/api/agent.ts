// Vercel serverless function: a single fetch-style handler that exposes
// the worker over HTTP. Deploys with `vercel deploy`; point a hosted
// Substructure backend at https://<your-deploy>/api/agent.
//
// Worker handlers are stateless per request: each invocation runs one
// decision and returns. The backend keeps the session + event log, so
// this function can scale to zero between turns.

import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const { agent } = sub;

const getWeather = agent.tool({
    name: "get_weather",
    description: "Get current weather for a city",
    parameters: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
    },
    execute: (args) => {
        const { city } = JSON.parse(args);
        return JSON.stringify({ city, temp_f: 62, condition: "sunny" });
    },
});

const weatherAgent = agent({ id: "weather" })
    .use(agent.messageHistory("Weather assistant. Be concise."))
    .use(agent.tools([getWeather]))
    .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));

const worker = sub.worker({ agents: [weatherAgent] });

export default {
    fetch: worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET }),
};
