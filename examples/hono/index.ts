// Mount the worker fetch handler inside a Hono app on Node.
//
// `worker.fetchHandler()` is just (Request) => Promise<Response>, so it
// drops into any Web-fetch-compatible framework. Use this shape when you
// already have a Node HTTP service and want the agent alongside your
// existing routes rather than as a separate process.
//
// Point a Substructure server at this URL:
//   substructure start --dev --port 9000 --worker-url http://localhost:3000/agent

import { serve } from "@hono/node-server";
import Substructure from "@substructure.ai/sdk";
import { Hono } from "hono";

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
    .use(agent.llmToolLoop({ generator: agent.serverGenerate({ model: "anthropic/claude-sonnet-4-6" }) }));

const worker = sub.worker({ agents: [weatherAgent] });
const agentHandler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });

const app = new Hono();

app.get("/health", (c) => c.text("ok"));
app.post("/agent", (c) => agentHandler(c.req.raw));

const port = Number(process.env.PORT ?? 3000);
serve({ fetch: app.fetch, port });
console.log(`Hono worker listening on http://localhost:${port}`);
