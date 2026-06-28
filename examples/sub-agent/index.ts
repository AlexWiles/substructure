// One agent delegates to another. The parent owns the conversation; the
// child is invoked as a tool, runs its own decision loop with its own
// tools and state, and returns a single result. Failures isolate to the
// child; token + cost usage rolls up to the parent.

import Substructure from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";

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
        return JSON.stringify({ city, temp_f: city === "San Francisco" ? 62 : 78, condition: "sunny" });
    },
});

const weatherAgent = agent({ id: "weather" })
    .use(agent.tools([getWeather]))
    .use(
        agent.llm({
            generator: agent.serverGenerate({ model: "anthropic/claude-sonnet-4-6" }),
            instructions: "Weather assistant. Look up the weather. Be concise.",
        }),
    );

const assistant = agent({ id: "assistant" })
    .use(agent.subAgents({ agents: [weatherAgent] }))
    .use(
        agent.llm({
            generator: agent.serverGenerate({ model: "anthropic/claude-sonnet-4-6" }),
            instructions: "Helpful assistant. Delegate weather questions to the weather agent.",
        }),
    );

const embedded = await SubstructureEmbedded.create({
    agents: [assistant, weatherAgent],
    db: "agent.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: assistant.agentId,
    payload: {
        type: "message",
        message: { role: "user", content: "What's the weather in San Francisco?" },
    },
    identity: { tenant_id: "default", id: "demo" },
});

const { data } = await embedded.turnResult(scope);
console.log(data);

await embedded.shutdown();
