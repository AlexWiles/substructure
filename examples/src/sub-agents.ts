import Substructure from "@substructure.ai/sdk";
import { randomUUID } from "crypto";

const sub = new Substructure();
const { agent } = sub;

// --- Weather sub-agent with its own tools ---

const getWeather = agent.tool({
    name: "get_weather",
    description: "Get the current weather for a city. Returns temperature in fahrenheit.",
    parameters: {
        type: "object",
        properties: {
            city: { type: "string", description: "City name" },
        },
        required: ["city"],
    },
    execute: (args: string) => {
        const { city } = JSON.parse(args);
        return { city, temp_f: city === "San Francisco" ? 62 : 78, condition: "sunny" };
    },
});

const weatherAgent = agent({ id: "weather-agent" })
    //.use(agent.logging())
    .use(agent.jsonState())
    .use(agent.messageHistory())
    .use(agent.systemMessage("You are a weather assistant. Look up the weather when asked. Be concise."))
    .use(agent.tools([getWeather]))
    .use(
        agent.llmLoop({
            request: { model: "anthropic/claude-sonnet-4" },
            llm_client: "openrouter",
        }),
    );

// --- Parent assistant that delegates weather questions to the sub-agent ---

const assistant = agent({ id: "assistant" })
    //.use(agent.logging())
    .use(agent.jsonState())
    .use(agent.messageHistory())
    .use(agent.systemMessage("You are a helpful assistant. Use the weather-agent when the user asks about weather."))
    .use(agent.subAgents({ agents: [weatherAgent] }))
    .use(
        agent.llmLoop({
            request: { model: "anthropic/claude-sonnet-4" },
            llm_client: "openrouter",
        }),
    );

const embedded = await sub.embedded({
    agents: [assistant, weatherAgent],
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const sessionId = randomUUID();
const identity = { tenant_id: "default", id: "example-user" };

async function turn(message: string) {
    console.log(`\n> ${message}`);
    const stream = embedded.submitAndListen({
        agentId: assistant.agentId,
        payload: { type: "message", message: { role: "user", content: message } },
        sessionId,
        identity,
        turnId: randomUUID(),
    });
    for await (const event of stream) {
        if (
            event.payload.type === "message.new" &&
            event.payload.message.role === "assistant" &&
            event.payload.message.content
        ) {
            console.log(event.payload.message.content);
        } else if (event.payload.type === "llm.call.errored") {
            console.log("LLM ERROR:", event.payload.error);
        }
    }
}

await turn("What's the weather in San Francisco?");
await turn("How about New York?");

await embedded.shutdown();
