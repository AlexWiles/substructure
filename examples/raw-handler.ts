import { Substructure, Agent, defineHandler, withJsonState, withLogging, retry, tool } from "@substructure.ai/sdk/substructure";
import { z } from "zod";

const add = tool({
    description: "Add two numbers",
    parameters: z.object({ a: z.number(), b: z.number() }),
    execute: ({ a, b }) => ({ result: a + b }),
});

const mathAgent = new Agent({
    id: "math-agent",
    description: "Performs math computations",
    llm: { model: "openrouter/hunter-alpha", client: "openrouter", retry: retry().timeout(120).retries(3).backoff(1, 10) },
    systemPrompt: "You are a math assistant. Compute whatever is asked. Be concise, return only the result.",
    tools: { add },
});

const getWeather = tool({
    description: "Get the current weather for a city. Returns temperature in fahrenheit.",
    parameters: z.object({ city: z.string().describe("City name") }),
    execute: ({ city }) => ({ city, temp_f: city === "San Francisco" ? 62 : 78, condition: "sunny" }),
});

const weatherAgent = new Agent({
    id: "weather-agent",
    description: "Answers questions about the weather",
    llm: { model: "openrouter/hunter-alpha", client: "openrouter", retry: retry().timeout(120).retries(3).backoff(1, 10) },
    systemPrompt: "You are a weather assistant. Use tools when appropriate. Be concise.",
    tools: { get_weather: getWeather },
    subAgents: [mathAgent],
});

const handler = defineHandler()
    .use(withLogging())
    .use(withJsonState())
    .use(weatherAgent)
    .use(mathAgent)

const sub = new Substructure({
    db: "data.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
    handler,
});

const stream = sub.run(
    "weather-agent",
    "What is the sum of the current temperatures in San Francisco and New York?",
    { sessionId: "raw-session-1", turnId: "turn-1" },
);

for await (const event of stream) {
    if (event.payload.type === "message.new") {
        console.log(event.payload.message.role, event.payload.message.content?.slice(0, 100));
    } else if (event.payload.type === "llm.call.errored") {
        console.log("LLM ERROR:", event.payload.error);
    }
}

const result = await stream.result;
console.log(result.artifacts[0].parts);

await sub.shutdown();
