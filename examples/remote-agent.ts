import { Substructure, Agent, defineHandler, withJsonState, withLogging, retry, tool } from "@substructure.ai/sdk/substructure";
import { z } from "zod";

const add = tool({
    description: "Add two numbers",
    parameters: z.object({ a: z.number(), b: z.number() }),
    execute: ({ a, b }) => ({ result: a + b }),
    retry: retry().timeout(20).retries(10).backoff(1, 10)
});

const mathAgent = new Agent({
    id: "math-agent",
    description: "Performs math computations",
    llm: {
        client: "openrouter",
        model: "arcee-ai/trinity-large-preview:free",
        retry: retry().timeout(20).retries(10).backoff(1, 10)
    },
    systemPrompt: "You are a math assistant. Compute whatever is asked. Be concise, return only the result.",
    tools: { add },
});

const getWeather = tool({
    description: "Get the current weather for a city. Returns temperature in fahrenheit.",
    parameters: z.object({ city: z.string().describe("City name") }),
    execute: ({ city }) => ({ city, temp_f: city === "San Francisco" ? 62 : 78, condition: "sunny" }),
    retry: retry().timeout(120).retries(3).backoff(1, 10)
});

const weatherAgent = new Agent({
    id: "weather-agent",
    description: "Answers questions about the weather",
    llm: {
        client: "openrouter",
        model: "arcee-ai/trinity-large-preview:free",
        retry: retry().timeout(120).retries(3).backoff(1, 10)
    },
    systemPrompt: "You are a weather assistant. Use tools when appropriate. Be concise.",
    tools: { getWeather },
    subAgents: [mathAgent],
});

const handler = defineHandler()
    .use(withLogging())
    .use(withJsonState())
    .use(weatherAgent)
    .use(mathAgent)

const WORKER_PORT = 4444;
const WORKER_API_KEY = "dev-worker-key";

const sub = new Substructure({
    url: "http://localhost:8080",
    // Same bearer token now protects /api/machine/workers/* and /api/machine/sessions/send.
    workerUrl: `http://localhost:${WORKER_PORT}`,
    workerAuth: { bearerToken: WORKER_API_KEY },
    handler
});

const server = Bun.serve({ port: WORKER_PORT, fetch: sub.fetchHandler() });

const stream = sub.run(
    weatherAgent.id,
    "What is the cube of the sum - the square of the diff of the current temperatures in San Francisco and New York?",
    {
        sessionId: "raw-session-6",
        turnId: "turn-1"
    },
);

for await (const event of stream) {
    if (event.payload.type === "message.new") {
        console.log(event.payload.message.role, event.payload.message.content?.slice(0, 100));
    } else if (event.payload.type === "llm.call.errored") {
        console.log("LLM ERROR:", event.payload.error);
    }
}

const result = await stream.result;
console.log("\nTurn result:", result.artifacts[0]);

await sub.shutdown();
