import Substructure from "@substructure.ai/sdk";
import { EmbeddedRuntime } from "@substructure.ai/runtime";

const sub = new Substructure();
const { agent } = sub;
const addRetry = {
    timeout_secs: 20,
    max_retries: 10,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

const mathRetry = {
    timeout_secs: 20,
    max_retries: 10,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

const weatherRetry = {
    timeout_secs: 120,
    max_retries: 3,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

const add = agent.tool({
    description: "Add two numbers",
    parameters: {
        type: "object",
        properties: {
            a: { type: "number" },
            b: { type: "number" },
        },
        required: ["a", "b"],
    },
    execute: (args: string) => {
        const { a, b } = JSON.parse(args);
        return { result: a + b };
    },
    retry: addRetry,
});

const mathHandler = agent({ id: "math-agent" })
    .use(agent.logging())
    .use(agent.state())
    .use(agent.messageHistory())
    .use(
        agent.systemMessage("You are a math assistant. Compute whatever is asked. Be concise, return only the result."),
    )
    .use(agent.tools({ add }))
    .use(
        agent.llmLoop({
            request: { model: "arcee-ai/trinity-large-preview:free" },
            llm_client: "openrouter",
            retry: mathRetry,
        }),
    );

const getWeather = agent.tool({
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
    retry: weatherRetry,
});

const weatherHandler = agent({ id: "weather-agent" })
    .use(agent.logging())
    .use(agent.state())
    .use(agent.messageHistory())
    .use(agent.systemMessage("You are a weather assistant. Use tools when appropriate. Be concise."))
    .use(agent.tools({ getWeather }))
    .use(
        agent.subAgents({
            delegates: [mathHandler],
            retry: weatherRetry,
        }),
    )
    .use(
        agent.llmLoop({
            request: { model: "arcee-ai/trinity-large-preview:free" },
            llm_client: "openrouter",
            retry: weatherRetry,
        }),
    );

const WORKER_PORT = 4444;

const backend = sub.backend.client({
    url: "http://localhost:8080",
    apiKey: "dev-worker-key",
});
const clientToken = (
    await backend.mintClientToken({
        tenantId: "default",
        sub: "frontend-user",
        ttlSeconds: 600,
    })
).token;

const frontend = sub.frontend.client({
    url: "http://localhost:8080",
    token: clientToken,
});

const runtime = new EmbeddedRuntime({ db: "remote-agent-example.db" });
const embedded = await sub.embedded({ agents: [weatherHandler, mathHandler], runtime });

const server = Bun.serve({ port: WORKER_PORT, fetch: embedded.fetchHandler() });

await backend.registerWorker({
    transport_type: "http",
    config: { endpoint_url: `http://localhost:${WORKER_PORT}` },
});

const stream = frontend.submit({
    agentId: "weather-agent",
    payload: {
        type: "message",
        message: {
            role: "user",
            content:
                "What is the cube of the sum - the square of the diff of the current temperatures in San Francisco and New York?",
        },
    },
    sessionId: "raw-session-6",
    turnId: "turn-1",
});

for await (const event of stream) {
    if (event.payload.type === "message.new") {
        console.log(event.payload.message.role, event.payload.message.content?.slice(0, 100));
    } else if (event.payload.type === "llm.call.errored") {
        console.log("LLM ERROR:", event.payload.error);
    }
}

const result = await stream.result;
console.log("\nTurn result:", result.data);

await embedded.shutdown();
server.stop();
