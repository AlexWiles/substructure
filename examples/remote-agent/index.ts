import {
    Substructure,
    defineAgent,
    withState,
    withLogging,
    tool,
    withConversation,
    withSystemMessage,
    withTools,
    withCallLLM,
    withSubAgents,
} from "@substructure.ai/sdk/substructure";
import { BackendClient, FrontendClient } from "@substructure.ai/sdk";
import { EmbeddedRuntime } from "@substructure.ai/runtime";
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

const add = tool({
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

const mathHandler = defineAgent("math-agent")
    .use(withLogging())
    .use(withState())
    .use(withConversation())
    .use(withSystemMessage(() => "You are a math assistant. Compute whatever is asked. Be concise, return only the result."))
    .use(withTools(() => ({ add })))
    .use(withCallLLM((_state) => ({
        request: { model: "arcee-ai/trinity-large-preview:free" },
        llm_client: "openrouter",
        retry: mathRetry,
    })));

const getWeather = tool({
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

const weatherHandler = defineAgent("weather-agent")
    .use(withLogging())
    .use(withState())
    .use(withConversation())
    .use(withSystemMessage(() => "You are a weather assistant. Use tools when appropriate. Be concise."))
    .use(withTools(() => ({ getWeather })))
    .use(withSubAgents({
        delegates: [mathHandler],
        retry: weatherRetry,
    }))
    .use(withCallLLM((_state) => ({
        request: { model: "arcee-ai/trinity-large-preview:free" },
        llm_client: "openrouter",
        retry: weatherRetry,
    })));

const WORKER_PORT = 4444;

const backend = new BackendClient({
    url: "http://localhost:8080",
    apiKey: "dev-worker-key",
});
const clientToken = (await backend.mintClientToken({
    tenantId: "default",
    sub: "frontend-user",
    ttlSeconds: 600,
})).token;

const frontend = new FrontendClient({
    url: "http://localhost:8080",
    token: clientToken,
});

const runtime = new EmbeddedRuntime({ db: "remote-agent-example.db" });
const sub = new Substructure({ runtime });

sub.agent(weatherHandler);
sub.agent(mathHandler);

const server = Bun.serve({ port: WORKER_PORT, fetch: sub.fetchHandler() });

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
            content: "What is the cube of the sum - the square of the diff of the current temperatures in San Francisco and New York?",
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

await sub.shutdown();
server.stop();
