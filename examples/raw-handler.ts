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
import type { Message } from "@substructure.ai/sdk/types";
import { z } from "zod";

const mathRetry = {
    timeout_secs: 120,
    max_retries: 3,
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
    parameters: z.object({ a: z.number(), b: z.number() }),
    execute: ({ a, b }) => ({ result: a + b }),
});

const getWeather = tool({
    description: "Get the current weather for a city. Returns temperature in fahrenheit.",
    parameters: z.object({ city: z.string().describe("City name") }),
    execute: ({ city }) => ({
        city,
        temp_f: city === "San Francisco" ? 62 : 78,
        condition: "sunny"
    }),
});

const MATH_AGENT_ID = "math-agent";
const WEATHER_AGENT_ID = "weather-agent";

type State = {
    messages: Message[];
    subAgentTracker: Record<string, { toolCallId: string; name: string }>;
};

const messagesAdapter = {
    getMessages: (state: State) => state.messages,
    setMessages: (state: State, messages: Message[]) => {
        state.messages = messages;
    },
};

const subAgentTrackerAdapter = {
    getSubAgentTracker: (state: State) => state.subAgentTracker,
    setSubAgentTracker: (state: State, tracker: State["subAgentTracker"]) => {
        state.subAgentTracker = tracker;
    },
};

const mathHandler = defineAgent(MATH_AGENT_ID)
    .use(withLogging())
    .use(withState<State>({ messages: [], subAgentTracker: {} }))
    .use(withConversation<State>(messagesAdapter))
    .use(withSystemMessage<State>("You are a math assistant. Compute whatever is asked. Be concise, return only the result."))
    .use(withTools<State>({ add }))
    .use(withCallLLM<State>((state) => ({
        request: {
            model: "openrouter/hunter-alpha",
        },
        llm_client: "openrouter",
        retry: mathRetry,
    })));

const weatherHandler = defineAgent(WEATHER_AGENT_ID)
    .use(withLogging())
    .use(withState<State>({ messages: [], subAgentTracker: {} }))
    .use(withConversation<State>(messagesAdapter))
    .use(withSystemMessage<State>("You are a weather assistant. Use tools when appropriate. Be concise."))
    .use(withTools<State>({ get_weather: getWeather }))
    .use(withSubAgents<State>({
        delegates: [mathHandler],
        tracker: subAgentTrackerAdapter,
    }))
    .use(withCallLLM<State>((state) => ({
        request: {
            model: "openrouter/hunter-alpha",
        },
        llm_client: "openrouter",
        retry: weatherRetry,
    })));

const sub = new Substructure({
    db: "data.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

sub.agent(weatherHandler);
sub.agent(mathHandler);

const stream = sub.run(
    WEATHER_AGENT_ID,
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
console.log(result.data);

await sub.shutdown();
