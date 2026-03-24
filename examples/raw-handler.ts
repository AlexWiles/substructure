import { Substructure, contentText } from "@substructure.ai/sdk/substructure";
import type {
    ClientIdentity,
    WorkerAction,
    WorkerDecisionRequestWire,
    LlmTool,
    Message,
    RetryPolicy,
} from "@substructure.ai/sdk/types";

type State = {
    messages: Message[];
};

const WEATHER_AGENT_ID = "weather-agent";
const RETRY: RetryPolicy = {
    timeout_secs: 120,
    max_retries: 3,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

const WEATHER_TOOL: LlmTool = {
    function: {
        name: "get_weather",
        description: "Get the current weather for a city. Returns temperature in fahrenheit.",
        parameters: {
            type: "object",
            properties: {
                city: { type: "string", description: "City name" },
            },
            required: ["city"],
        },
    },
};

const SYSTEM_MESSAGE: Message = {
    role: "system",
    content: "You are a weather assistant. Use tools when appropriate. Be concise.",
};

function decodeState(raw: string): State {
    if (!raw) {
        return { messages: [] };
    }
    return JSON.parse(Buffer.from(raw, "base64").toString("utf-8")) as State;
}

function encodeState(state: State): string {
    return Buffer.from(JSON.stringify(state), "utf-8").toString("base64");
}

function callLlmAction(state: State): WorkerAction {
    return {
        type: "call_llm",
        llm_client: "openrouter",
        request: {
            model: "arcee-ai/trinity-large-preview:free",
            messages: [SYSTEM_MESSAGE, ...state.messages],
            tools: [WEATHER_TOOL],
        },
        stream: false,
        retry: RETRY,
    };
}

function appendMessageForTrigger(state: State, trigger: WorkerDecisionRequestWire["trigger"]): void {
    switch (trigger.type) {
        case "user_message":
        case "llm_response":
            state.messages.push(trigger.message);
            break;
        case "tool_result":
            state.messages.push({
                role: "tool",
                content: trigger.result.content,
                tool_call_id: trigger.result.tool_call_id,
                name: trigger.result.name,
            });
            break;
    }
}

const rawDecision = async (
    request: WorkerDecisionRequestWire,
): Promise<{ actions: WorkerAction[]; state: string }> => {
    const state = decodeState(request.worker_state);
    const trigger = request.trigger;

    appendMessageForTrigger(state, trigger);

    switch (trigger.type) {
        case "user_message":
        case "tool_result":
            return {
                actions: [callLlmAction(state)],
                state: encodeState(state),
            };

        case "llm_response": {
            if (trigger.message.tool_calls?.length) {
                return {
                    actions: trigger.message.tool_calls.map((tc) => ({
                        type: "call_tool",
                        tool_call_id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                        handler: "worker",
                        retry: RETRY,
                    })),
                    state: encodeState(state),
                };
            }

            return {
                actions: [{ type: "done", data: contentText(trigger.message.content) || null }],
                state: encodeState(state),
            };
        }

        case "tool_execute": {
            if (trigger.name !== "get_weather") {
                return {
                    actions: [{
                        type: "return_tool_error",
                        tool_call_id: trigger.tool_call_id,
                        error: `Unknown tool: ${trigger.name}`,
                        retryable: false,
                        attempt: trigger.attempt,
                    }],
                    state: encodeState(state),
                };
            }

            try {
                const args = JSON.parse(trigger.arguments) as { city?: unknown };
                const city = typeof args.city === "string" ? args.city : "";
                if (!city) {
                    throw new Error("city is required");
                }

                const result = {
                    city,
                    temp_f: city === "San Francisco" ? 62 : 78,
                    condition: "sunny",
                };

                return {
                    actions: [{
                        type: "return_tool_result",
                        tool_call_id: trigger.tool_call_id,
                        result: JSON.stringify(result),
                        attempt: trigger.attempt,
                    }],
                    state: encodeState(state),
                };
            } catch (error) {
                return {
                    actions: [{
                        type: "return_tool_error",
                        tool_call_id: trigger.tool_call_id,
                        error: error instanceof Error ? error.message : String(error),
                        retryable: false,
                        attempt: trigger.attempt,
                    }],
                    state: encodeState(state),
                };
            }
        }

        default:
            return {
                actions: [],
                state: encodeState(state),
            };
    }
};

const weatherHandler = {
    agentId: WEATHER_AGENT_ID,
    toDecisionHandler: () => rawDecision,
};

const sub = new Substructure({
    db: "data.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const auth: ClientIdentity = {
    tenant_id: "default",
    sub: "example-user",
};

sub.agent(weatherHandler);

const stream = sub.submit({
    agentId: WEATHER_AGENT_ID,
    payload: {
        type: "message",
        message: {
            role: "user",
            content: "What is the sum of the current temperatures in San Francisco and New York?",
        },
    },
    sessionId: "raw-session-1",
    auth,
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
console.log(result.data);

await sub.shutdown();
