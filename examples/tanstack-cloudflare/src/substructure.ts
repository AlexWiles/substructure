// Shared substructure setup. Imported only by server routes (never the
// client bundle). Runs fine on the Cloudflare Workers runtime — the worker is
// a plain Web fetch handler, and the backend client is just HTTP.

import Substructure from "@substructure.ai/sdk";

export const AGENT_ID = "weather-agent";

const sub = new Substructure();
const { agent } = sub;

const getWeather = agent.tool({
    name: "get_weather",
    description: "Get the current weather for a city.",
    parameters: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
    },
    execute: (args: string) => {
        const { city } = JSON.parse(args);
        return JSON.stringify({ city, temp_f: 62, condition: "sunny" });
    },
});

// A frontend (client-handled) tool. The worker only DECLARES it — `handler:
// "client"` + `ctx.defer()` means the engine suspends the turn and waits for the
// browser to execute it and submit the result. The matching executor lives in
// each React chat client (src/components/*-chat.tsx) and updates the on-screen
// color mixer; the engine never runs it.
const setColor = agent.tool({
    name: "set_color",
    description:
        "Set the color shown in the user's on-screen color mixer. Give red, green, and " +
        "blue as integers from 0 to 255. Runs in the user's browser.",
    parameters: {
        type: "object",
        properties: {
            red: { type: "integer", minimum: 0, maximum: 255 },
            green: { type: "integer", minimum: 0, maximum: 255 },
            blue: { type: "integer", minimum: 0, maximum: 255 },
        },
        required: ["red", "green", "blue"],
    },
    handler: "client",
    execute: (_args: string, ctx) => ctx.defer(),
});

// The read counterpart to set_color: returns whatever the user currently has in
// their on-screen mixer (which they can drag themselves). Also browser-run.
const getColor = agent.tool({
    name: "get_color",
    description:
        "Read the color the user currently has set in their on-screen color mixer. " +
        "Returns red, green, and blue (0–255) plus the hex. Runs in the user's browser.",
    parameters: { type: "object", properties: {} },
    handler: "client",
    execute: (_args: string, ctx) => ctx.defer(),
});

const weatherAgent = agent({ id: AGENT_ID })
    .use(
        agent.messageHistory(
            "You are a concise, friendly assistant. Use get_weather for weather. " +
                "When the user asks to set, change, or mix a color — e.g. “make it sunset " +
                "orange”, “a calming teal”, “warmer”, “darker” — call set_color with " +
                "red/green/blue (0–255) to update their on-screen color mixer. When they " +
                "ask what color is showing (e.g. “what color is this?”, “what's the hex?”), " +
                "call get_color to read it.",
        ),
    )
    .use(agent.tools([getWeather, setColor, getColor]))
    // `reasoning` turns on the model's thinking; it streams to the clients as
    // REASONING_* events. Use `max_tokens` for Anthropic/Gemini/Qwen, or
    // `{ effort: "medium" }` for OpenAI/Grok (the two are mutually exclusive).
    .use(
        agent.llmLoop({
            request: { model: "minimax/minimax-m3" },
            stream: true,
        }),
    );

const worker = sub.worker({ agents: [weatherAgent] });

/** Handle a decision webhook from the engine. Reads SIGNING_SECRET per call so
 *  Workers secrets are resolved at request time. */
export function handleAgentRequest(request: Request): Promise<Response> {
    const handler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });
    return handler(request);
}

/** Mint a short-lived, identity-locked client token for the browser. In a real
 *  app, authenticate the request first and bind identity.id to that user. */
export async function mintBrowserToken(): Promise<{ token: string; substructureUrl: string; agentId: string }> {
    const backend = sub.backend.client({
        url: process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000",
        apiKey: process.env.SUBSTRUCTURE_API_KEY ?? "dev-worker-key",
    });
    const { token } = await backend.mintClientToken({
        identity: { id: "demo-user" },
        ttlSeconds: 60 * 15,
    });
    // The browser talks to the engine directly, so it needs the engine's
    // public URL (CORS-open). Falls back to the server-side URL for local dev.
    const substructureUrl =
        process.env.SUBSTRUCTURE_PUBLIC_URL ?? process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000";
    return { token, substructureUrl, agentId: AGENT_ID };
}
