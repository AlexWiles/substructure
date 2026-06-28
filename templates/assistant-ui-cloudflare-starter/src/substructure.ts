// Server-only Substructure setup: the agent definition, the worker webhook the
// engine calls (server-to-server), and the browser-token minter. Never imported
// into the client bundle.

import Substructure from "@substructure.ai/sdk";

export const AGENT_ID = "assistant";

const sub = new Substructure();
const { agent } = sub;

// A server-side tool: the engine runs `execute` and feeds the result back to
// the model.
const getCurrentTime = agent.tool({
    name: "get_current_time",
    description: "Get the current date and time.",
    parameters: { type: "object", properties: {} },
    execute: () => new Date().toString(),
});

// A client-side (frontend) tool. `handler: "client"` routes execution to the
// browser: the engine advertises the tool to the model, emits the call, and
// waits for the browser to post the result — so no server-side `execute` is
// needed. The browser executor lives in the chat client (src/routes/index.tsx).
// The engine ignores AG-UI's client-declared `tools`, so the tool is declared
// here for the model to see.
const browserAlert = agent.tool({
    name: "browser_alert",
    description: "Display a native browser alert dialog to the user. Runs in the user's browser.",
    parameters: {
        type: "object",
        properties: {
            message: { type: "string", description: "Text to display inside the alert dialog." },
        },
        required: ["message"],
    },
    handler: "client",
});

const assistant = agent({ id: AGENT_ID })
    .use(agent.tools([getCurrentTime, browserAlert]))
    .use(
        agent.llm({
            generator: agent.serverGenerate({ model: "anthropic/claude-sonnet-4-6" }),
            stream: true,
            instructions:
                "You are a concise, friendly assistant. Use get_current_time when asked about the " +
                "current date or time, and browser_alert to pop a native alert in the user's browser.",
        }),
    );

const worker = sub.worker({ agents: [assistant] });

export const substructureHandler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });

// Mint a short-lived, identity-locked client token for the browser. In a real
// app, authenticate the user first and bind `identity.id` to them.
export async function mintBrowserToken(): Promise<{ token: string; substructureUrl: string; agentId: string }> {
    const backend = sub.backend.client({
        url: process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000",
        apiKey: process.env.SUBSTRUCTURE_API_KEY ?? "dev-worker-key",
    });
    const { token } = await backend.mintClientToken({
        identity: { id: "demo-user" },
        ttlSeconds: 60 * 15,
    });
    // The browser streams from the engine directly, so it needs the public URL.
    const substructureUrl =
        process.env.SUBSTRUCTURE_PUBLIC_URL ?? process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000";
    return { token, substructureUrl, agentId: AGENT_ID };
}
