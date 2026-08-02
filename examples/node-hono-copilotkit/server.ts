// A chat agent and its web UI, served with Hono. Three jobs:
//   POST /agent  — the substructure worker; the engine calls it to make decisions
//   GET  /token  — mint a short-lived client token for the browser
//   GET  /*      — serve the built CopilotKit frontend (web/dist)
// Protocol types in `protocol.ts` are generated from the JSON Schema (see README).
import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import type { DecisionRequest, DecisionResponse } from "./protocol.ts";

const AGENT_ID = "chat-agent";
const ENGINE_URL = process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000";
const ENGINE_PUBLIC_URL = process.env.SUBSTRUCTURE_PUBLIC_URL ?? ENGINE_URL;
const API_KEY = process.env.SUBSTRUCTURE_API_KEY ?? "dev"; // `subs serve --dev` ignores it

// Worker-executed tools: the engine sends `tool.execute` here and we run them.
const tools = [
    {
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: () => new Date().toISOString(),
    },
];

function decide({ trigger, proposed }: DecisionRequest): DecisionResponse {
    // On session start, declare the agent the engine should run. We list only our
    // worker tools — the browser declares its own client tools (get_timezone; see
    // web/chat.tsx) on each run, and the engine layers them onto the proposal.
    if (trigger.type === "session.start") {
        return {
            agent: {
                ...proposed.agent!,
                system:
                    "You are a concise, friendly assistant. Use get_current_time for the current " +
                    "UTC time and get_timezone for the user's time zone.",
                tools: tools.map(({ name, description }) => ({ name, description })),
            },
        };
    }

    // Run our tool when the model calls it.
    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        return { actions: [{ type: "tool.result", result: tool?.exec() }] };
    }

    // Accept the engine's proposal for every other decision — including
    // `client.messages`, whose proposal already carries the browser's client tools.
    return proposed;
}

const app = new Hono();

// The worker: the engine posts decisions here.
app.post("/agent", async (c) => c.json(decide(await c.req.json())));

// Mint a short-lived token so the browser can reach the engine's AG-UI endpoint.
// In a real app, authenticate the user first and bind identity.id to them.
app.get("/token", async (c) => {
    const res = await fetch(`${ENGINE_URL}/api/machine/client-tokens`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${API_KEY}` },
        body: JSON.stringify({ identity: { id: "demo-user" }, ttl_seconds: 900 }),
    });
    const { token } = (await res.json()) as { token: string };
    return c.json({ token, url: ENGINE_PUBLIC_URL, agentId: AGENT_ID });
});

// Serve the built frontend (index.html + assets).
app.use("/*", serveStatic({ root: "./web/dist" }));

serve({ fetch: app.fetch, port: 4444 }, () => console.log("app listening on http://localhost:4444"));
