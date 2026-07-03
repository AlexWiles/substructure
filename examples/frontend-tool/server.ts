// Frontend tools example: the agent's tools run in the browser (geolocation, theme), not the worker.
// Registered with `handler: "client"` and no `execute`; the browser runs each call and returns it via `settleEffect`.

import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import Substructure, { agent, tool, toolLoop, worker } from "@substructure.ai/sdk";
import { Hono } from "hono";

const sub = new Substructure();

// `handler: "client"`: the browser completes this tool via `settleEffect`, so no `execute` is needed.
const getUserLocation = tool({
    name: "get_user_location",
    description:
        "Get the user's current latitude and longitude from their browser via geolocation. Requires user permission.",
    parameters: { type: "object", properties: {}, required: [] },
    handler: "client",
});

const setTheme = tool({
    name: "set_theme",
    description: "Set the page's background and accent colors. Use valid CSS colors (hex, rgb, hsl, or named).",
    parameters: {
        type: "object",
        properties: {
            background: { type: "string", description: "Page background color." },
            accent: { type: "string", description: "Accent color used for the user chat bubble and button." },
        },
        required: ["background", "accent"],
    },
    handler: "client",
});

const browserAgent = agent({
    name: "browser-assistant",
    decide: toolLoop({
        llm: { model: "anthropic/claude-sonnet-4-6", stream: true },
        instructions:
            "You are a friendly assistant embedded in a web page. Two tools run in the user's browser: " +
            "`get_user_location` reads the device's GPS (the user is prompted to allow it), and `set_theme` " +
            "repaints the page. Use them when the user asks about where they are or how the page looks. " +
            "Keep replies short and conversational.",
        tools: [getUserLocation, setTheme],
    }),
});

const agentHandler = worker([browserAgent]).fetch({ signingSecret: process.env.SIGNING_SECRET });

const substructureUrl = process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000";
const backend = sub.backend.client({
    url: substructureUrl,
    apiKey: process.env.SUBSTRUCTURE_API_KEY ?? "dev-worker-key",
});

const app = new Hono();

app.post("/agent", (c) => agentHandler(c.req.raw));

// Mint a short-lived per-user token. In a real app, authenticate first and bind `identity.id` to the user.
app.post("/token", async (c) => {
    const { token, expiresAt } = await backend.mintClientToken({
        identity: { id: "demo-user" },
        ttlSeconds: 60 * 15,
    });
    return c.json({ token, expiresAt, substructureUrl });
});

app.use("/*", serveStatic({ root: "./public" }));

const port = Number(process.env.PORT ?? 3333);
serve({ fetch: app.fetch, port });
console.log(`open http://localhost:${port}`);
