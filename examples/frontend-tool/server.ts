// Frontend tools example: a browser chat UI where the agent's tools run
// in the *browser*, not the worker. The page asks navigator.geolocation
// for the user's location and mutates document.documentElement for the
// theme — neither possible from a backend tool.
//
// Architecture:
//   browser ──(SSE)──> substructure (:9000) ──(http)──> this worker (:3333)
//
// The worker registers two tools, but their `execute` returns
// `ctx.defer()`. The engine still emits `tool.call.requested` events,
// which the browser sees via the frontend client's SSE stream; it
// executes the tool locally and posts the result back with
// `submitToolCallResult`. The agent resumes as if the tool had returned
// synchronously.

import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import Substructure from "@substructure.ai/sdk";
import { Hono } from "hono";

const sub = new Substructure();
const { agent } = sub;

// `handler: "client"` tells the engine this tool is completed by the
// browser. The worker is never asked to execute it; the SDK's
// `submitToolCallResult` call from the frontend client delivers the
// result. `execute` is required by the type but never runs.
const getUserLocation = agent.tool({
    name: "get_user_location",
    description:
        "Get the user's current latitude and longitude from their browser via geolocation. Requires user permission.",
    parameters: { type: "object", properties: {}, required: [] },
    handler: "client",
    execute: (_args, ctx) => ctx.defer(),
});

const setTheme = agent.tool({
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
    execute: (_args, ctx) => ctx.defer(),
});

const browserAgent = agent({ id: "browser-assistant" })
    .use(agent.tools([getUserLocation, setTheme]))
    .use(
        agent.llm({
            generator: agent.serverGenerate({ model: "anthropic/claude-sonnet-4-6" }),
            stream: true,
            instructions:
                "You are a friendly assistant embedded in a web page. Two tools run in the user's browser: " +
                "`get_user_location` reads the device's GPS (the user is prompted to allow it), and `set_theme` " +
                "repaints the page. Use them when the user asks about where they are or how the page looks. " +
                "Keep replies short and conversational.",
        }),
    );

const worker = sub.worker({ agents: [browserAgent] });
const agentHandler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });

const substructureUrl = process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000";
const backend = sub.backend.client({
    url: substructureUrl,
    apiKey: process.env.SUBSTRUCTURE_API_KEY ?? "dev-worker-key",
});

const app = new Hono();

app.post("/agent", (c) => agentHandler(c.req.raw));

// Mint a short-lived per-user token for the browser. In a real app you'd
// authenticate the request first and bind `identity.id` to the logged-in user.
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
