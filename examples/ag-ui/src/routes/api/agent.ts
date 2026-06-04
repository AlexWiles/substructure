// The substructure worker webhook. The ENGINE posts decision requests here
// (server-to-server); this returns the worker's actions. Point the engine at
// it: `substructure start --worker-url https://<app>/api/agent`.
import Substructure from "@substructure.ai/sdk";
import { createFileRoute } from "@tanstack/react-router";

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
    .use(
        agent.llmLoop({
            request: { model: "minimax/minimax-m3" },
            stream: true,
        }),
    );

const worker = sub.worker({ agents: [weatherAgent] });

export const handler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });

export const Route = createFileRoute("/api/agent")({
    server: {
        handlers: {
            POST: ({ request }) => handler(request),
        },
    },
});
