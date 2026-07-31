// A chat agent with tools, served with Hono. Types are generated from
// schemas/protocol.schema.json (see README).
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import type { DecisionRequest, DecisionResponse } from "./protocol.ts";

const tools = [
    {
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: () => new Date().toISOString(),
    },
    {
        name: "get_current_time_zone",
        description: "Get the user's current timezone",
        exec: () => Intl.DateTimeFormat().resolvedOptions().timeZone,
    },
];

function decide({ trigger, proposed }: DecisionRequest): DecisionResponse {
    if (trigger.type === "session.start") {
        // The engine will use this agent config to generate proposed actions.
        return {
            agent: {
                ...proposed.agent!,
                stream: true,
                tools: tools.map(({ name, description }) => ({ name, description })),
            },
        };
    }

    // Run our tool when the model calls it.
    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        return { actions: [{ type: "tool.result", result: tool?.exec() }] };
    }

    // Accept the engine's proposal for every other decision.
    return proposed;
}

const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () => console.log("worker listening on http://localhost:4444"));
