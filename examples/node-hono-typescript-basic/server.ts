// A complete chat agent, served with Hono. Types are generated from
// schemas/protocol.schema.json (see README).
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import type { DecisionRequest, DecisionResponseClass } from "./protocol.ts";

function decide({ trigger, proposed }: DecisionRequest): DecisionResponseClass | null {
    if (trigger.type === "session.start") {
        // The engine will use this agent config to generate proposed actions.
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true,
            },
        };
    }

    // Accept the engine's proposal for every other decision.
    return proposed ?? null;
}

const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () => console.log("worker listening on http://localhost:4444"));
