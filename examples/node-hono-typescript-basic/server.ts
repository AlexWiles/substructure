// A complete chat agent, served with Hono. Types are generated from
// schemas/protocol.schema.json (see README).
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import type { DecisionRequest, DecisionResponse } from "./protocol.ts";

// The whole worker: accept the engine's proposal for every decision. The agent
// is declared in subs.toml, and arrives as the `session.start`
// proposal, so there is nothing to author until you want to override something.
function decide({ proposed }: DecisionRequest): DecisionResponse {
    return proposed;
}

const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () => console.log("worker listening on http://localhost:4444"));
