// An agent whose tools come from a connector.
//
// Compare with node-hono-mcp, where the worker speaks MCP itself. Here the
// worker names a connection and nothing else: it never sees the URL, never
// holds the token, and never runs the tool. The engine does all three.
import { serve } from "@hono/node-server";
import { Hono } from "hono";

function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                ...proposed.agent,
                // "issues" is the server in subs.toml. read_only keeps
                // close_issue away from the model.
                mcp: [{ id: "issues", tools: { read_only: true } }]
            }
        };
    }

    // No tool.execute arrives for a connector tool — the engine runs it. The
    // outcome still shows up here.
    if (trigger.type === "tool.finished") {
        console.log(`${trigger.name} -> ${trigger.ok ? "ok" : trigger.error}`);
    }

    return proposed;
}

const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
