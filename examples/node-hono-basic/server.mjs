// A complete chat agent. No SDK — the decision protocol by hand, served with Hono.
// The engine POSTs a decision request; this returns the next actions.
//
// The worker accepts every decision the engine has a default for (`proposed`)
// and authors only the one that is genuinely its own: the LLM request — the
// agent's identity. Everything else — model replies and model failures — flows
// through the proposed-first line at the top.
//
// Point a local Substructure server at it:
//   subs serve --dev --provider anthropic --worker-url http://localhost:4444
import { serve } from "@hono/node-server";
import { Hono } from "hono";

function decide({ trigger: t, proposed }) {
    // The engine proposes actions for a default agent tool loop
    if (proposed) return proposed;

    // The client sent an updated message list
    if (t.type === "client.messages") {
        return {
            // echo back the messages to update the session state
            messages: t.messages,
            // call the LLM with the messages
            actions: [{
                type: "llm.call", stream: true,
                request: {
                    model: "claude-haiku-4-5-20251001",
                    messages: t.messages
                }
            }],
        };
    }

    return { actions: [] };
}


const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

const PORT = 4444;
serve({ fetch: app.fetch, port: PORT }, () =>
    console.log(`worker listening on http://localhost:${PORT}`));
