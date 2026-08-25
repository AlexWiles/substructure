// A chat agent with a client-side tool, served with Hono.
import { serve } from "@hono/node-server";
import { Hono } from "hono";

function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            // The declared config arrives as the proposal; spread it to keep
            // the `llm` and `model` subs.toml names.
            agent: {
                ...proposed.agent,
                system: "For location-based questions, call get_location instead of guessing.",
                tools: [
                    {
                        name: "get_location",
                        description: "Get the user's current city. Only the client can answer this.",
                        handler: "client"
                    }
                ]
            }
        };
    }

    return proposed;
}


const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
