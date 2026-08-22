// A chat agent with a tool, served with Hono.
import { serve } from "@hono/node-server";
import { Hono } from "hono";

const tools = [
    {
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: () => new Date().toISOString()
    },
    {
        name: "get_current_time_zone",
        description: "Get the user's current timezone",
        // Slow, then fails: what the CLI draws while a call runs, and what it
        // draws when the call does not answer.
        exec: async () => {
            await new Promise((resolve) => setTimeout(resolve, 2000));
            throw new Error("Error fetching timezone");
        }
    }
];

async function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        // The engine will use this agent config to generate proposed actions.
        return {
            agent: {
                ...proposed.agent,
                tools: tools.map(({ name, description }) => ({ name, description })),
                system: "Please answer is slack compatible mrkdwn"
            }
        };
    }

    // Run our tool when the model calls it.
    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        try {
            // `exec` is async: without the await, the result is a promise and
            // the throw never reaches this catch.
            const text = await tool.exec();
            return {
                actions: [
                    { type: "tool.result", result: { content: [{ type: "text", text }] } }
                ]
            };
        } catch (e) {
            return {
                actions: [{ type: "tool.error", error: e.message }]
            };
        }
    }

    // Accept the engine's proposal for every other decision.
    return proposed;
}


const app = new Hono();
app.post("/", async (c) => c.json(await decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
