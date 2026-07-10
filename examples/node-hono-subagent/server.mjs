// A chat agent that delegates to a weather sub-agent, served with Hono.
import { serve } from "@hono/node-server";
import { Hono } from "hono";

function assistant({ trigger, proposed }) {
    switch (trigger.type) {
        case "session.start":
            return {
                agent: {
                    model: "claude-haiku-4-5-20251001",
                    stream: true,
                    system: "Delegate weather questions to the weather agent.",
                    sub_agents: [{ id: "weather", description: "Answers weather questions for a city." }]
                }
            };

        default:
            return proposed;
    }
}

function weather({ trigger, proposed }) {
    switch (trigger.type) {
        case "session.start":
            return {
                agent: {
                    model: "claude-haiku-4-5-20251001",
                    stream: true,
                    system: "You report the weather. Look it up, then answer in one line.",
                    tools: [
                        {
                            name: "get_weather",
                            description: "Get the current weather for a city.",
                            input: {
                                type: "object",
                                properties: { city: { type: "string" } },
                                required: ["city"]
                            }
                        }
                    ]
                }
            };

        case "tool.execute": {
            const { city } = trigger.input.value;
            return { actions: [{ type: "tool.result", result: { city, tempF: 68, sky: "clear" } }] };
        }

        default:
            return proposed;
    }
}

// One worker, two agents told apart by `agent_id`.
function decide(req) {
    switch (req.agent_id) {
        case "assistant":
            return assistant(req);

        case "weather":
            return weather(req);
    }
}


const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
