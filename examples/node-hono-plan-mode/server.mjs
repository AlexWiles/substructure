// Plan mode: build a plan over several turns, then flip a switch to execute it.
// Uses session state to store the plan across turns
import { serve } from "@hono/node-server";
import { Hono } from "hono";

const planningTools = [
    {
        name: "add_step",
        description: "Append a step to the plan.",
        input: {
            type: "object",
            properties: { text: { type: "string" } },
            required: ["text"]
        },
        exec: (plan, { text }) => plan.push({ text, done: false })
    }
];

const executingTools = [
    {
        name: "complete_step",
        description: "Mark the numbered plan step done.",
        input: {
            type: "object",
            properties: { number: { type: "integer" } },
            required: ["number"]
        },
        exec: (plan, { number }) => (plan[number - 1].done = true)
    }
];

// One profile per mode: planning fills the checklist, executing walks it.
// `llm` and `model` name what substructure.toml declares: a config built
// outside a decision cannot inherit them from the proposal.
const PROFILES = {
    planning: {
        llm: "claude",
        model: "claude-haiku-4-5-20251001",
        system: "You are planning. Break the goal into steps; call add_step for each. Do not execute yet.",
        tools: planningTools
    },
    executing: {
        llm: "claude",
        model: "claude-haiku-4-5-20251001",
        system: "You are executing. Work through each pending step in order; call complete_step with its number when done.",
        tools: executingTools
    }
};

const renderPlan = (plan) =>
    plan.length
        ? plan.map((s, i) => `${i + 1}. [${s.done ? "x" : " "}] ${s.text}`).join("\n")
        : "(empty plan)";

function decide({ trigger, proposed, state }) {
    state = state ?? { mode: "planning", plan: [] };

    if (trigger.type === "session.start") {
        return { agent: PROFILES[state.mode], state };
    }

    // Flip modes with an action, not a message. Entering execution forks a fresh
    // branch seeded with the plan so the executor starts clean.
    if (trigger.type === "client.action" && trigger.name === "set_mode") {
        const mode = trigger.args?.mode;
        state.mode = mode;

        switch (state.mode) {
            case "executing": {
                const seed = { role: "user", content: renderPlan(state.plan) };
                return {
                    agent: PROFILES.executing,
                    state,
                    messages: [seed],
                    actions: [{ type: "llm.call" }]
                };
            }
            case "planning":
                return { agent: PROFILES.planning, state };
            default:
                return { state };
        }
    }

    if (trigger.type === "tool.execute") {
        const tool = PROFILES[state.mode].tools.find((t) => t.name === trigger.name);
        // tools mutate the state
        tool.exec(state.plan, trigger.input.value);
        // return updated state and the tool result
        return {
            state,
            actions: [{ type: "tool.result", result: { content: [{ type: "text", text: renderPlan(state.plan) }] } }],
        };
    }

    // Accept the engine's proposal for every other decision.
    return proposed;
}


const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
