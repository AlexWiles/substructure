// Plan mode: build a plan over several turns, then flip a switch to execute it.
// Entering execution branches a fresh thread so the executor sees only the plan.
// The agent reads its mode from state and picks the model, prompt, and tools per mode.

import { agent, type Llm, tool, toolLoop } from "@substructure.ai/sdk";
import type { DecisionTrigger } from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";

// ── Domain ──────────────────────────────────────────────────────────────────

type Mode = "planning" | "executing";
type Step = { id: string; text: string; done: boolean };
type Plan = { goal: string; steps: Step[]; nextId: number };
type State = { mode: Mode; plan: Plan };

const initialPlan = (): Plan => ({ goal: "", steps: [], nextId: 1 });

// ── Tools, per mode ───────────────────────────────────────────────────────────
// Built fresh each decision so `execute` closes over the live plan.

function planTools(state: State) {
    const plan = state.plan;
    const planning = [
        tool({
            name: "set_goal",
            description: "Set or replace the overall goal the plan is working toward.",
            parameters: { type: "object", properties: { goal: { type: "string" } }, required: ["goal"] },
            execute: (args) => {
                plan.goal = JSON.parse(args).goal;
                return JSON.stringify(plan);
            },
        }),
        tool({
            name: "add_step",
            description: "Append a new step to the plan. Returns the created step.",
            parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
            execute: (args) => {
                const step: Step = { id: `s${plan.nextId++}`, text: JSON.parse(args).text, done: false };
                plan.steps.push(step);
                return JSON.stringify(step);
            },
        }),
        tool({
            name: "update_step",
            description: "Rewrite the text of an existing step by id.",
            parameters: {
                type: "object",
                properties: { id: { type: "string" }, text: { type: "string" } },
                required: ["id", "text"],
            },
            execute: (args) => {
                const { id, text } = JSON.parse(args);
                const step = plan.steps.find((s) => s.id === id);
                if (!step) throw new Error(`unknown step: ${id}`);
                step.text = text;
                return JSON.stringify(step);
            },
        }),
        tool({
            name: "remove_step",
            description: "Remove a step from the plan by id.",
            parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
            execute: (args) => {
                const idx = plan.steps.findIndex((s) => s.id === JSON.parse(args).id);
                if (idx === -1) throw new Error(`unknown step: ${JSON.parse(args).id}`);
                return JSON.stringify(plan.steps.splice(idx, 1)[0]);
            },
        }),
    ];
    const executing = [
        tool({
            name: "complete_step",
            description: "Mark a plan step as completed with a one-line summary of what was done.",
            parameters: {
                type: "object",
                properties: {
                    id: { type: "string" },
                    note: { type: "string", description: "Short summary of how the step was completed." },
                },
                required: ["id", "note"],
            },
            execute: (args) => {
                const { id, note } = JSON.parse(args);
                const step = plan.steps.find((s) => s.id === id);
                if (!step) throw new Error(`unknown step: ${id}`);
                step.done = true;
                return JSON.stringify({ id: step.id, text: step.text, note });
            },
        }),
    ];
    return state.mode === "planning" ? planning : executing;
}

// ── Prompts ───────────────────────────────────────────────────────────────────

const renderPlan = (plan: Plan) => {
    const goalLine = `Goal: ${plan.goal || "(unset)"}`;
    if (plan.steps.length === 0) return `${goalLine}\n  (no steps yet)`;
    const stepLines = plan.steps.map((s, i) => `  ${i + 1}. [${s.done ? "x" : " "}] (${s.id}) ${s.text}`);
    return [goalLine, ...stepLines].join("\n");
};

const profiles = {
    planning: {
        llm: { model: "anthropic/claude-opus-4-7" },
        instructions: [
            "You are in PLANNING MODE.",
            "Work with the user to break the goal down into concrete steps.",
            "Use the plan tools: set_goal, add_step, update_step, remove_step.",
            "Do not execute anything yet. Be concise. After tool calls, summarize the change in one line.",
        ].join("\n"),
    },
    executing: {
        llm: { model: "anthropic/claude-sonnet-4-6" },
        instructions: [
            "You are in EXECUTING MODE.",
            "Work through every pending step in the plan above, in order. For each one,",
            "call complete_step with a one-line note about how you handled it.",
            "Stop when every step is done.",
        ].join("\n"),
    },
} satisfies Record<Mode, { llm: Llm; instructions: string }>;

// ── Agent ───────────────────────────────────────────────────────────────────

const planner = agent<State>({
    name: "planner",
    decide: async (req) => {
        const state: State = {
            mode: req.state?.mode ?? "planning",
            plan: req.state?.plan ?? initialPlan(),
        };

        // A `set_mode` action switches modes; entering execution forks a fresh thread seeded with only the plan.
        let trigger: DecisionTrigger = req.trigger;
        let transcript = req.transcript;
        if (req.trigger.type === "client.action" && req.trigger.name === "set_mode") {
            const requested = (req.trigger.args as { mode?: Mode } | undefined)?.mode;
            if (requested !== "planning" && requested !== "executing") return { actions: [], state };
            const entering = requested !== state.mode;
            state.mode = requested;
            if (entering && requested === "executing") {
                transcript = [];
                trigger = { type: "user.message", message: { role: "user", content: renderPlan(state.plan) } };
            }
        }

        // Everything else is the SDK's default loop, parameterized by the current mode.
        const loop = toolLoop<State>({
            llm: profiles[state.mode].llm,
            instructions: profiles[state.mode].instructions,
            tools: planTools(state),
        });

        return loop({ ...req, trigger, transcript, state });
    },
});

// ── CLI driver ──────────────────────────────────────────────────────────────
// Usage:
//   pnpm tsx index.ts <session-id> "<message>"
//   pnpm tsx index.ts <session-id> /mode planning
//   pnpm tsx index.ts <session-id> /mode executing
// Reuse one session id across calls; sessions persist in agent.db.

const [, , sessionId, ...rest] = process.argv;
const input = rest.join(" ");
if (!sessionId || !input) {
    console.error('Usage: pnpm tsx index.ts <session-id> "<message>" | /mode planning | /mode executing');
    process.exit(1);
}

const payload = input.startsWith("/mode ")
    ? { type: "action" as const, name: "set_mode", args: { mode: input.slice(6).trim() as Mode } }
    : { type: "message" as const, message: { role: "user" as const, content: input } };

const embedded = await SubstructureEmbedded.create({
    agents: [planner],
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: "planner",
    payload,
    identity: { tenant_id: "default", id: "demo" },
    sessionId,
});

const label =
    payload.type === "action"
        ? `action:${payload.name}(${JSON.stringify(payload.args)})`
        : `message:"${payload.message.content}"`;
console.log(`── ${label}\nsession ${scope.sessionId}\n`);

for await (const event of embedded.stream(scope)) {
    const p = event.payload;
    switch (p.type) {
        case "llm.call.requested":
            console.log(`  → llm.call.requested (${p.request.model})`);
            break;
        case "llm.call.completed": {
            const text = p.response.content;
            if (text) console.log(`  ← llm: ${text.trim().replace(/\n+/g, " ⏎ ")}`);
            break;
        }
        case "tool.call.requested":
            console.log(`  → tool.call ${p.name}(${p.arguments})`);
            break;
        case "tool.call.completed":
            console.log(`  ← tool.result ${p.name}: ${p.result}`);
            break;
        case "tool.call.errored":
            console.log(`  ✗ tool.error ${p.name}: ${p.error}`);
            break;
        case "turn.completed":
            console.log(
                `✓ turn.completed${p.turn_cost ? `  cost=${p.turn_cost}` : ""}${
                    p.turn_token_usage ? `  tokens=${JSON.stringify(p.turn_token_usage)}` : ""
                }`,
            );
            break;
    }
}

await embedded.shutdown();
