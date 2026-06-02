// Plan mode: build a plan iteratively over multiple turns, then flip a
// switch and execute it. Two middlewares split the work — `planMode`
// owns `{ mode, plan }` and records mode changes from `client.action`
// payloads; `modeAwareHistory` owns the conversation transcript and
// resets it whenever it observes the mode change. The same `set_mode`
// action that flips the state also kicks off execution: the chain
// proceeds normally, system prompt and tools swap to the executing
// variants, and the LLM call fires on the client.action trigger.
//
// What this shows:
//   - `client.action` as a non-message way for the client to drive
//     state changes that the chain reacts to.
//   - Selector-based middleware (`tools`, `llmLoop`) swapping output
//     based on a state slice. Mode-dependent tool gating means the
//     agent literally cannot call execution tools while planning, and
//     a smaller model handles planning while a bigger one handles
//     execution.
//   - A custom history middleware that reads a foreign slice (`mode`)
//     to know when to reset itself and to swap its own system prompt.
//
// Domain is intentionally generic: a TODO list. Planning mode edits
// the steps; executing mode walks them one by one with a single
// `complete_step` tool. Swap that tool for real work to adapt.

import Substructure, {
    middleware,
    prependHistoryToLlmCalls,
    triggerToMessage,
    type Message,
} from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";

const sub = new Substructure();
const { agent } = sub;

// ── Domain ──────────────────────────────────────────────────────────────────

type Mode = "planning" | "executing";
type Step = { id: string; text: string; done: boolean };
type Plan = { goal: string; steps: Step[]; nextId: number };

// ── Shared state ────────────────────────────────────────────────────────────
// Both middlewares (and downstream tools / selectors) operate on this shape.
// Multiple `middleware<PlanState>` calls below all declare the same type and
// the same initial values; `??=` semantics in the runtime mean redeclaring
// keys is harmless (whichever middleware initializes a key first wins).

type PlanState = {
    mode: Mode;
    plan: Plan;
    messages: Message[];
    lastMode?: Mode;
};

const initialState: PlanState = {
    mode: "planning",
    plan: { goal: "", steps: [], nextId: 1 },
    messages: [],
    lastMode: undefined,
};

// ── planMode: records `mode` changes from client.action payloads ────────────

const planMode = middleware<PlanState>({
    state: initialState,
    handler: async (ctx, next) => {
        const { trigger } = ctx.request;
        if (trigger.type === "client.action" && trigger.name === "set_mode") {
            const args = (trigger.args ?? {}) as { mode?: Mode };
            if (args.mode === "planning" || args.mode === "executing") {
                ctx.state.mode = args.mode;
            }
        }
        return next(ctx);
    },
});

// ── modeAwareHistory: transcript that resets on mode transition ─────────────

const modeAwareHistory = middleware<PlanState>({
    state: initialState,
    handler: async (ctx, next) => {
        if (ctx.state.mode !== ctx.state.lastMode) {
            ctx.state.messages = [];
            ctx.state.lastMode = ctx.state.mode;
        }

        const msg = triggerToMessage(ctx.request.trigger);
        if (msg) ctx.state.messages.push(msg);

        // System prompt swaps with the mode and rides ahead of the transcript.
        const sysMsg: Message = {
            role: "system",
            content: ctx.state.mode === "executing" ? executingPrompt(ctx.state.plan) : planningPrompt(ctx.state.plan),
        };

        const result = await next(ctx);
        return {
            ...result,
            actions: prependHistoryToLlmCalls([sysMsg, ...ctx.state.messages], result.actions),
        };
    },
});

// ── Plan-editing tools (planning mode only) ─────────────────────────────────

const setGoal = agent.tool({
    name: "set_goal",
    description: "Set or replace the overall goal the plan is working toward.",
    parameters: {
        type: "object",
        properties: { goal: { type: "string" } },
        required: ["goal"],
    },
    state: planMode,
    execute: (args, state) => {
        const { goal } = JSON.parse(args) as { goal: string };
        state.plan.goal = goal;
        return JSON.stringify(state.plan);
    },
});

const addStep = agent.tool({
    name: "add_step",
    description: "Append a new step to the plan. Returns the created step.",
    parameters: {
        type: "object",
        properties: { text: { type: "string" } },
        required: ["text"],
    },
    state: planMode,
    execute: (args, state) => {
        const { text } = JSON.parse(args) as { text: string };
        const step: Step = { id: `s${state.plan.nextId++}`, text, done: false };
        state.plan.steps.push(step);
        return JSON.stringify(step);
    },
});

const updateStep = agent.tool({
    name: "update_step",
    description: "Rewrite the text of an existing step by id.",
    parameters: {
        type: "object",
        properties: { id: { type: "string" }, text: { type: "string" } },
        required: ["id", "text"],
    },
    state: planMode,
    execute: (args, state) => {
        const { id, text } = JSON.parse(args) as { id: string; text: string };
        const step = state.plan.steps.find((s: Step) => s.id === id);
        if (!step) throw new Error(`unknown step: ${id}`);
        step.text = text;
        return JSON.stringify(step);
    },
});

const removeStep = agent.tool({
    name: "remove_step",
    description: "Remove a step from the plan by id.",
    parameters: {
        type: "object",
        properties: { id: { type: "string" } },
        required: ["id"],
    },
    state: planMode,
    execute: (args, state) => {
        const { id } = JSON.parse(args) as { id: string };
        const idx = state.plan.steps.findIndex((s: Step) => s.id === id);
        if (idx === -1) throw new Error(`unknown step: ${id}`);
        const [removed] = state.plan.steps.splice(idx, 1);
        return JSON.stringify(removed);
    },
});

// ── Execution tool (executing mode only) ────────────────────────────────────
// One tool. Mark a step done with a short note about how it was completed.
// Real agents would have richer execution surfaces (file ops, shell, network,
// sub-agents); the example only needs one to demonstrate the mode handoff.

const completeStep = agent.tool({
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
    state: planMode,
    execute: (args, state) => {
        const { id, note } = JSON.parse(args) as { id: string; note: string };
        const step = state.plan.steps.find((s: Step) => s.id === id);
        if (!step) throw new Error(`unknown step: ${id}`);
        step.done = true;
        return JSON.stringify({ id, text: step.text, note });
    },
});

const planningTools = [setGoal, addStep, updateStep, removeStep];
const executingTools = [completeStep];

// ── Prompt construction ─────────────────────────────────────────────────────

const renderPlan = (plan: Plan) => {
    const goalLine = `Goal: ${plan.goal || "(unset)"}`;
    if (plan.steps.length === 0) return `${goalLine}\n  (no steps yet)`;
    const stepLines = plan.steps.map((s, i) => `  ${i + 1}. [${s.done ? "x" : " "}] (${s.id}) ${s.text}`);
    return [goalLine, ...stepLines].join("\n");
};

const planningPrompt = (plan: Plan) =>
    [
        "You are in PLANNING MODE.",
        "Work with the user to break the goal down into concrete steps.",
        "Use the plan tools: set_goal, add_step, update_step, remove_step.",
        "Do not execute anything yet. Be concise. After tool calls, summarize the change in one line.",
        "",
        "Current plan:",
        renderPlan(plan),
    ].join("\n");

const executingPrompt = (plan: Plan) =>
    [
        "You are in EXECUTING MODE.",
        "Work through every pending step in order. For each one, call complete_step",
        "with a one-line note about how you handled it. Stop when every step is done.",
        "",
        "Plan:",
        renderPlan(plan),
    ].join("\n");

// ── Agent ───────────────────────────────────────────────────────────────────

const planner = agent({ id: "planner" })
    .use(planMode)
    .use(modeAwareHistory)
    .use(agent.tools<PlanState>((state) => (state.mode === "planning" ? planningTools : executingTools)))
    .use(
        agent.llmLoop<PlanState>((state) => ({
            request: {
                model: state.mode === "planning" ? "anthropic/claude-opus-4-7" : "anthropic/claude-sonnet-4-6",
            },
        })),
    );

// ── CLI driver ──────────────────────────────────────────────────────────────
// Usage:
//   pnpm tsx index.ts <session-id> "<message>"
//   pnpm tsx index.ts <session-id> /mode planning
//   pnpm tsx index.ts <session-id> /mode executing
//
// Generate a fresh id with `uuidgen` (macOS/linux) or any UUID generator
// and reuse it across calls. Sessions are persisted in agent.db.

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
    agentId: planner.agentId,
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
