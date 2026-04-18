// Substructure SDK – Custom Middleware Example
// This example shows how to write your own middleware. Middleware is a function
// that wraps the agent's request/response pipeline, letting you inspect or
// modify behavior at every decision point.
//
// We build two custom middleware:
//   1. contextInjector – appends the current time to the user's message in LLM calls
//   2. turnCounter    – tracks how many turns have occurred (demonstrates stateful middleware)

import Substructure, { middleware } from "@substructure.ai/sdk";
import { randomUUID } from "crypto";

const sub = new Substructure();
const { agent } = sub;

// ── Middleware 1: Context Injector (stateless) ──────────────────────────────
// A middleware that appends the current time to the user's message in outgoing
// LLM calls. It intercepts `call.llm` actions on the way back up and modifies
// the last user message. No `state` key — this middleware is purely a wrapper.

const contextInjector = middleware({
    handler: async (req, next) => {
        const result = await next(req);
        const now = new Date().toLocaleString();

        const actions = result.actions.map((action) => {
            if (action.type !== "call.llm") return action;

            // Find the last user message and append the timestamp.
            const messages = action.request.messages.map((msg) => msg);
            for (let i = messages.length - 1; i >= 0; i--) {
                if (messages[i].role === "user") {
                    messages[i] = { ...messages[i], content: `${messages[i].content}\n\n[Current time: ${now}]` };
                    break;
                }
            }

            return { ...action, request: { ...action.request, messages } };
        });

        return { ...result, actions };
    },
});

// ── Middleware 2: Turn Counter (stateful) ───────────────────────────────────
// When middleware needs its own persistent state, use the `middleware()` helper
// with a `state` object. This registers a "state slice" — the SDK merges it
// into the agent's overall state and persists it between turns automatically.

const turnCounter = middleware({
    state: { turnCount: 0 },

    handler: async (req, next) => {
        // Only count user-initiated turns (not tool results, LLM responses, etc.)
        if (req.trigger.type === "user.message") {
            req.state.turnCount += 1;
            console.log(`  [turnCounter] turn #${req.state.turnCount}`);
        }

        const res = await next(req);
        return res;
    },
});

// ── Agent Definition ────────────────────────────────────────────────────────
// Custom middleware slots into the `.use()` chain just like built-in middleware.
// Order matters: contextInjector wraps the response on the way back up,
// modifying `call.llm` actions produced by downstream middleware.

const assistant = agent({ id: "assistant" })
    .use(agent.jsonState())
    .use(agent.systemMessage("You are a helpful assistant named Alex. Be concise."))
    .use(agent.messageHistory())
    .use(turnCounter)
    .use(contextInjector)
    .use(
        agent.llmLoop({
            request: { model: "anthropic/claude-sonnet-4" },
            llm_client: "openrouter",
        }),
    );

// ── Run it ──────────────────────────────────────────────────────────────────

const embedded = await sub.embedded({
    agents: [assistant],
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const sessionId = randomUUID();
const identity = { tenant_id: "default", id: "alice" };

async function turn(message: string) {
    console.log(`\n> ${message}`);
    const stream = embedded.submit({
        agentId: assistant.agentId,
        payload: { type: "message", message: { role: "user", content: message } },
        sessionId,
        identity,
        turnId: randomUUID(),
    });
    for await (const event of stream) {
        if (
            event.payload.type === "message.new" &&
            event.payload.message.role === "assistant" &&
            event.payload.message.content
        ) {
            console.log(event.payload.message.content);
        }
    }
}

await turn("What is your name?");
await turn("What time is it?");
await turn("What's 2 + 2?");

await embedded.shutdown();
