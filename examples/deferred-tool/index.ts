// Deferred (async) tool call: the tool returns `DEFERRED` to tell the
// worker not to emit any result action right now. A `setTimeout` later
// calls `embedded.submitToolCallResult(...)`, which the engine routes
// back into the session as a `tool.result` trigger — and the agent
// resumes.
//
// This is the pattern for tools that kick off real async work (webhooks,
// long jobs, human approvals) where the result arrives out-of-band.

import Substructure from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";

const sub = new Substructure();
const { agent } = sub;

let embedded: SubstructureEmbedded;

const wait = agent.tool({
    name: "wait",
    description: "Wait for the given number of seconds, then return.",
    parameters: {
        type: "object",
        properties: { seconds: { type: "number" } },
        required: ["seconds"],
    },
    execute: (args, ctx) => {
        const { seconds } = JSON.parse(args) as { seconds: number };
        const ms = Math.max(0, seconds) * 1000;

        setTimeout(() => {
            embedded
                .submitToolCallResult({
                    sessionId: ctx.sessionId,
                    toolCallId: ctx.toolCallId,
                    attempt: ctx.attempt,
                    result: JSON.stringify({ waited_seconds: seconds }),
                })
                .catch((err) => console.error("submitToolCallResult failed:", err));
        }, ms);

        return ctx.defer();
    },
});

const waitAgent = agent({ id: "waiter" })
    .use(agent.jsonState())
    .use(agent.systemMessage("You wait for the requested number of seconds, then tell the user you're done."))
    .use(agent.messageHistory())
    .use(agent.tools([wait]))
    .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));

embedded = await SubstructureEmbedded.create({
    agents: [waitAgent],
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: waitAgent.agentId,
    payload: { type: "message", message: { role: "user", content: "Wait 3 seconds." } },
    identity: { tenant_id: "default", id: "demo" },
});

console.log(`session ${scope.sessionId}`);

for await (const event of embedded.stream(scope)) {
    const p = event.payload;
    switch (p.type) {
        case "tool.call.requested":
            console.log(`  → tool.call ${p.name}(${p.arguments})  [deferred]`);
            break;
        case "tool.call.completed":
            console.log(`  ← tool.result ${p.name}: ${p.result}`);
            break;
        case "turn.completed":
            console.log(`✓ done: ${JSON.stringify(p.data)}`);
            break;
    }
}

await embedded.shutdown();
