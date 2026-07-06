// Deferred tool: `deferred: true` makes `execute` a kick-off with no result action, so the
// engine leaves the call pending until `settleEffect` routes the out-of-band result back as
// a `tool.finished` trigger, the pattern for webhooks, long jobs, or human approvals.

import { agent, tool, toolLoop } from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";

let embedded: SubstructureEmbedded;

const wait = tool({
    name: "wait",
    description: "Wait for the given number of seconds, then return.",
    parameters: {
        type: "object",
        properties: { seconds: { type: "number" } },
        required: ["seconds"],
    },
    deferred: true,
    execute: (args, request) => {
        const { seconds } = JSON.parse(args) as { seconds: number };
        if (request.trigger.type !== "tool.execute") throw new Error("wait ran outside a tool call");
        const { id, attempt } = request.trigger;
        const sessionId = request.session_id;

        setTimeout(() => {
            embedded
                .settleEffect({
                    sessionId,
                    id,
                    attempt,
                    result: JSON.stringify({ waited_seconds: seconds }),
                })
                .catch((err) => console.error("settleEffect failed:", err));
        }, Math.max(0, seconds) * 1000);
    },
});

const waitAgent = agent({
    name: "waiter",
    decide: toolLoop({
        llm: { model: "anthropic/claude-sonnet-4-6" },
        instructions: "You wait for the requested number of seconds, then tell the user you're done.",
        tools: [wait],
    }),
});

embedded = await SubstructureEmbedded.create({
    agents: [waitAgent],
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: "waiter",
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
