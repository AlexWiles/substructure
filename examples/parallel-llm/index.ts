// Parallel LLM calls layered over the stock `toolLoop`: a thin wrapper owns the
// fan-out, and every trigger it doesn't recognize delegates straight to the loop.
//
// On the user's message the wrapper fans out one `call.llm` per lens — each named
// by its own `id`, all in flight at once, each settling independently as its own
// `effect.settled { kind: "llm_call" }` (serialized on the decision stream, in
// completion order). Once every lens has landed, the wrapper hands the enriched
// question to the wrapped `toolLoop`, which runs the normal loop to synthesize
// the final answer.
//
// Two flavors:
//   - `fanout`          — engine-handled lenses (handler: "server"). The engine
//                         calls its provider for each; no async machinery in the
//                         worker. Note the engine executes a session's calls one
//                         at a time, so this decouples your loop rather than
//                         shrinking wall-clock time.
//   - `deferred-fanout` — worker-handled lenses (handler: "worker"). Each comes
//                         back as an `effect.execute`; the worker starts it in
//                         the background, returns the decision immediately, and
//                         settles each out-of-band with `settleEffect` as it
//                         finishes — true wall-clock overlap. Fully stubbed, so
//                         it runs without an API key.

import { agent, contentText, toolLoop } from "@substructure.ai/sdk";
import type { Agent, LlmRequest, LlmResponse, WorkerAction } from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";

const MODEL = "anthropic/claude-sonnet-4-6";

// Two independent questions about the same input — a natural fan-out.
const LENSES = [
    { id: "lens-summary", prompt: (t: string) => `Summarize in one sentence: ${t}` },
    {
        id: "lens-sentiment",
        prompt: (t: string) => `One word — positive, neutral, or negative — for the tone of: ${t}`,
    },
];
const isLens = (id: string) => LENSES.some((l) => l.id === id);

// `pending` is the set of lens ids still in flight. `d.effects` (frozen on each
// decision at commit time, delivered in completion order) empties exactly on the
// last settle, so branching on it is equally correct for a single fan-out;
// tracking issued ids also stays correct if you spread a fan-out across turns.
interface FanoutState {
    question: string;
    pending: string[];
    results: Record<string, string>;
}

let embedded: SubstructureEmbedded;

// Stands in for your real provider call in the deferred flavor. In production
// this is where you'd hit the model (and stream via `d.emitDelta`).
async function runModel(request: LlmRequest): Promise<LlmResponse> {
    const last = request.messages.at(-1);
    await new Promise((r) => setTimeout(r, 150));
    return {
        model: request.model,
        content: `stub answer for: ${contentText(last?.content)}`,
        tool_calls: [],
        finish_reason: "stop",
    };
}

const INSTRUCTIONS = "Combine the lens analyses into one short answer for the user.";

/** Wrap a `toolLoop` with a lens fan-out pre-pass. The wrapper owns the `lens-*`
 *  effect ids; everything else falls through to the loop untouched. */
function withLenses(loop: Agent<FanoutState>, handler: "server" | "worker"): Agent<FanoutState> {
    return async (d) => {
        // The user's message: fan out the lenses instead of starting the loop.
        if (d.trigger.type === "user.message") {
            const question = contentText(d.trigger.message.content);
            return {
                actions: LENSES.map(
                    (l): WorkerAction => ({
                        type: "call.llm",
                        id: l.id,
                        handler,
                        request: { model: MODEL, messages: [{ role: "user", content: l.prompt(question) }] },
                    }),
                ),
                state: { question, pending: LENSES.map((l) => l.id), results: {} },
            };
        }

        // (deferred flavor) a lens `effect.execute`: start the call in the
        // background and return the decision *now* with no actions — the next
        // execute promotes immediately, so the calls overlap. Settle each
        // out-of-band whenever it finishes.
        if (d.trigger.type === "effect.execute" && d.trigger.kind === "llm_call" && isLens(d.trigger.id)) {
            const { id, attempt, request } = d.trigger;
            const sessionId = d.session_id;
            void runModel(request)
                .then((response) => embedded.settleEffect({ sessionId, id, attempt, kind: "llm_call", response }))
                .catch((err) =>
                    embedded.settleEffect({
                        sessionId,
                        id,
                        attempt,
                        kind: "llm_call",
                        error: err instanceof Error ? err.message : String(err),
                        retryable: false,
                    }),
                );
            return { state: d.state };
        }

        // A lens settling: fold it in; once the last one lands, hand the enriched
        // question to the wrapped loop as a fresh user message.
        if (d.trigger.type === "effect.settled" && d.trigger.kind === "llm_call" && isLens(d.trigger.id)) {
            const settledId = d.trigger.id;
            const prev = d.state ?? { question: "", pending: [], results: {} };
            const answer =
                d.trigger.ok && d.trigger.message
                    ? contentText(d.trigger.message.content)
                    : `(failed: ${d.trigger.error})`;
            const state = {
                ...prev,
                pending: prev.pending.filter((p) => p !== settledId),
                results: { ...prev.results, [settledId]: answer },
            };
            if (state.pending.length > 0) return { state };
            const briefing = [
                state.question,
                "",
                "Lens analyses (from parallel calls):",
                ...LENSES.map((l) => `- ${l.id}: ${state.results[l.id]}`),
            ].join("\n");
            return loop({
                ...d,
                state,
                trigger: { type: "user.message", message: { role: "user", content: briefing } },
            });
        }

        // Everything else — the loop's own llm settles, tools, done — is the loop's.
        return loop(d);
    };
}

const fanoutAgent = agent({
    name: "fanout",
    decide: withLenses(toolLoop({ llm: { model: MODEL }, instructions: INSTRUCTIONS }), "server"),
});

// The deferred flavor keeps the loop's own synthesis call on the worker too
// (`handler: "worker"` + `run`), so the whole demo runs offline.
const deferredAgent = agent({
    name: "deferred-fanout",
    decide: withLenses(
        toolLoop({ llm: { model: MODEL, handler: "worker", run: runModel }, instructions: INSTRUCTIONS }),
        "worker",
    ),
});

embedded = await SubstructureEmbedded.create({
    agents: [fanoutAgent, deferredAgent],
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const agentId = process.argv[2] === "deferred" ? "deferred-fanout" : "fanout";
console.log(`agent: ${agentId}`);

const scope = await embedded.startTurn({
    agentId,
    payload: {
        type: "message",
        message: { role: "user", content: "Substructure lets one agent fan out several model calls at once." },
    },
    identity: { tenant_id: "default", id: "demo" },
});

console.log(`session ${scope.sessionId}`);

for await (const event of embedded.stream(scope)) {
    const p = event.payload;
    switch (p.type) {
        case "llm.call.requested":
            console.log(`  → call.llm ${p.call_id}`);
            break;
        case "llm.call.completed":
            console.log(`  ← settled ${p.call_id}`);
            break;
        case "turn.completed":
            console.log(`✓ done: ${JSON.stringify(p.data, null, 2)}`);
            break;
    }
}

await embedded.shutdown();
