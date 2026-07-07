import { describe, expect, it } from "vitest";

import { toolLoop } from "../src/agent";
import type { Agent } from "../src/core";
import type { SettleEffectArgs, WorkerAction } from "../src/types";
import { contentText, toSettleEffectRequest } from "../src/types";
import { actionsOfType, callLlm, clientMessage, llmResponse, runAgent } from "./harness";

describe("llm.call id", () => {
    it("ask() names each llm.call with a fresh id", async () => {
        const loop = toolLoop({ llm: { model: "test-model" }, instructions: "SYS" });
        const result = await runAgent(loop, { trigger: clientMessage("hi") });
        const call = callLlm(result);
        expect(typeof call?.id).toBe("string");
        expect(call?.id).toBeTruthy();
    });
});

describe("concurrent llm fan-out", () => {
    // A minimal fan-out agent: two engine-handled calls, folded by tracking the
    // ids it issued (race-free vs. the effects snapshot), `done` once both land.
    interface St {
        pending: string[];
        results: Record<string, string>;
    }
    const call = (id: string): WorkerAction => ({
        type: "llm.call",
        id,
        handler: "server",
        request: { model: "m", messages: [] },
    });
    const fanout: Agent = (d) => {
        if (d.trigger.type === "client.messages") {
            return { actions: [call("a"), call("b")], state: { pending: ["a", "b"], results: {} } };
        }
        if (d.trigger.type === "llm.finished") {
            const prev = (d.state as St) ?? { pending: [], results: {} };
            const answer = d.trigger.ok && d.trigger.message ? contentText(d.trigger.message.content) : "(failed)";
            const results = { ...prev.results, [d.trigger.id]: answer };
            const pending = prev.pending.filter((p) => p !== d.trigger.id);
            if (pending.length > 0) return { state: { pending, results } };
            return { actions: [{ type: "done", data: results }], state: { pending, results } };
        }
        return { state: d.state };
    };

    for (const order of [
        ["a", "b"],
        ["b", "a"],
    ] as const) {
        it(`folds settles arriving in ${order.join(", ")} order and finishes once`, async () => {
            const start = await runAgent(fanout, { trigger: clientMessage("go") });
            expect(
                actionsOfType(start, "llm.call")
                    .map((c) => c.id)
                    .sort(),
            ).toEqual(["a", "b"]);

            const first = await runAgent(fanout, {
                trigger: llmResponse({ role: "assistant", content: `R-${order[0]}` }, order[0]),
                state: start.state as Record<string, unknown>,
            });
            expect(actionsOfType(first, "done")).toHaveLength(0);

            const second = await runAgent(fanout, {
                trigger: llmResponse({ role: "assistant", content: `R-${order[1]}` }, order[1]),
                state: first.state as Record<string, unknown>,
            });
            const done = actionsOfType(second, "done");
            expect(done).toHaveLength(1);
            expect(done[0].data).toEqual({ a: "R-a", b: "R-b" });
        });
    }
});

describe("toSettleEffectRequest", () => {
    it("maps a tool result to a tool.result", () => {
        expect(toSettleEffectRequest({ sessionId: "s", id: "tc-1", attempt: 0, result: "ok" })).toEqual({
            type: "tool.result",
            id: "tc-1",
            result: "ok",
            attempt: 0,
        });
    });

    it("maps an llm response to an llm.result", () => {
        const response = { model: "m", tool_calls: [] };
        const args: SettleEffectArgs = { sessionId: "s", id: "llm-1", attempt: 2, kind: "llm_call", response };
        expect(toSettleEffectRequest(args)).toEqual({
            type: "llm.result",
            id: "llm-1",
            response,
            attempt: 2,
        });
    });

    it("maps a failure to the error type its kind selects", () => {
        expect(
            toSettleEffectRequest({
                sessionId: "s",
                id: "llm-1",
                attempt: 0,
                kind: "llm_call",
                error: "boom",
                retryable: true,
            }),
        ).toEqual({
            type: "llm.error",
            id: "llm-1",
            error: "boom",
            retryable: true,
            attempt: 0,
        });
    });
});
