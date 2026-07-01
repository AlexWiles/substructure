import { describe, expect, it } from "vitest";

import { toolLoop } from "../src/agent";
import type { LlmGenerator } from "../src/core";
import type { DecisionTrigger, LlmTokenDeltaInput } from "../src/types";
import { actionsOfType, runAgent } from "./harness";

// A worker-handled generator that streams two text deltas, then returns.
const streamingModel: LlmGenerator = {
    request: { model: "mock-model" },
    handler: "worker",
    stream: true,
    run: async (_request, ctx) => {
        await ctx.emitDelta?.({ type: "text-delta", delta: "Hel" });
        await ctx.emitDelta?.({ type: "text-delta", delta: "lo" });
        return { model: "mock-model", content: "Hello", tool_calls: [] };
    },
};

describe("worker LLM streaming", () => {
    it("runs the generator on llm.request, streams flattened deltas, and returns the result", async () => {
        const deltas: LlmTokenDeltaInput[] = [];
        const trigger: DecisionTrigger = {
            type: "llm.request",
            call_id: "c1",
            request: { model: "mock-model", messages: [] },
            stream: true,
            attempt: 0,
        };

        const result = await runAgent(toolLoop({ model: streamingModel }), {
            trigger,
            emitDelta: async (d) => {
                deltas.push(d);
            },
        });

        // The generator's StreamParts reach the worker's emitDelta as token deltas.
        expect(deltas).toEqual([{ text: "Hel" }, { text: "lo" }]);

        // The call's result is returned to the engine.
        const [ret] = actionsOfType(result, "return.llm.result");
        expect(ret?.call_id).toBe("c1");
        expect(ret?.response.content).toBe("Hello");
    });

    it("returns a retryable-free error when a worker model has no run", async () => {
        const noRun: LlmGenerator = { request: { model: "x" }, handler: "worker" };
        const trigger: DecisionTrigger = {
            type: "llm.request",
            call_id: "c2",
            request: { model: "x", messages: [] },
            stream: false,
            attempt: 0,
        };
        const result = await runAgent(toolLoop({ model: noRun }), { trigger });
        const [err] = actionsOfType(result, "return.llm.error");
        expect(err?.call_id).toBe("c2");
        expect(err?.retryable).toBe(false);
    });
});
