import { describe, expect, it } from "vitest";

import { toolLoop } from "../src/agent";
import type { Llm } from "../src/core";
import type { DecisionTrigger, LlmTokenDeltaInput } from "../src/types";
import { actionsOfType, runAgent } from "./harness";

// A worker-handled LLM that streams two text deltas, then returns.
const streamingModel: Llm = {
    model: "mock-model",
    handler: "worker",
    stream: true,
    run: async (_request, ctx) => {
        await ctx.emitDelta?.({ text: "Hel" });
        await ctx.emitDelta?.({ text: "lo" });
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

        const result = await runAgent(toolLoop({ llm: streamingModel }), {
            trigger,
            emitDelta: async (d) => {
                deltas.push(d);
            },
        });

        // The token deltas the generator emits reach the worker's emitDelta unchanged.
        expect(deltas).toEqual([{ text: "Hel" }, { text: "lo" }]);

        // The call's result is returned to the engine.
        const [ret] = actionsOfType(result, "return.llm.result");
        expect(ret?.call_id).toBe("c1");
        expect(ret?.response.content).toBe("Hello");
    });
});
