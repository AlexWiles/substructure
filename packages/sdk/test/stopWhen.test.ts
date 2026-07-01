import { describe, expect, it } from "vitest";

import { toolLoop } from "../src/agent";
import { serverGenerate, stepCountIs } from "../src/core";
import type { Message } from "../src/types";
import { actionsOfType, appendedMessages, linearTree, runAgent, toolResult, userMessage } from "./harness";

const model = serverGenerate({ model: "test-model" });

const assistant: Message = {
    role: "assistant",
    content: "round 1",
    tool_calls: [{ id: "call_a", type: "function", function: { name: "getWeather", arguments: "{}" } }],
};

// The tool result is not in the tree yet — firing the back edge appends it and completes the round.
const pending = linearTree({ role: "user", content: "go" }, assistant);

const backEdge = () => toolResult("call_a", "getWeather", "RA");

describe("stopWhen", () => {
    it("lets the loop continue when the condition is false", async () => {
        const loop = toolLoop({ model, stopWhen: stepCountIs(5) });
        const result = await runAgent(loop, { trigger: backEdge(), messageTree: pending });
        expect(actionsOfType(result, "call.llm")).toHaveLength(1);
        expect(actionsOfType(result, "done")).toHaveLength(0);
    });

    it("replaces the back-edge call.llm with done while keeping the result in the transcript when the condition holds", async () => {
        const loop = toolLoop({ model, stopWhen: stepCountIs(1) });
        const result = await runAgent(loop, { trigger: backEdge(), messageTree: pending });
        expect(appendedMessages(result)).toMatchObject([{ role: "tool", tool_call_id: "call_a" }]);
        expect(actionsOfType(result, "call.llm")).toHaveLength(0);
        expect(actionsOfType(result, "done")[0]?.data).toBe("round 1");
    });

    it("ignores triggers other than tool.result", async () => {
        const loop = toolLoop({ model, stopWhen: stepCountIs(1) });
        const result = await runAgent(loop, { trigger: userMessage("hi"), messageTree: pending });
        expect(actionsOfType(result, "call.llm")).toHaveLength(1);
        expect(actionsOfType(result, "done")).toHaveLength(0);
    });
});
