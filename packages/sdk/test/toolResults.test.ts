import { describe, expect, it } from "vitest";

import { llm, serverGenerate } from "../src/middleware";
import type { Message } from "../src/types";
import { callLlm, linearTree, runChain, toolResult, toolResults } from "./harness";

const loop = llm({ generator: serverGenerate({ model: "test-model" }), instructions: "SYS" });

// The assistant turn that issued the calls, as the engine recorded it.
const assistant: Message = {
    role: "assistant",
    content: "",
    tool_calls: [
        { id: "call_a", type: "function", function: { name: "getWeather", arguments: "{}" } },
        { id: "call_b", type: "function", function: { name: "researcher", arguments: "{}" } },
    ],
};

describe("tool.results", () => {
    it("makes exactly one follow-up call.llm", async () => {
        const result = await runChain([loop], {
            trigger: toolResults([toolResult("call_a", "RA")]),
        });
        expect(result.actions.filter((a) => a.type === "call.llm")).toHaveLength(1);
    });

    it("sends the recorded tool results to the model, in order", async () => {
        // By the time the batch trigger fires, the results are nodes in the tree.
        const messageTree = linearTree(
            { role: "user", content: "go" },
            assistant,
            { role: "tool", content: "RA", tool_call_id: "call_a", name: "getWeather" },
            { role: "tool", content: "RB", tool_call_id: "call_b", name: "researcher" },
        );
        const result = await runChain([loop], {
            trigger: toolResults([toolResult("call_a", "RA"), toolResult("call_b", "RB")]),
            messageTree,
        });
        const toolMsgs = (callLlm(result)?.request.messages ?? []).filter((m) => m.role === "tool");
        expect(toolMsgs).toEqual([
            { role: "tool", content: "RA", tool_call_id: "call_a", name: "getWeather" },
            { role: "tool", content: "RB", tool_call_id: "call_b", name: "researcher" },
        ]);
    });
});
