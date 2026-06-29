import { describe, expect, it } from "vitest";

import { llm, serverGenerate, toolResultNode } from "../src/middleware";
import type { Message } from "../src/types";
import { callLlm, linearTree, runChain, toolResults } from "./harness";

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
    it("continues once every tool call is answered", async () => {
        // Both results are in the tree → the loop continues.
        const messageTree = linearTree(
            { role: "user", content: "go" },
            assistant,
            { role: "tool", content: "RA", tool_call_id: "call_a", name: "getWeather" },
            { role: "tool", content: "RB", tool_call_id: "call_b", name: "researcher" },
        );
        const result = await runChain([loop], { trigger: toolResults(), messageTree });
        expect(result.actions.filter((a) => a.type === "call.llm")).toHaveLength(1);

        // The recorded results reach the model, read from the tree.
        const toolMsgs = (callLlm(result)?.request.messages ?? []).filter((m) => m.role === "tool");
        expect(toolMsgs).toMatchObject([
            { role: "tool", content: "RA", tool_call_id: "call_a", name: "getWeather" },
            { role: "tool", content: "RB", tool_call_id: "call_b", name: "researcher" },
        ]);
    });

    it("does not continue while a tool call is still unanswered", async () => {
        // Only call_a has landed; call_b is still pending → no continuation.
        const messageTree = linearTree({ role: "user", content: "go" }, assistant, {
            role: "tool",
            content: "RA",
            tool_call_id: "call_a",
            name: "getWeather",
        });
        const result = await runChain([loop], {
            trigger: toolResults([{ tool_call_id: "call_a", name: "getWeather" }]),
            messageTree,
        });
        expect(result.actions.filter((a) => a.type === "call.llm")).toHaveLength(0);
    });

    it("does not continue when the latest assistant turn issued no tool calls", async () => {
        const messageTree = linearTree({ role: "user", content: "go" }, { role: "assistant", content: "all done" });
        const result = await runChain([loop], { trigger: toolResults(), messageTree });
        expect(result.actions.filter((a) => a.type === "call.llm")).toHaveLength(0);
    });

    it("looks up a landed result node by tool_call_id", () => {
        const messageTree = linearTree({ role: "user", content: "go" }, assistant, {
            role: "tool",
            content: "RA",
            tool_call_id: "call_a",
            name: "getWeather",
        });
        expect(toolResultNode(messageTree, "call_a")?.content).toBe("RA");
        expect(toolResultNode(messageTree, "call_b")).toBeUndefined();
    });
});
