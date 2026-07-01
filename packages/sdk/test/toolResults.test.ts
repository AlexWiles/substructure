import { describe, expect, it } from "vitest";

import { toolLoop } from "../src/agent";
import { serverGenerate } from "../src/core";
import type { Message } from "../src/types";
import { actionsOfType, appendedMessages, callLlm, linearTree, runAgent, toolResult } from "./harness";

const loop = toolLoop({ model: serverGenerate({ model: "test-model" }), instructions: "SYS" });

const assistant: Message = {
    role: "assistant",
    content: "",
    tool_calls: [
        { id: "call_a", type: "function", function: { name: "getWeather", arguments: "{}" } },
        { id: "call_b", type: "function", function: { name: "researcher", arguments: "{}" } },
    ],
};

describe("tool.result", () => {
    it("records the result and continues when nothing is left pending", async () => {
        // call_a already landed; call_b just completed and no effect is pending.
        const messageTree = linearTree({ role: "user", content: "go" }, assistant, {
            role: "tool",
            content: "RA",
            tool_call_id: "call_a",
            name: "getWeather",
        });
        const result = await runAgent(loop, {
            trigger: toolResult("call_b", "researcher", "RB"),
            messageTree,
            pending: { tool_calls: 0, sub_agents: 0, llm_calls: 0 },
        });

        expect(appendedMessages(result)).toMatchObject([
            { role: "tool", content: "RB", tool_call_id: "call_b", name: "researcher" },
        ]);
        expect(actionsOfType(result, "call.llm")).toHaveLength(1);

        const toolMsgs = (callLlm(result)?.request.messages ?? []).filter((m) => m.role === "tool");
        expect(toolMsgs).toMatchObject([
            { role: "tool", content: "RA", tool_call_id: "call_a", name: "getWeather" },
            { role: "tool", content: "RB", tool_call_id: "call_b", name: "researcher" },
        ]);
    });

    it("records but does not continue while another result is still pending", async () => {
        // call_a lands first; call_b (a sub-agent) is still pending, so the worker
        // records the result but does not prompt.
        const messageTree = linearTree({ role: "user", content: "go" }, assistant);
        const result = await runAgent(loop, {
            trigger: toolResult("call_a", "getWeather", "RA"),
            messageTree,
            pending: { tool_calls: 0, sub_agents: 1, llm_calls: 0 },
        });

        expect(appendedMessages(result)).toMatchObject([
            { role: "tool", content: "RA", tool_call_id: "call_a", name: "getWeather" },
        ]);
        expect(actionsOfType(result, "call.llm")).toHaveLength(0);
    });
});
