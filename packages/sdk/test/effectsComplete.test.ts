import { describe, expect, it } from "vitest";

import { llmToolLoop, messageHistory } from "../src/middleware";
import { callLlm, effectsComplete, historyMessages, runChain, toolResult } from "./harness";

const llm = llmToolLoop({ model: "test-model" });

// The assistant turn that issued the calls is already in history.
const issued = {
    messages: [
        { role: "user", content: "go" },
        {
            role: "assistant",
            content: "",
            tool_calls: [
                { id: "call_a", type: "function", function: { name: "getWeather", arguments: "{}" } },
                { id: "call_b", type: "function", function: { name: "researcher", arguments: "{}" } },
            ],
        },
    ],
};

describe("effects.complete", () => {
    it("makes exactly one follow-up call.llm", async () => {
        const result = await runChain([messageHistory("SYS"), llm], {
            trigger: effectsComplete([toolResult("call_a", "RA")]),
            state: issued,
        });
        expect(result.actions.filter((a) => a.type === "call.llm")).toHaveLength(1);
    });

    it("folds every result into the call as a tool message, in order", async () => {
        const result = await runChain([messageHistory("SYS"), llm], {
            trigger: effectsComplete([
                toolResult("call_a", "RA", "getWeather"),
                toolResult("call_b", "RB", "researcher"),
            ]),
            state: issued,
        });
        const toolMsgs = (callLlm(result)?.request.messages ?? []).filter((m) => m.role === "tool");
        expect(toolMsgs).toEqual([
            { role: "tool", name: "getWeather", tool_call_id: "call_a", content: "RA" },
            { role: "tool", name: "researcher", tool_call_id: "call_b", content: "RB" },
        ]);
    });

    it("folds a mix of tool and sub-agent results uniformly", async () => {
        const result = await runChain([messageHistory("SYS"), llm], {
            trigger: effectsComplete([
                toolResult("call_a", "weather", "getWeather"),
                toolResult("call_b", "findings", "researcher"),
            ]),
            state: issued,
        });
        const contents = (callLlm(result)?.request.messages ?? [])
            .filter((m) => m.role === "tool")
            .map((m) => m.content);
        expect(contents).toEqual(["weather", "findings"]);
    });

    it("retains the folded results in history for later turns", async () => {
        const result = await runChain([messageHistory("SYS"), llm], {
            trigger: effectsComplete([toolResult("call_a", "RA"), toolResult("call_b", "RB")]),
            state: issued,
        });
        const toolMsgs = historyMessages(result).filter((m) => m.role === "tool");
        expect(toolMsgs.map((m) => m.tool_call_id)).toEqual(["call_a", "call_b"]);
    });

    it("carries an error result through unchanged", async () => {
        const result = await runChain([messageHistory("SYS"), llm], {
            trigger: effectsComplete([toolResult("call_a", "boom", "getWeather", true)]),
            state: issued,
        });
        const toolMsg = (callLlm(result)?.request.messages ?? []).find((m) => m.role === "tool");
        expect(toolMsg?.content).toBe("boom");
    });
});
