import { describe, expect, it } from "vitest";

import { llm, serverGenerate, stepCountIs, stopWhen } from "../src/middleware";
import type { Message } from "../src/types";
import { actionsOfType, linearTree, runChain, toolResult, toolResults, userMessage } from "./harness";

const loop = llm({ generator: serverGenerate({ model: "test-model" }) });

const assistant: Message = {
    role: "assistant",
    content: "round 1",
    tool_calls: [{ id: "call_a", type: "function", function: { name: "getWeather", arguments: "{}" } }],
};

// One completed round: user -> assistant(tool call) -> tool result.
const oneRound = linearTree({ role: "user", content: "go" }, assistant, {
    role: "tool",
    content: "RA",
    tool_call_id: "call_a",
    name: "getWeather",
});

const backEdge = () => toolResults([toolResult("call_a", "RA", "getWeather")]);

describe("stopWhen", () => {
    it("lets the loop continue when the condition is false", async () => {
        const result = await runChain([stopWhen(stepCountIs(5)), loop], {
            trigger: backEdge(),
            messageTree: oneRound,
        });
        expect(actionsOfType(result, "call.llm")).toHaveLength(1);
        expect(actionsOfType(result, "done")).toHaveLength(0);
    });

    it("replaces the back-edge call.llm with done when the condition holds", async () => {
        const result = await runChain([stopWhen(stepCountIs(1)), loop], {
            trigger: backEdge(),
            messageTree: oneRound,
        });
        expect(actionsOfType(result, "call.llm")).toHaveLength(0);
        expect(actionsOfType(result, "done")[0]?.data).toBe("round 1");
    });

    it("chains as OR with a single done when multiple conditions hold", async () => {
        const result = await runChain([stopWhen(stepCountIs(1)), stopWhen(stepCountIs(1)), loop], {
            trigger: backEdge(),
            messageTree: oneRound,
        });
        expect(actionsOfType(result, "done")).toHaveLength(1);
        expect(actionsOfType(result, "call.llm")).toHaveLength(0);
    });

    it("ignores triggers other than tool.results", async () => {
        // A user message enters the loop; it is not a back-edge to guard.
        const result = await runChain([stopWhen(stepCountIs(1)), loop], {
            trigger: userMessage("hi"),
            messageTree: oneRound,
        });
        expect(actionsOfType(result, "call.llm")).toHaveLength(1);
        expect(actionsOfType(result, "done")).toHaveLength(0);
    });
});
