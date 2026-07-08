import { describe, expect, it } from "vitest";

import { toolLoop } from "../src/agent";
import { tool } from "../src/core";
import type { DecisionTrigger, Message } from "../src/types";
import { actionsOfType, appendedMessages, callLlm, linearTree, runAgent, subAgentResult, toolResult } from "./harness";

const loop = toolLoop({ llm: { model: "test-model" }, instructions: "SYS" });

const assistant: Message = {
    role: "assistant",
    content: "",
    tool_calls: [
        { id: "call_a", type: "function", function: { name: "getWeather", arguments: "{}" } },
        { id: "call_b", type: "function", function: { name: "researcher", arguments: "{}" } },
    ],
};

describe("finished calls", () => {
    it("records the result and continues when nothing is left pending", async () => {
        // call_a already landed; call_b (a sub-agent) just completed and no effect is pending.
        const messageTree = linearTree({ role: "user", content: "go" }, assistant, {
            role: "tool",
            content: "RA",
            tool_call_id: "call_a",
            name: "getWeather",
        });
        const result = await runAgent(loop, {
            trigger: subAgentResult("s-1", "call_b", "researcher", "RB"),
            messageTree,
        });

        expect(appendedMessages(result)).toMatchObject([
            { role: "tool", content: "RB", tool_call_id: "call_b", name: "researcher" },
        ]);
        expect(actionsOfType(result, "llm.call")).toHaveLength(1);

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
            calls: [
                {
                    id: "call_b",
                    kind: "sub_agent",
                    agent_id: "researcher",
                    session_id: "s-1",
                    status: "pending",
                    attempt: 0,
                },
            ],
        });

        expect(appendedMessages(result)).toMatchObject([
            { role: "tool", content: "RA", tool_call_id: "call_a", name: "getWeather" },
        ]);
        expect(actionsOfType(result, "llm.call")).toHaveLength(0);
    });
});

describe("deferred tools", () => {
    const executeTrigger: DecisionTrigger = {
        type: "tool.execute",
        id: "tc-1",
        name: "wait",
        arguments: "{}",
        input: { status: "valid", value: {} },
        attempt: 0,
    };

    it("runs the kick-off but emits no result action, leaving the call pending", async () => {
        let kickedOff = false;
        const wait = tool({
            name: "wait",
            description: "settled out-of-band",
            deferred: true,
            execute: () => {
                kickedOff = true;
            },
        });
        const result = await runAgent(toolLoop({ llm: { model: "test-model" }, tools: [wait] }), {
            trigger: executeTrigger,
        });

        expect(kickedOff).toBe(true);
        expect(result.actions).toHaveLength(0);
    });

    it("a throwing kick-off still reports tool.error", async () => {
        const wait = tool({
            name: "wait",
            description: "settled out-of-band",
            deferred: true,
            execute: () => {
                throw new Error("enqueue failed");
            },
        });
        const result = await runAgent(toolLoop({ llm: { model: "test-model" }, tools: [wait] }), {
            trigger: executeTrigger,
        });

        expect(actionsOfType(result, "tool.error")).toMatchObject([{ id: "tc-1", error: "enqueue failed" }]);
    });
});
