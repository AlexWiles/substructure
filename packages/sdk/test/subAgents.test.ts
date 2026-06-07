import { describe, expect, it } from "vitest";

import { llmLoop, messageHistory, subAgents } from "../src/middleware";
import { HandlerBuilder } from "../src/worker";
import {
    actionsOfType,
    callLlm,
    historyMessages,
    llmResponse,
    runChain,
    subAgentComplete,
    subAgentError,
    toolCall,
    toolMessages,
    userMessage,
} from "./harness";

const researcher = new HandlerBuilder("researcher");
const llm = llmLoop({ request: { model: "test-model" } });

// State as it stands after the model has delegated to a sub-agent: the
// delegating turn is in history and the spawned session is tracked.
const delegated = {
    subAgentTracker: { "child-1": { toolCallId: "call_sub", name: "researcher" } },
    messages: [
        { role: "user", content: "do research" },
        { role: "assistant", content: "", tool_calls: [toolCall("researcher", {}, "call_sub")] },
    ],
};

describe("subAgents", () => {
    describe("tool exposure", () => {
        it("presents each sub-agent to the model as a tool", async () => {
            const result = await runChain([messageHistory(), subAgents({ agents: [researcher] }), llm], {
                trigger: userMessage("hi"),
            });
            expect(callLlm(result)?.request.tools?.map((t) => t.function.name)).toEqual(["researcher"]);
        });
    });

    describe("delegation", () => {
        it("spawns the sub-agent when the model calls it", async () => {
            const result = await runChain([messageHistory(), subAgents({ agents: [researcher] }), llm], {
                trigger: llmResponse({
                    role: "assistant",
                    content: "",
                    tool_calls: [toolCall("researcher", { message: "find X" })],
                }),
            });
            expect(actionsOfType(result, "spawn.sub_agent")).toMatchObject([{ agent_id: "researcher" }]);
        });

        it("forwards the model's message to the spawned sub-agent", async () => {
            const result = await runChain([messageHistory(), subAgents({ agents: [researcher] }), llm], {
                trigger: llmResponse({
                    role: "assistant",
                    content: "",
                    tool_calls: [toolCall("researcher", { message: "find X" })],
                }),
            });
            const [spawn] = actionsOfType(result, "spawn.sub_agent");
            const send = actionsOfType(result, "send.message").find((m) => m.session_id === spawn.session_id);
            expect(send?.message).toEqual({ role: "user", content: "find X" });
        });
    });

    describe("completion", () => {
        it("folds the sub-agent result back as a tool result", async () => {
            const result = await runChain([messageHistory("SYS"), subAgents({ agents: [researcher] }), llm], {
                trigger: subAgentComplete("child-1", "researcher", "RESULT"),
                state: delegated,
            });
            expect(toolMessages(result, "call_sub")).toEqual([
                { role: "tool", name: "researcher", tool_call_id: "call_sub", content: "RESULT" },
            ]);
        });

        it("includes the result once when messageHistory wraps subAgents", async () => {
            const result = await runChain([messageHistory("SYS"), subAgents({ agents: [researcher] }), llm], {
                trigger: subAgentComplete("child-1", "researcher", "RESULT"),
                state: delegated,
            });
            expect(toolMessages(result, "call_sub")).toHaveLength(1);
        });

        it("includes the result once when subAgents wraps messageHistory", async () => {
            const result = await runChain([subAgents({ agents: [researcher] }), messageHistory("SYS"), llm], {
                trigger: subAgentComplete("child-1", "researcher", "RESULT"),
                state: delegated,
            });
            expect(toolMessages(result, "call_sub")).toHaveLength(1);
        });

        it("serializes non-string sub-agent results to JSON", async () => {
            const result = await runChain([messageHistory("SYS"), subAgents({ agents: [researcher] }), llm], {
                trigger: subAgentComplete("child-1", "researcher", { score: 42 }),
                state: delegated,
            });
            expect(toolMessages(result, "call_sub")[0]?.content).toBe('{"score":42}');
        });

        it("reports a failed sub-agent as an error tool result", async () => {
            const result = await runChain([messageHistory("SYS"), subAgents({ agents: [researcher] }), llm], {
                trigger: subAgentError("child-1", "researcher", "boom"),
                state: delegated,
            });
            const [msg] = toolMessages(result, "call_sub");
            expect(msg?.content).toContain("boom");
        });

        it("ignores completions for sessions it is not tracking", async () => {
            const result = await runChain([messageHistory("SYS"), subAgents({ agents: [researcher] }), llm], {
                trigger: subAgentComplete("unknown-session", "researcher", "RESULT"),
                state: delegated,
            });
            expect(toolMessages(result, "call_sub")).toHaveLength(0);
        });

        it("folds the result back without messageHistory in the chain", async () => {
            const result = await runChain([subAgents({ agents: [researcher] }), llm], {
                trigger: subAgentComplete("child-1", "researcher", "RESULT"),
                state: delegated,
            });
            expect(toolMessages(result, "call_sub")).toEqual([
                { role: "tool", name: "researcher", tool_call_id: "call_sub", content: "RESULT" },
            ]);
        });
    });

    describe("history retention", () => {
        it("records the result in history when subAgents wraps messageHistory", async () => {
            const result = await runChain([subAgents({ agents: [researcher] }), messageHistory("SYS"), llm], {
                trigger: subAgentComplete("child-1", "researcher", "RESULT"),
                state: delegated,
            });
            expect(
                historyMessages(result).filter((m) => m.role === "tool" && m.tool_call_id === "call_sub"),
            ).toHaveLength(1);
        });

        it("leaves the result out of history when messageHistory wraps subAgents", async () => {
            const result = await runChain([messageHistory("SYS"), subAgents({ agents: [researcher] }), llm], {
                trigger: subAgentComplete("child-1", "researcher", "RESULT"),
                state: delegated,
            });
            expect(historyMessages(result).some((m) => m.role === "tool" && m.tool_call_id === "call_sub")).toBe(false);
        });
    });
});
