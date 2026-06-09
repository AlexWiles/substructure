import { describe, expect, it } from "vitest";

import { llmToolLoop, messageHistory, serverGenerate, subAgents } from "../src/middleware";
import { HandlerBuilder } from "../src/worker";
import { actionsOfType, callLlm, llmResponse, runChain, toolCall, userMessage } from "./harness";

const researcher = new HandlerBuilder("researcher");
const llm = llmToolLoop({ generator: serverGenerate({ model: "test-model" }) });

const delegation = (toolCallId = "call_sub") =>
    llmResponse({
        role: "assistant",
        content: "",
        tool_calls: [toolCall("researcher", { message: "find X" }, toolCallId)],
    });

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
                trigger: delegation(),
            });
            expect(actionsOfType(result, "spawn.sub_agent")).toMatchObject([{ agent_id: "researcher" }]);
        });

        it("tags the spawn with the originating tool_call_id", async () => {
            const result = await runChain([messageHistory(), subAgents({ agents: [researcher] }), llm], {
                trigger: delegation("call_42"),
            });
            expect(actionsOfType(result, "spawn.sub_agent")[0]?.tool_call_id).toBe("call_42");
        });

        it("forwards the model's message to the spawned sub-agent", async () => {
            const result = await runChain([messageHistory(), subAgents({ agents: [researcher] }), llm], {
                trigger: delegation(),
            });
            const [spawn] = actionsOfType(result, "spawn.sub_agent");
            const send = actionsOfType(result, "send.message").find((m) => m.session_id === spawn.session_id);
            expect(send?.message).toEqual({ role: "user", content: "find X" });
        });

        it("does not emit a call.tool for a sub-agent", async () => {
            const result = await runChain([messageHistory(), subAgents({ agents: [researcher] }), llm], {
                trigger: delegation(),
            });
            expect(actionsOfType(result, "call.tool")).toHaveLength(0);
        });

        it("does not spawn for a tool call that is not a sub-agent", async () => {
            const result = await runChain([messageHistory(), subAgents({ agents: [researcher] }), llm], {
                trigger: llmResponse({ role: "assistant", content: "", tool_calls: [toolCall("getWeather", {})] }),
            });
            expect(actionsOfType(result, "spawn.sub_agent")).toHaveLength(0);
        });
    });
});
