import { describe, expect, it } from "vitest";

import { toolLoop } from "../src/agent";
import { activePath } from "../src/core";
import type { Message, MessageTree, Node } from "../src/types";
import { appendedMessages, callLlm, runAgent, clientMessage } from "./harness";

function node(id: string, parentId: string | undefined, content: string, role: Message["role"] = "user"): Node {
    return { kind: "message", parent_id: parentId, message: { id, role, content } };
}

describe("activePath", () => {
    it("walks the head-to-root path in order", () => {
        const tree: MessageTree = {
            nodes: [node("u1", undefined, "U1"), node("a1", "u1", "A1", "assistant"), node("u2", "a1", "U2")],
            head_id: "u2",
        };
        expect(activePath(tree).map((m) => m.content)).toEqual(["U1", "A1", "U2"]);
    });

    it("follows the active leaf past an abandoned branch", () => {
        const tree: MessageTree = {
            nodes: [
                node("u1", undefined, "U1"),
                node("a1", "u1", "A1", "assistant"),
                node("u2", "a1", "U2"),
                node("a2", "u2", "A2", "assistant"),
                node("u2b", "a1", "U2 edited"),
            ],
            head_id: "u2b",
        };
        expect(activePath(tree).map((m) => m.content)).toEqual(["U1", "A1", "U2 edited"]);
    });

    it("stops at a missing parent instead of looping or throwing", () => {
        const tree: MessageTree = { nodes: [node("x", "gone", "X")], head_id: "x" };
        expect(activePath(tree).map((m) => m.content)).toEqual(["X"]);
    });

    it("is empty for an absent tree", () => {
        expect(activePath(undefined)).toEqual([]);
    });
});

describe("agent prompt", () => {
    const loop = toolLoop({ llm: { model: "test-model" }, instructions: "SYS" });
    const promptOf = (result: Awaited<ReturnType<typeof runAgent>>) =>
        (callLlm(result)?.request.messages ?? []).map((m) => m.content);

    it("seeds [system, user] on a cold start (empty tree)", async () => {
        const result = await runAgent(loop, { trigger: clientMessage("hi") });
        expect(appendedMessages(result).map((m) => m.role)).toEqual(["system", "user"]);
        expect(appendedMessages(result).map((m) => m.content)).toEqual(["SYS", "hi"]);
        expect(promptOf(result)).toEqual(["SYS", "hi"]);
    });

    it("appends the new user turn under the head, after the active path", async () => {
        const tree: MessageTree = {
            nodes: [
                node("s1", undefined, "SYS", "system"),
                node("u1", "s1", "U1"),
                node("a1", "u1", "A1", "assistant"),
            ],
            head_id: "a1",
        };
        const result = await runAgent(loop, { trigger: clientMessage("U2"), messageTree: tree });
        expect(appendedMessages(result).map((m) => m.content)).toEqual(["U2"]);
        // The new turn is the tail of the transcript, continuing the active path.
        expect(result.transcript.map((m) => m.content)).toEqual(["SYS", "U1", "A1", "U2"]);
        expect(promptOf(result)).toEqual(["SYS", "U1", "A1", "U2"]);
    });

    it("does not re-seed system when the active path already leads with one", async () => {
        const tree: MessageTree = {
            nodes: [node("s1", undefined, "SYS-IN-TREE", "system"), node("u1", "s1", "U1")],
            head_id: "u1",
        };
        const result = await runAgent(loop, { trigger: clientMessage("U2"), messageTree: tree });
        expect(appendedMessages(result).map((m) => m.role)).toEqual(["user"]);
        expect(promptOf(result)).toEqual(["SYS-IN-TREE", "U1", "U2"]);
    });

    it("omits the system message when no instructions are given", async () => {
        const bare = toolLoop({ llm: { model: "test-model" } });
        const tree: MessageTree = { nodes: [node("u1", undefined, "U1")], head_id: "u1" };
        const result = await runAgent(bare, { trigger: clientMessage("U2"), messageTree: tree });
        expect(appendedMessages(result).map((m) => m.content)).toEqual(["U2"]);
        expect(promptOf(result)).toEqual(["U1", "U2"]);
    });
});
