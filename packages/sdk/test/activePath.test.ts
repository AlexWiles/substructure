import { describe, expect, it } from "vitest";

import { activePath, llm, serverGenerate } from "../src/middleware";
import type { Message, MessageTree, Node } from "../src/types";
import { callLlm, runChain, userMessage } from "./harness";

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
        // u2/a2 were abandoned when u2b forked off a1; head points at u2b.
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

describe("llm prompt", () => {
    const loop = llm({ generator: serverGenerate({ model: "test-model" }), instructions: "SYS" });
    const promptOf = (result: Awaited<ReturnType<typeof runChain>>) =>
        (callLlm(result)?.request.messages ?? []).map((m) => m.content);

    it("seeds [system, user] on the first turn (empty tree)", async () => {
        const result = await runChain([loop], { trigger: userMessage("hi") });
        expect(promptOf(result)).toEqual(["SYS", "hi"]);
    });

    it("appends the new user turn after the system and the active path", async () => {
        const tree: MessageTree = {
            nodes: [node("u1", undefined, "U1"), node("a1", "u1", "A1", "assistant")],
            head_id: "a1",
        };
        const result = await runChain([loop], { trigger: userMessage("U2"), messageTree: tree });
        expect(promptOf(result)).toEqual(["SYS", "U1", "A1", "U2"]);
    });

    it("does not re-seed system when the active path already leads with one", async () => {
        const tree: MessageTree = {
            nodes: [node("s1", undefined, "SYS-IN-TREE", "system"), node("u1", "s1", "U1")],
            head_id: "u1",
        };
        const result = await runChain([loop], { trigger: userMessage("U2"), messageTree: tree });
        expect(promptOf(result)).toEqual(["SYS-IN-TREE", "U1", "U2"]);
    });

    it("omits the system message when no instructions are given", async () => {
        const bare = llm({ generator: serverGenerate({ model: "test-model" }) });
        const tree: MessageTree = { nodes: [node("u1", undefined, "U1")], head_id: "u1" };
        const result = await runChain([bare], { trigger: userMessage("U2"), messageTree: tree });
        expect(promptOf(result)).toEqual(["U1", "U2"]);
    });
});
