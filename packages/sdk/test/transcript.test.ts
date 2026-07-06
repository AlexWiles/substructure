import { describe, expect, it } from "vitest";

import { toolLoop } from "../src/agent";
import type { Message } from "../src/types";
import { appendedMessages, callLlm, clientTranscript, linearTree, runAgent } from "./harness";

// The AG-UI `/run` path: the client sends its full transcript (with the ids it
// knows) and the worker reconciles the new/branched tail into the tree.
const loop = toolLoop({ llm: { model: "test-model" }, instructions: "SYS" });

const prompt = (r: Awaited<ReturnType<typeof runAgent>>) => (callLlm(r)?.request.messages ?? []).map((m) => m.content);

describe("client.transcript (AG-UI reconcile)", () => {
    it("cold start roots [system, user] from an id-less transcript", async () => {
        const result = await runAgent(loop, {
            trigger: clientTranscript([{ role: "user", content: "hi" }]),
        });

        expect(appendedMessages(result).map((m) => m.role)).toEqual(["system", "user"]);
        expect(prompt(result)).toEqual(["SYS", "hi"]);
    });

    it("appends only the new tail and reuses the existing system root", async () => {
        // Tree already holds [system n0, user n1, assistant n2]; the client view
        // (no system) re-sends n1/n2 and adds a fresh user message.
        const messageTree = linearTree(
            { role: "system", content: "SYS" },
            { role: "user", content: "U1" },
            { role: "assistant", content: "A1" },
        );
        const view: Message[] = [
            { id: "n1", role: "user", content: "U1" },
            { id: "n2", role: "assistant", content: "A1" },
            { role: "user", content: "U2" },
        ];
        const result = await runAgent(loop, { trigger: clientTranscript(view), messageTree });

        // Only the new user message is new; it is the tail of the returned transcript.
        const appended = appendedMessages(result);
        expect(appended.map((m) => m.content)).toEqual(["U2"]);
        expect(result.transcript.map((m) => m.content)).toEqual(["SYS", "U1", "A1", "U2"]);
        // The reused system root carries its committed id, so the engine continues
        // (rather than forks) when it reconciles the transcript.
        expect(result.transcript[0]).toMatchObject({ role: "system", id: "n0" });

        // The prompt is the reconciled path, rooted at the existing system node.
        expect(prompt(result)).toEqual(["SYS", "U1", "A1", "U2"]);
    });
});
