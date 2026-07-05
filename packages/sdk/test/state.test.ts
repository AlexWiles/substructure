// State opinion semantics at the worker boundary and in toolLoop.
//
// The engine treats an omitted `state` as "keep the current state". The boundary
// must therefore only send `state` when the agent expressed one, and toolLoop
// must express none — otherwise every fork would silently carry the abandoned
// branch's state (the engine's echo-dedup can't catch a cross-branch echo).

import { describe, expect, it } from "vitest";
import { toolLoop } from "../src/agent";
import type { Agent } from "../src/core";
import type { WorkerDecisionRequestWire } from "../src/types";
import { Worker } from "../src/worker";
import { linearTree, clientMessage, clientTranscript } from "./harness";

function wireRequest(overrides: Partial<WorkerDecisionRequestWire> = {}): WorkerDecisionRequestWire {
    return {
        session_id: "00000000-0000-0000-0000-000000000000",
        tenant_id: "test",
        decision_id: "decision-0",
        agent_id: "a",
        identity: { tenant_id: "test", id: "tester" },
        trigger: clientMessage("hi"),
        state: null,
        effects: [],
        pending_effects: 0,
        transcript: [],
        span: { trace_id: "0".repeat(32), span_id: "0".repeat(16), trace_flags: 1 },
        attempts: 0,
        ...overrides,
    };
}

function serve(decide: Agent): Worker {
    return new Worker([Object.assign(decide, { agentName: "a" })]);
}

describe("state at the worker boundary", () => {
    it("delivers state to the agent untouched, null included", async () => {
        let seen: unknown = "unset";
        const w = serve((req) => {
            seen = req.state;
            return {};
        });

        await w.handleDecision(wireRequest({ state: { memory: ["fact"] } }));
        expect(seen).toEqual({ memory: ["fact"] });

        await w.handleDecision(wireRequest({ state: null }));
        expect(seen).toBeNull();
    });

    it("omits state from the submit when the agent expressed no opinion", async () => {
        const w = serve(() => ({}));
        const submit = await w.handleDecision(wireRequest({ state: { v: 1 } }));
        expect(submit.state).toBeUndefined();
        expect("state" in submit && submit.state !== undefined).toBe(false);
    });

    it("forwards the state the agent returned verbatim, null included", async () => {
        // The boundary is faithful: a returned value is sent, a returned `null` is
        // sent as `null`. But `null` is NOT a clear — the engine's `Option<WorkerState>`
        // folds a wire `null` into "keep", indistinguishable from an omitted field.
        // Only a present non-null empty value (e.g. `{}`) clears.
        const w = serve(() => ({ state: { v: 2 } }));
        expect((await w.handleDecision(wireRequest())).state).toEqual({ v: 2 });

        const returnsNull = serve(() => ({ state: null }));
        const submit = await returnsNull.handleDecision(wireRequest({ state: { v: 1 } }));
        expect(submit.state).toBeNull();

        const clearing = serve(() => ({ state: {} }));
        const cleared = await clearing.handleDecision(wireRequest({ state: { v: 1 } }));
        expect(cleared.state).toEqual({});
    });
});

describe("toolLoop state opinion", () => {
    const loop = toolLoop({ llm: { model: "test-model" } });

    it("expresses no state on a user message", async () => {
        const w = serve(loop);
        const submit = await w.handleDecision(wireRequest({ state: { memory: "old" } }));
        expect(submit.state).toBeUndefined();
        expect(submit.actions.length).toBeGreaterThan(0);
    });

    it("expresses no state on a forking client.transcript, so the engine resolves as-of the fork", async () => {
        const tree = linearTree(
            { id: "", role: "user", content: "one" },
            { id: "", role: "assistant", content: "two" },
            { id: "", role: "user", content: "three" },
        );
        // The engine delivers the old branch's state on the forking decision;
        // toolLoop must not hand it back.
        const w = serve(loop);
        const submit = await w.handleDecision(
            wireRequest({
                trigger: clientTranscript([
                    { id: "n0", role: "user", content: "one" },
                    { id: "", role: "user", content: "edited" },
                ]),
                state: { memory: "from the abandoned branch" },
                message_tree: tree,
            }),
        );
        expect(submit.state).toBeUndefined();
    });

    it("a wrapping agent persists its own state alongside the loop's decision", async () => {
        const wrapper: Agent = async (req) => {
            const state = { count: ((req.state as { count?: number } | null)?.count ?? 0) + 1 };
            return { ...(await loop({ ...req, state })), state };
        };
        const w = serve(wrapper);
        const submit = await w.handleDecision(wireRequest({ state: { count: 41 } }));
        expect(submit.state).toEqual({ count: 42 });
    });
});
