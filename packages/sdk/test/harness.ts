// Pure-function harness for middleware composition tests.
//
// A middleware chain is just `(trigger, state) -> { actions, state }` with no
// I/O. `runChain` composes raw middlewares (the same fold as the worker's
// `composeChain`) and runs a single trigger against a plain-object state — no
// base64, no `jsonState`, no engine. Deterministic and synchronous.

import type {
    CompletedToolCall,
    DecisionTrigger,
    LlmTokenDeltaInput,
    Message,
    MessageTree,
    Node,
    ToolCall,
    WorkerAction,
    WorkerDecisionRequestWire,
} from "../src/types";
import type { AgentContext, MiddlewareFn, Next } from "../src/worker";

const FALLBACK: Next<unknown> = (ctx) => ({ actions: [], state: ctx.state });

export interface RunResult {
    actions: WorkerAction[];
    state: unknown;
}

export interface RunOptions {
    trigger: DecisionTrigger;
    state?: Record<string, unknown>;
    /** The conversation tree the engine ships on the request. */
    messageTree?: MessageTree;
    emitDelta?: (delta: LlmTokenDeltaInput) => Promise<void>;
}

export async function runChain(middlewares: MiddlewareFn<any, any>[], opts: RunOptions): Promise<RunResult> {
    let fn: Next<unknown> = FALLBACK;
    for (let i = middlewares.length - 1; i >= 0; i--) {
        const mw = middlewares[i];
        const next = fn;
        fn = (ctx) => mw(ctx, next);
    }
    const ctx: AgentContext<unknown> = {
        state: structuredClone(opts.state ?? {}),
        emitDelta: opts.emitDelta,
        request: makeRequest(opts.trigger, opts.messageTree),
    };
    const result = await fn(ctx);
    return { actions: result.actions, state: result.state };
}

function makeRequest(trigger: DecisionTrigger, messageTree?: MessageTree): WorkerDecisionRequestWire {
    return {
        session_id: "00000000-0000-0000-0000-000000000000",
        tenant_id: "test",
        decision_id: "decision-0",
        agent_id: "test-agent",
        identity: { tenant_id: "test", id: "tester" },
        trigger,
        worker_state: "",
        message_tree: messageTree,
        span: { trace_id: "0".repeat(32), span_id: "0".repeat(16), trace_flags: 1 },
        attempts: 0,
    };
}

/** Build a linear (unbranched) conversation tree from messages in order. */
export function linearTree(...messages: Message[]): MessageTree {
    const nodes: Node[] = messages.map((message, i) => ({
        kind: "message",
        parent_id: i === 0 ? undefined : `n${i - 1}`,
        message: { ...message, id: `n${i}` },
    }));
    return { nodes, head_id: nodes.at(-1) ? `n${messages.length - 1}` : undefined };
}

// ── Trigger builders ─────────────────────────────────────────────────────────

let nodeCounter = 0;
function nextNodeId(): string {
    return `node-${nodeCounter++}`;
}

export function userMessage(content: string, stream = false): DecisionTrigger {
    return { type: "user.message", stream, messages: [{ role: "user", content }], anchor: "continue" };
}

export function llmResponse(message: Message, callId = "call-0"): DecisionTrigger {
    return { type: "llm.response", call_id: callId, message, truncated: false, id: nextNodeId() };
}

export function toolCall(name: string, args: unknown, id = "tc-0"): ToolCall {
    return { id, type: "function", function: { name, arguments: JSON.stringify(args) } };
}

/** A tool/sub-agent completion. The result is already a node in the tree; the
 *  trigger just names what landed (the default loop reads continuation off the
 *  tree, so `completed` is irrelevant to it). */
export function toolResults(completed: CompletedToolCall[] = []): DecisionTrigger {
    return { type: "tool.results", completed };
}

// ── Assertion helpers ────────────────────────────────────────────────────────

export function actionsOfType<T extends WorkerAction["type"]>(
    result: RunResult,
    type: T,
): Extract<WorkerAction, { type: T }>[] {
    return result.actions.filter((a): a is Extract<WorkerAction, { type: T }> => a.type === type);
}

export function callLlm(result: RunResult): Extract<WorkerAction, { type: "call.llm" }> | undefined {
    return actionsOfType(result, "call.llm")[0];
}

export function toolMessages(result: RunResult, toolCallId: string): Message[] {
    return (callLlm(result)?.request.messages ?? []).filter((m) => m.role === "tool" && m.tool_call_id === toolCallId);
}
