// Pure-function harness for middleware composition tests.
//
// A middleware chain is just `(trigger, state) -> { actions, state }` with no
// I/O. `runChain` composes raw middlewares (the same fold as the worker's
// `composeChain`) and runs a single trigger against a plain-object state — no
// base64, no `jsonState`, no engine. Deterministic and synchronous.

import type {
    DecisionTrigger,
    LlmTokenDeltaInput,
    Message,
    ToolCall,
    ToolResult,
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
        request: makeRequest(opts.trigger),
    };
    const result = await fn(ctx);
    return { actions: result.actions, state: result.state };
}

function makeRequest(trigger: DecisionTrigger): WorkerDecisionRequestWire {
    return {
        session_id: "00000000-0000-0000-0000-000000000000",
        tenant_id: "test",
        decision_id: "decision-0",
        agent_id: "test-agent",
        identity: { tenant_id: "test", id: "tester" },
        trigger,
        worker_state: "",
        span: { trace_id: "0".repeat(32), span_id: "0".repeat(16), trace_flags: 1 },
        attempts: 0,
    };
}

// ── Trigger builders ─────────────────────────────────────────────────────────

export function userMessage(content: string, stream = false): DecisionTrigger {
    return { type: "user.message", stream, message: { role: "user", content } };
}

export function llmResponse(message: Message, callId = "call-0"): DecisionTrigger {
    return { type: "llm.response", call_id: callId, message, truncated: false };
}

export function toolCall(name: string, args: unknown, id = "tc-0"): ToolCall {
    return { id, type: "function", function: { name, arguments: JSON.stringify(args) } };
}

export function toolResult(toolCallId: string, content: string, name = "", isError = false): ToolResult {
    return { tool_call_id: toolCallId, name, content, is_error: isError };
}

/** The engine's batched delivery of all of a turn's tool + sub-agent results. */
export function effectsComplete(results: ToolResult[]): DecisionTrigger {
    return { type: "effects.complete", results };
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

/** The conversation history retained in the returned worker state. */
export function historyMessages(result: RunResult, key = "messages"): Message[] {
    const state = result.state as Record<string, Message[] | undefined>;
    return state[key] ?? [];
}
