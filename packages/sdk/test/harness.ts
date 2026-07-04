// Pure-function harness for agent tests.
//
// An agent is `(DecisionRequest) -> Decision`. `runAgent` builds a request from a
// trigger (+ optional tree/pending/state) and runs the agent against it,
// mirroring the worker boundary, passing `state` straight through. Deterministic.

import type { Agent, DecisionRequest } from "../src/core";
import { activePath } from "../src/core";
import type {
    DecisionTrigger,
    Effect,
    LlmTokenDeltaInput,
    Message,
    MessageTree,
    Node,
    ToolCall,
    WorkerAction,
    WorkerDecisionRequestWire,
} from "../src/types";
import { nodeId } from "../src/types";

export interface RunResult {
    actions: WorkerAction[];
    state: unknown;
    transcript: Message[];
    /** Node ids present in the input tree, so helpers can isolate the new tail. */
    inputIds: Set<string>;
}

export interface RunOptions {
    trigger: DecisionTrigger;
    state?: Record<string, unknown>;
    messageTree?: MessageTree;
    /** In-flight effects the engine would surface on the request (default none). */
    effects?: Effect[];
    emitDelta?: (delta: LlmTokenDeltaInput) => Promise<void>;
}

export async function runAgent(agent: Agent, opts: RunOptions): Promise<RunResult> {
    const transcript = opts.messageTree ? activePath(opts.messageTree) : undefined;
    const req: DecisionRequest = {
        ...makeRequest(opts, transcript),
        state: opts.state ?? {},
        emitDelta: opts.emitDelta,
    };
    const out = await agent(req);

    const inputIds = new Set<string>();
    for (const node of opts.messageTree?.nodes ?? []) {
        const id = nodeId(node);
        if (id) inputIds.add(id);
    }
    return {
        actions: out.actions ?? [],
        state: out.state !== undefined ? out.state : req.state,
        transcript: out.transcript ?? req.transcript ?? [],
        inputIds,
    };
}

function makeRequest(opts: RunOptions, transcript?: Message[]): WorkerDecisionRequestWire {
    const effects = opts.effects ?? [];
    return {
        session_id: "00000000-0000-0000-0000-000000000000",
        tenant_id: "test",
        decision_id: "decision-0",
        agent_id: "test-agent",
        identity: { tenant_id: "test", id: "tester" },
        trigger: opts.trigger,
        worker_state: {},
        effects,
        pending_effects: effects.filter(
            (e) =>
                (e.kind === "tool_call" || e.kind === "sub_agent") &&
                (e.status === "pending" || e.status === "retry_scheduled"),
        ).length,
        transcript,
        message_tree: opts.messageTree,
        span: { trace_id: "0".repeat(32), span_id: "0".repeat(16), trace_flags: 1 },
        attempts: 0,
    };
}

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

export function userMessage(content: string): DecisionTrigger {
    return { type: "user.message", message: { id: nextNodeId(), role: "user", content } };
}

export function userTranscript(messages: Message[]): DecisionTrigger {
    return { type: "user.transcript", messages };
}

export function llmResponse(message: Message, callId = "call-0"): DecisionTrigger {
    return { type: "effect.settled", id: callId, ok: true, kind: "llm_call", message, truncated: false };
}

export function toolCall(name: string, args: unknown, id = "tc-0"): ToolCall {
    return { id, type: "function", function: { name, arguments: JSON.stringify(args) } };
}

/** One tool completion; the worker records its result in the transcript.
 *  Whether it then prompts is driven by the `tool_call`/`sub_agent` effects still in flight. */
export function toolResult(toolCallId: string, name: string, result: string, isError = false): DecisionTrigger {
    return {
        type: "effect.settled",
        id: toolCallId,
        ok: !isError,
        kind: "tool_call",
        name,
        result,
    };
}

/** One sub-agent completion; `id` is the child session, `tool_call_id` the model call it answers. */
export function subAgentResult(
    sessionId: string,
    toolCallId: string,
    agentId: string,
    result: string,
    isError = false,
): DecisionTrigger {
    return {
        type: "effect.settled",
        id: sessionId,
        ok: !isError,
        kind: "sub_agent",
        tool_call_id: toolCallId,
        agent_id: agentId,
        result,
    };
}

// ── Transcript helpers ───────────────────────────────────────────────────────

/** The new tail of `result.transcript`: messages absent from the input tree
 *  (id-less messages count as new), in order; the engine appends exactly these. */
export function appendedMessages(result: RunResult): Message[] {
    return result.transcript.filter((m) => m.id == null || !result.inputIds.has(m.id));
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
