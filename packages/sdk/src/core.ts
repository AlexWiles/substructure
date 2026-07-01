import type {
    DecisionTrigger,
    LlmHandler,
    LlmRequest,
    LlmResponse,
    LlmTokenDeltaInput,
    Message,
    MessageTree,
    ReasoningConfig,
    RetryPolicy,
    StreamPart,
    ToolHandler,
    WorkerAction,
    WorkerDecisionRequestWire,
} from "./types";
import { nodeId } from "./types";

// ── Agent, decision, return ──────────────────────────────────────────────────

/** What the engine sends the agent: the wire envelope with `worker_state` decoded
 *  into `state`. Everything else (`trigger`, `transcript`, `pending`, `session_id`,
 *  `identity`, …) is the envelope, read directly. */
export type DecisionRequest<S = unknown> = WorkerDecisionRequestWire & {
    state: S;
    /** Stream token deltas — present on an `llm.request` trigger with streaming. */
    emitDelta?: (delta: LlmTokenDeltaInput) => Promise<void>;
};

/** What the agent decides: the actions to take, plus the transcript and state to
 *  persist. `actions` defaults to none; `transcript`/`state` default to echoing
 *  the request's. */
export interface Decision {
    actions?: WorkerAction[];
    transcript?: Message[];
    state?: unknown;
}

/** An agent: `decide` takes a `DecisionRequest` and returns a `Decision`. Compose
 *  it by calling it; `agent({ name, ... })` it to deploy or use as a sub-agent. */
export interface Agent<S = unknown> {
    (req: DecisionRequest<S>): Decision | Promise<Decision>;
    /** The id this agent is deployed and addressed under. */
    agentName?: string;
}

export const DEFAULT_RETRY: RetryPolicy = {
    timeout_secs: 120,
    max_retries: 0,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

// ── Tools ────────────────────────────────────────────────────────────────────

/**
 * Sentinel from `ctx.defer()`: the worker emits no result for this `tool.execute`,
 * and the engine leaves the call pending until `submitToolCallResult` is called.
 */
export const DEFERRED: unique symbol = Symbol.for("substructure.tool.deferred");
export type Deferred = typeof DEFERRED;

export interface ToolExecutionContext {
    sessionId: string;
    toolCallId: string;
    attempt: number;
    /** The decision request — read `request.identity.id`, etc. */
    request: DecisionRequest;
    /** Signal out-of-band completion: `return ctx.defer();`. */
    defer: () => Deferred;
}

export type ToolResult = string | Deferred | Promise<string | Deferred>;
export type ToolFn = (args: string, ctx: ToolExecutionContext) => ToolResult;

export interface ToolDef {
    name: string;
    description: string;
    parameters: unknown;
    execute: ToolFn;
    retry?: RetryPolicy;
    /** "worker" (default) runs `execute` on the worker; "client" routes the call
     *  to the frontend, which completes it via `submitToolCallResult` — the worker
     *  never runs `execute` for a client tool. */
    handler?: ToolHandler;
}

// A tool returns its result string (call `JSON.stringify` yourself for structured
// data) or `ctx.defer()` to complete out-of-band. State, if any, lives in your own
// store — reach it through `ctx` (e.g. keyed by `ctx.sessionId`). `handler`
// discriminates: "worker" (default) runs `execute`; "client" completes in the
// browser, so `execute` is optional and never runs on the worker.
export function tool(
    config: {
        name: string;
        description: string;
        parameters: unknown;
        retry?: RetryPolicy;
    } & ({ handler?: "worker"; execute: ToolFn } | { handler: "client"; execute?: ToolFn }),
): ToolDef {
    return {
        name: config.name,
        description: config.description,
        parameters: config.parameters,
        execute:
            config.execute ??
            (() => {
                throw new Error(`Tool "${config.name}" has no server-side execute.`);
            }),
        retry: config.retry,
        handler: config.handler,
    };
}

// ── Model ────────────────────────────────────────────────────────────────────

/** One worker-side LLM call (a generator's `run`); stream deltas via `ctx.emitDelta`. */
export type LlmGenerate = (
    request: LlmRequest,
    ctx: { emitDelta?: (part: StreamPart) => Promise<void> },
) => Promise<LlmResponse>;

/** A bound LLM backend. A worker generator supplies `run` (the call runs on your
 *  worker); a server generator omits it (the Substructure server calls its
 *  configured provider). */
export interface LlmGenerator {
    request: Omit<LlmRequest, "messages" | "tools">;
    handler: LlmHandler;
    stream?: boolean;
    run?: LlmGenerate;
}

/** `model` is the server's provider model id, e.g. "anthropic/claude-sonnet-4-6". */
export function serverGenerate(settings: {
    model: string;
    temperature?: number;
    maxTokens?: number;
    reasoning?: ReasoningConfig;
    stream?: boolean;
}): LlmGenerator {
    const request: Omit<LlmRequest, "messages" | "tools"> = { model: settings.model };
    if (settings.temperature !== undefined) request.temperature = settings.temperature;
    if (settings.maxTokens !== undefined) request.max_completion_tokens = settings.maxTokens;
    if (settings.reasoning !== undefined) request.reasoning = settings.reasoning;
    return { request, handler: "server", stream: settings.stream ?? false };
}

// ── Transcript helpers ───────────────────────────────────────────────────────

/** The `leaf`-to-root path as messages, with control nodes (interrupts) filtered out. */
export function pathTo(tree: MessageTree | undefined, leaf: string | undefined): Message[] {
    if (!tree || leaf == null) return [];
    const out: Message[] = [];
    let target: string | undefined = leaf;
    for (let i = tree.nodes.length - 1; i >= 0 && target != null; i--) {
        const node = tree.nodes[i];
        if (nodeId(node) !== target) continue;
        if (node.kind === "message") out.push(node.message);
        target = node.parent_id;
    }
    return out.reverse();
}

/** The active transcript: the `head_id`-to-root path. */
export function activePath(tree?: MessageTree): Message[] {
    return pathTo(tree, tree?.head_id);
}

/** The tool-result for `toolCallId` on the active path, so a stale branch's result doesn't shadow the live one. */
export function toolResultNode(tree: MessageTree | undefined, toolCallId: string): Message | undefined {
    return activePath(tree).find((m) => m.role === "tool" && m.tool_call_id === toolCallId);
}

/** Stamp a fresh node id onto a message so the engine records it as a new node. */
export function stamp(message: Message): Message {
    return { ...message, id: crypto.randomUUID() };
}

/** The single message a trigger carries (`user.message`/`llm.response`), or `null`. */
export function triggerToMessage(trigger: DecisionTrigger): Message | null {
    return trigger.type === "llm.response" || trigger.type === "user.message" ? trigger.message : null;
}

/** All messages a trigger carries into the tree. */
export function triggerToMessages(trigger: DecisionTrigger): Message[] {
    switch (trigger.type) {
        case "user.message":
        case "llm.response":
            return [trigger.message];
        case "user.transcript":
            return trigger.messages;
        default:
            return [];
    }
}

// ── Stop conditions ──────────────────────────────────────────────────────────

export interface StopInfo<S = unknown> {
    /** Assistant rounds taken since the last user message. */
    steps: number;
    /** The assistant message whose tool calls just ran. */
    lastResponse: Message | undefined;
    /** The active transcript. */
    history: Message[];
    state: S;
}

export type StopCondition<S = unknown> = (info: StopInfo<S>) => boolean | Promise<boolean>;

export const stepCountIs =
    (n: number): StopCondition =>
    (info) =>
        info.steps >= n;
