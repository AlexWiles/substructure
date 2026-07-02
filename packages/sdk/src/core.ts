import type {
    LlmParams,
    LlmRequest,
    LlmResponse,
    LlmTokenDeltaInput,
    Message,
    MessageTree,
    RetryPolicy,
    ToolHandler,
    WorkerAction,
    WorkerDecisionRequestWire,
} from "./types";
import { nodeId } from "./types";

// ── Agent, decision, return ──────────────────────────────────────────────────

/** Push one streamed token delta to whoever is listening (SSE client, embedded
 *  runtime). Present on a streaming llm `effect.execute`; a worker-run model
 *  calls it. */
export type EmitDelta = (delta: LlmTokenDeltaInput) => Promise<void>;

/** What the engine sends the agent: the wire envelope with `worker_state` decoded
 *  into `state`. Everything else (`trigger`, `transcript`, `effects`, `session_id`,
 *  `identity`, …) is the envelope, read directly. `effects` lists the in-flight
 *  effects by kind — the step gate is "no tool/sub-agent effect left". */
export type DecisionRequest<S = unknown> = WorkerDecisionRequestWire & {
    state: S;
    /** Stream token deltas — present on a streaming llm `effect.execute` trigger. */
    emitDelta?: EmitDelta;
};

/** What the agent decides: the actions to take, plus the transcript and state to
 *  persist. `actions` defaults to none; `transcript`/`state` default to echoing
 *  the request's. */
export interface Decision {
    actions?: WorkerAction[];
    transcript?: Message[];
    state?: unknown;
}

/** An agent: takes a `DecisionRequest` and returns a `Decision`. Compose it by
 *  calling it; `agent({ name, decide })` names it into a `NamedAgent`. */
export type Agent<S = unknown> = (req: DecisionRequest<S>) => Decision | Promise<Decision>;

/** A named agent — what `agent({ name, decide })` returns. `agentName` is the id
 *  the engine deploys and addresses it under, so only a `NamedAgent` can be served
 *  by `worker([...])` or delegated to as a sub-agent. */
export interface NamedAgent<S = unknown> extends Agent<S> {
    agentName: string;
}

// ── Tools ────────────────────────────────────────────────────────────────────

/**
 * Sentinel from `ctx.defer()`: the worker emits no result for this `effect.execute`,
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

// ── LLM ──────────────────────────────────────────────────────────────────────

/** One worker-side LLM call (an `Llm`'s `run`); stream deltas via `ctx.emitDelta`. */
export type LlmGenerate = (request: LlmRequest, ctx: { emitDelta?: EmitDelta }) => Promise<LlmResponse>;

/** The LLM a loop calls: the model and per-call params (`LlmParams`) plus how the
 *  call runs. A server LLM omits `run` — the Substructure server calls its
 *  configured provider; a worker LLM sets `handler: "worker"` and supplies `run`
 *  to make the call on your worker. The two are a union, so `handler: "worker"`
 *  without a `run` is a compile error. */
export type Llm = LlmParams & { stream?: boolean } & (
        | { handler?: "server"; run?: never }
        | { handler: "worker"; run: LlmGenerate }
    );

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

/** Stamp a fresh node id onto a message so the engine records it as a new node. */
export function stamp(message: Message): Message {
    return { ...message, id: crypto.randomUUID() };
}
