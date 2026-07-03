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

/** Push one streamed token delta to whoever is listening (SSE client, embedded runtime). */
export type EmitDelta = (delta: LlmTokenDeltaInput) => Promise<void>;

/** What the engine sends the agent: the wire envelope with `worker_state` decoded into `state`. */
export type DecisionRequest<S = unknown> = WorkerDecisionRequestWire & {
    state: S;
    /** Stream token deltas, present on a streaming llm `effect.execute` trigger. */
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

/** A named agent: what `agent({ name, decide })` returns. `agentName` is the id
 *  the engine deploys and addresses it under, so only a `NamedAgent` can be served
 *  by `worker([...])` or delegated to as a sub-agent. */
export interface NamedAgent<S = unknown> extends Agent<S> {
    agentName: string;
}

// ── Tools ────────────────────────────────────────────────────────────────────

export type ToolResult = string | Promise<string>;
/** `request` is the decision request the tool runs under; read `request.session_id`,
 *  `request.identity.id`, or the `effect.execute` trigger (`request.trigger.id`,
 *  `request.trigger.attempt`) it was dispatched from. */
export type ToolFn = (args: string, request: DecisionRequest) => ToolResult;
/** A deferred tool's `execute` only *starts* the work (enqueue a job, fire a
 *  webhook); its return value is ignored. The call stays pending until
 *  `settleEffect` delivers the result out-of-band. */
export type DeferredToolFn = (args: string, request: DecisionRequest) => void | Promise<void>;

export interface ToolDef {
    name: string;
    description: string;
    parameters: unknown;
    execute: ToolFn | DeferredToolFn;
    retry?: RetryPolicy;
    /** "worker" (default) runs `execute` on the worker; "client" routes the call
     *  to the frontend, which completes it via `settleEffect`; the worker
     *  never runs `execute` for a client tool. */
    handler?: ToolHandler;
    /** A deferred tool's `execute` kicks off out-of-band work: the loop emits no
     *  result action, the call stays pending, and the result arrives later via
     *  `settleEffect`. Worker-handled only. */
    deferred?: boolean;
}

export function tool(
    config: {
        name: string;
        description: string;
        parameters: unknown;
        retry?: RetryPolicy;
    } & (
        | { handler?: "worker"; deferred?: false; execute: ToolFn }
        | { handler?: "worker"; deferred: true; execute: DeferredToolFn }
        | { handler: "client"; deferred?: never; execute?: ToolFn }
    ),
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
        deferred: config.deferred || undefined,
    };
}

// ── LLM ──────────────────────────────────────────────────────────────────────

/** One worker-side LLM call (an `Llm`'s `run`); stream deltas via `ctx.emitDelta`. */
export type LlmGenerate = (request: LlmRequest, ctx: { emitDelta?: EmitDelta }) => Promise<LlmResponse>;

/** The LLM a loop calls. A server LLM omits `run` (the server calls its provider);
 *  a worker LLM sets `handler: "worker"` and supplies `run` to make the call on your worker. */
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
