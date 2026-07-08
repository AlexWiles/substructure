import type {
    LlmParams,
    LlmRequest,
    LlmResponse,
    LlmTokenDeltaInput,
    Message,
    MessageInput,
    MessageTree,
    RetryPolicy,
    ToolHandler,
    WorkerAction,
    WireDecisionRequest,
} from "./types";
import { nodeId } from "./types";

// ── Agent, decision, return ──────────────────────────────────────────────────

/** Push one streamed token delta to whoever is listening. */
export type EmitDelta = (delta: LlmTokenDeltaInput) => Promise<void>;

/** What the engine sends the agent: the wire envelope, with absent `state` normalized to `null`. */
export type DecisionRequest<S = unknown> = WireDecisionRequest & {
    state: S | null;
    /** Stream token deltas, present on a streaming `llm.execute` trigger. */
    emitDelta?: EmitDelta;
};

/** What the agent decides. `actions` default to none, `messages` to the request's;
 *  omit `state` to keep the current one, or return a value to write it (`{}` clears).
 *  Only return a `state` you mean to set — a forking decision carries it to the new branch. */
export interface Decision {
    actions?: WorkerAction[];
    messages?: MessageInput[];
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
/** `request` is the decision the tool runs under (session, identity, `tool.execute` trigger). */
export type ToolFn = (args: string, request: DecisionRequest) => ToolResult;
/** A deferred tool's `execute` only starts the work; its return is ignored and the
 *  call stays pending until `settleEffect` delivers the result. */
export type DeferredToolFn = (args: string, request: DecisionRequest) => void | Promise<void>;

export interface ToolDef {
    name: string;
    description: string;
    parameters: unknown;
    execute: ToolFn | DeferredToolFn;
    retry?: RetryPolicy;
    /** "worker" (default) runs `execute` here; "client" routes the call to the frontend, which settles it. */
    handler?: ToolHandler;
    /** `execute` kicks off out-of-band work; the call stays pending until `settleEffect` delivers the result. Worker-only. */
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

/** The LLM a loop calls. A server LLM omits `run`; a worker LLM sets `handler: "worker"` and supplies `run`. */
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
