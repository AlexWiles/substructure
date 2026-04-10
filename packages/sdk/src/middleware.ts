import type { Message, LlmTool, LlmRequest, RetryPolicy, WorkerAction, ToolResult } from "./types";
import type { Handler, AgentRequest, AgentResponse, MiddlewareFn, Next, StateContributor } from "./worker";

export const DEFAULT_RETRY: RetryPolicy = {
    timeout_secs: 120,
    max_retries: 0,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

function decodeWorkerState(raw: string): unknown {
    if (!raw || raw === "") return {};
    return JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(raw), (c) => c.charCodeAt(0))));
}

function encodeWorkerState(value: unknown): string {
    return btoa(String.fromCharCode(...new TextEncoder().encode(JSON.stringify(value))));
}

/**
 * Base64 JSON state serialization middleware.
 * Decodes `wire.worker_state` into `req.state` (falling back to `{}`),
 * runs the chain, then encodes `res.state` back into `workerState`.
 */
export function state(): MiddlewareFn<unknown> {
    return async (req: AgentRequest<unknown>, next: Next<unknown>) => {
        const enriched: AgentRequest<unknown> = {
            ...req,
            state: decodeWorkerState(req.wire.worker_state),
        };
        const res = await next(enriched);
        return {
            ...res,
            workerState: encodeWorkerState(res.state),
        };
    };
}

/**
 * Create a StateContributor middleware that owns a slice of state.
 * The `init` object provides default values and the TypeScript type brand.
 * The middleware function receives a fully typed `req.state`.
 */
export function stateSlice<A extends object>(
    init: A,
    fn: (req: AgentRequest<A>, next: Next<A>) => Promise<AgentResponse> | AgentResponse,
): StateContributor<A> & MiddlewareFn<unknown, A> {
    return Object.assign(
        (req: AgentRequest<unknown>, next: Next<A>) => {
            const rawState = req.state;
            const state = (rawState && typeof rawState === "object" ? rawState : {}) as Record<string, unknown>;
            for (const key of Object.keys(init)) {
                state[key] ??= structuredClone((init as Record<string, unknown>)[key]);
            }
            const typedReq: AgentRequest<A> = {
                ...req,
                state: state as A,
            };
            return fn(typedReq, next);
        },
        { _contributes: init },
    ) as StateContributor<A> & MiddlewareFn<unknown, A>;
}

export type ToolFn = (args: string) => Promise<unknown>;

export interface ToolDef {
    name: string;
    description: string;
    parameters: unknown;
    execute: ToolFn;
    retry?: RetryPolicy;
}

/** Define a tool from plain data. */
export function tool(config: {
    name: string;
    description: string;
    parameters: unknown;
    execute: (args: string) => unknown | Promise<unknown>;
    retry?: RetryPolicy;
}): ToolDef {
    return {
        name: config.name,
        description: config.description,
        parameters: config.parameters,
        execute: async (args: string) => config.execute(args),
        retry: config.retry,
    };
}

export type LogLevel = "debug" | "info" | "warn" | "error";

export interface Logger {
    debug(msg: string, data?: Record<string, unknown>): void;
    info(msg: string, data?: Record<string, unknown>): void;
    warn(msg: string, data?: Record<string, unknown>): void;
    error(msg: string, data?: Record<string, unknown>): void;
}

export interface LoggingOptions {
    label?: string;
    logger?: Logger;
    level?: LogLevel;
}

const LOG_LEVELS: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

function defaultLogger(minLevel: LogLevel): Logger {
    const min = LOG_LEVELS[minLevel];
    const noop = () => {};
    const emit = (level: LogLevel) => {
        if (LOG_LEVELS[level] < min) return noop;
        return (msg: string, data?: Record<string, unknown>) => {
            console.log(JSON.stringify({ level, msg, ...data, ts: new Date().toISOString() }));
        };
    };
    return { debug: emit("debug"), info: emit("info"), warn: emit("warn"), error: emit("error") };
}

export function logging(options?: string | LoggingOptions): MiddlewareFn<unknown, unknown> {
    const opts: LoggingOptions = typeof options === "string" ? { label: options } : (options ?? {});
    const level = opts.level ?? "info";
    const log = opts.logger ?? defaultLogger(level);
    const label = opts.label;

    return async (req, next) => {
        const t = req.trigger;
        const tag = t.type === "tool.execute" ? `${t.type}:${t.name}` : t.type;
        const ctx: Record<string, unknown> = {
            agent: req.agentId,
            session: req.wire.session_id,
            decision: req.wire.decision_id,
            trigger: tag,
        };
        if (label) ctx.label = label;

        log.info("decision.start", ctx);
        log.debug("decision.trigger", { ...ctx, payload: t });

        const start = performance.now();
        try {
            const result = await next(req);
            const durationMs = Number((performance.now() - start).toFixed(1));
            const actionTypes = result.actions.map((a) => a.type);

            log.info("decision.end", { ...ctx, actions: actionTypes, durationMs });
            log.debug("decision.actions", { ...ctx, actions: result.actions, durationMs });

            return result;
        } catch (err) {
            const durationMs = Number((performance.now() - start).toFixed(1));
            log.error("decision.error", {
                ...ctx,
                error: err instanceof Error ? err.message : String(err),
                durationMs,
            });
            throw err;
        }
    };
}

export type MessageSelector<S> = (state: S, req: AgentRequest<S>) => Message[];

/**
 * Conversation history middleware. Contributes `{ messages: Message[] }` to state.
 * Records incoming messages and augments `call_llm` actions with the full history.
 */
export function messageHistory(): StateContributor<{ messages: Message[] }> &
    MiddlewareFn<unknown, { messages: Message[] }> {
    return stateSlice({ messages: [] as Message[] }, async (req, next) => {
        const history = req.state.messages;
        const { trigger } = req;

        switch (trigger.type) {
            case "user.message":
            case "llm.response":
                history.push(trigger.message);
                break;
            case "client.action":
                break;
            case "tool.result":
                history.push({
                    role: "tool",
                    content: trigger.result.content,
                    tool_call_id: trigger.result.tool_call_id,
                    name: trigger.result.name,
                });
                break;
        }

        const result = await next(req);

        const actions = result.actions.map((action) => {
            if (action.type !== "call.llm") return action;
            return {
                ...action,
                request: {
                    ...action.request,
                    messages: history,
                },
            };
        });

        return { ...result, actions };
    });
}

export type SystemMessageSelector<S> = (state: S, req: AgentRequest<S>) => string;

export function systemMessage<S>(selectorOrValue: SystemMessageSelector<S> | string): MiddlewareFn<S> {
    const selector: SystemMessageSelector<S> =
        typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;
    return async (req, next) => {
        const systemMessage: Message = { role: "system", content: selector(req.state, req) };

        const result = await next(req);
        const actions = result.actions.map((action) => {
            if (action.type !== "call.llm") return action;
            return {
                ...action,
                request: {
                    ...action.request,
                    messages: [systemMessage, ...action.request.messages],
                },
            };
        });

        return { ...result, actions };
    };
}

export type ToolInput = Record<string, ToolDef> | ToolDef[];
export type ToolSelector<S> = (state: S, req: AgentRequest<S>) => ToolInput;

function resolveTools(input: ToolInput): Record<string, ToolDef> {
    if (Array.isArray(input)) {
        const record: Record<string, ToolDef> = {};
        for (const def of input) {
            record[def.name] = def;
        }
        return record;
    }
    return input;
}

/**
 * Tool execution and pending-tracking middleware.
 * Contributes `{ pendingToolCalls: string[] }` to state.
 *
 * On `llm_response`: records tool_call_ids as pending.
 * On `tool_result`: removes the ID. Suppresses `call_llm` until all results are in.
 * On `tool_execute`: executes the tool and prepends the result to downstream actions.
 */
export function tools<S>(
    selectorOrValue: ToolSelector<S> | ToolInput,
): StateContributor<{ pendingToolCalls: string[] }> & MiddlewareFn<unknown, { pendingToolCalls: string[] }> {
    const selector: ToolSelector<S> = typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;
    return stateSlice({ pendingToolCalls: [] as string[] }, async (req, next) => {
        const tools = resolveTools(selector(req.state as S, req as unknown as AgentRequest<S>));

        const downstream = await next(req);

        // Handle tool execution
        if (req.trigger.type === "tool.execute") {
            const tool = tools[req.trigger.name];
            if (!tool) {
                return {
                    ...downstream,
                    actions: [
                        {
                            type: "return.tool.error" as const,
                            tool_call_id: req.trigger.tool_call_id,
                            error: `Unknown tool: ${req.trigger.name}`,
                            retryable: false,
                            attempt: req.trigger.attempt,
                        },
                        ...downstream.actions,
                    ],
                };
            }

            try {
                const output = await tool.execute(req.trigger.arguments);
                return {
                    ...downstream,
                    actions: [
                        {
                            type: "return.tool.result" as const,
                            tool_call_id: req.trigger.tool_call_id,
                            result: typeof output === "string" ? output : JSON.stringify(output),
                            attempt: req.trigger.attempt,
                        },
                        ...downstream.actions,
                    ],
                };
            } catch (error: unknown) {
                return {
                    ...downstream,
                    actions: [
                        {
                            type: "return.tool.error" as const,
                            tool_call_id: req.trigger.tool_call_id,
                            error: error instanceof Error ? error.message : String(error),
                            retryable: false,
                            attempt: req.trigger.attempt,
                        },
                        ...downstream.actions,
                    ],
                };
            }
        }

        // Track pending tool calls from LLM response and emit call_tool actions
        if (req.trigger.type === "llm.response") {
            const toolCalls = req.trigger.message.tool_calls;
            if (toolCalls && toolCalls.length > 0) {
                req.state.pendingToolCalls = toolCalls.map((tc) => tc.id);

                const callToolActions: WorkerAction[] = toolCalls.map((tc) => {
                    const def = tools[tc.function.name];
                    return {
                        type: "call.tool" as const,
                        tool_call_id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                        handler: "worker" as const,
                        retry: def?.retry ?? DEFAULT_RETRY,
                    };
                });

                return {
                    ...downstream,
                    actions: [...callToolActions, ...downstream.actions],
                };
            }
        }

        // Augment actions with tool definitions and retry policies
        const actions = downstream.actions.map((action) => {
            if (action.type === "call.llm") {
                return {
                    ...action,
                    request: {
                        ...action.request,
                        tools: mergeTools(
                            action.request.tools,
                            Object.entries(tools).map(([name, def]) => ({
                                function: { name, description: def.description, parameters: def.parameters },
                            })),
                        ),
                    },
                };
            }
            if (action.type === "call.tool") {
                const def = tools[action.name];
                if (def?.retry) {
                    return { ...action, retry: def.retry };
                }
            }
            return action;
        });

        // Suppress call_llm until all tool results are in
        if (req.trigger.type === "tool.result") {
            const resultId = req.trigger.result.tool_call_id;
            req.state.pendingToolCalls = req.state.pendingToolCalls.filter((id) => id !== resultId);
            if (req.state.pendingToolCalls.length > 0) {
                return {
                    ...downstream,
                    actions: actions.filter((a) => a.type !== "call.llm"),
                };
            }
        }

        return { ...downstream, actions };
    });
}

export interface LlmLoopSelection {
    request: Omit<LlmRequest, "messages"> & { messages?: Message[] };
    llm_client: string;
    retry?: RetryPolicy;
    stream?: boolean;
    toolRetries?: Record<string, RetryPolicy>;
}

export function llmLoop<S>(
    selectorOrValue: ((state: S, req: AgentRequest<S>) => LlmLoopSelection) | LlmLoopSelection,
): MiddlewareFn<S> {
    const selector: (state: S, req: AgentRequest<S>) => LlmLoopSelection =
        typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;
    return async (req, next) => {
        const selection = selector(req.state, req);
        const downstream = await next(req);

        const { trigger } = req;
        switch (trigger.type) {
            case "user.message":
            case "client.action":
            case "tool.result": {
                return {
                    ...downstream,
                    actions: [
                        {
                            type: "call.llm",
                            request: {
                                ...selection.request,
                                // it is expected middleware will populate this
                                messages: [],
                            },
                            llm_client: selection.llm_client,
                            retry: selection.retry ?? DEFAULT_RETRY,
                            stream: selection.stream ?? false,
                        },
                        ...downstream.actions,
                    ],
                };
            }
            case "llm.response": {
                if (!trigger.message.tool_calls || trigger.message.tool_calls.length === 0) {
                    return {
                        ...downstream,
                        actions: [{ type: "done", data: trigger.message.content }, ...downstream.actions],
                    };
                }
            }

            default:
                return downstream;
        }
    };
}

export interface SubAgentTrack {
    toolCallId: string;
    name: string;
}

/**
 * Sub-agent delegation middleware.
 * Contributes `{ subAgentTracker: Record<string, SubAgentTrack> }` to state.
 */
export function subAgents<S>(config: {
    delegates: Handler[];
    retry?: RetryPolicy;
}): StateContributor<{ subAgentTracker: Record<string, SubAgentTrack> }> &
    MiddlewareFn<unknown, { subAgentTracker: Record<string, SubAgentTrack> }> {
    const subAgents: Record<string, { agentId: string }> = {};
    for (const handler of config.delegates) {
        subAgents[handler.agentId] = { agentId: handler.agentId };
    }

    return stateSlice({ subAgentTracker: {} as Record<string, SubAgentTrack> }, async (req, next) => {
        const tracker = req.state.subAgentTracker;

        switch (req.trigger.type) {
            case "llm.response": {
                const downstream = await next(req);
                const actions: WorkerAction[] = [];
                for (const action of downstream.actions) {
                    if (action.type === "call.tool") {
                        const sub = subAgents[action.name];
                        if (sub) {
                            const childSessionId = crypto.randomUUID();
                            let message = action.arguments;
                            try {
                                const args = JSON.parse(action.arguments);
                                if (typeof args?.message === "string") {
                                    message = args.message;
                                }
                            } catch {
                                // no-op
                            }
                            tracker[childSessionId] = {
                                toolCallId: action.tool_call_id,
                                name: action.name,
                            };
                            actions.push(
                                {
                                    type: "spawn.sub_agent",
                                    session_id: childSessionId,
                                    agent_id: sub.agentId,
                                    retry: config.retry ?? DEFAULT_RETRY,
                                },
                                {
                                    type: "send.message",
                                    session_id: childSessionId,
                                    message: {
                                        role: "user",
                                        content: message,
                                    },
                                },
                            );
                            continue;
                        }
                    }
                    if (action.type === "call.llm") {
                        actions.push({
                            ...action,
                            request: {
                                ...action.request,
                                tools: mergeTools(action.request.tools, handlersToLlmTools(config.delegates)),
                            },
                        });
                        continue;
                    }
                    actions.push(action);
                }
                return { ...downstream, actions };
            }

            case "sub_agent.turn.complete": {
                const tracked = tracker[req.trigger.session_id];
                if (!tracked) {
                    return next(req);
                }
                delete tracker[req.trigger.session_id];

                const content =
                    typeof req.trigger.data === "string" ? req.trigger.data : JSON.stringify(req.trigger.data);
                const result: ToolResult = {
                    tool_call_id: tracked.toolCallId,
                    name: tracked.name,
                    content,
                    is_error: false,
                };

                return next({
                    ...req,
                    trigger: {
                        type: "tool.result",
                        result,
                    },
                });
            }

            case "sub_agent.error": {
                const tracked = tracker[req.trigger.session_id];
                if (!tracked) {
                    return next(req);
                }
                delete tracker[req.trigger.session_id];

                const result: ToolResult = {
                    tool_call_id: tracked.toolCallId,
                    name: tracked.name,
                    content: `Sub-agent ${req.trigger.agent_id} failed: ${req.trigger.error}`,
                    is_error: true,
                };

                return next({
                    ...req,
                    trigger: {
                        type: "tool.result",
                        result,
                    },
                });
            }

            default: {
                const downstream = await next(req);
                const actions = downstream.actions.map((action) => {
                    if (action.type !== "call.llm") return action;
                    return {
                        ...action,
                        request: {
                            ...action.request,
                            tools: mergeTools(action.request.tools, handlersToLlmTools(config.delegates)),
                        },
                    };
                });
                return { ...downstream, actions };
            }
        }
    });
}

function handlersToLlmTools(handlers: Handler[]): LlmTool[] {
    return handlers.map((handler) => ({
        function: {
            name: handler.agentId,
            description: `Delegate to ${handler.agentId}`,
            parameters: {
                type: "object",
                properties: {
                    message: {
                        type: "string",
                        description: "The message to send to the agent",
                    },
                },
                required: ["message"],
            },
        },
    }));
}

function mergeTools(existing: LlmTool[] | undefined, added: LlmTool[]): LlmTool[] {
    const byName = new Map<string, LlmTool>();
    for (const tool of existing ?? []) {
        byName.set(tool.function.name, tool);
    }
    for (const tool of added) {
        byName.set(tool.function.name, tool);
    }
    return Array.from(byName.values());
}
