import type {
    Message,
    LlmTool,
    LlmRequest,
    RetryPolicy,
    WorkerAction,
    ToolResult,
} from "./types";
import type {
    Handler,
    HandlerContext,
    HandlerResult,
    MiddlewareFn,
    Next,
    StateContributor,
} from "./worker";

/**
 * Base64 JSON state serialization middleware.
 * Decodes `request.worker_state` into `ctx.state` (falling back to `{}`),
 * runs the chain, then encodes `ctx.state` back.
 * Individual middleware contribute their own state slices via `StateContributor`.
 */
export function withState(): MiddlewareFn<unknown> {
    return async (ctx: HandlerContext<unknown>, next: Next<unknown>) => {
        const raw = ctx.request.worker_state;
        ctx.state =
            raw && raw !== ""
                ? JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(raw), c => c.charCodeAt(0))))
                : {};
        const result = await next(ctx);
        ctx.request.worker_state = btoa(String.fromCharCode(...new TextEncoder().encode(JSON.stringify(ctx.state))));
        return result;
    };
}

/**
 * Create a StateContributor middleware that owns a slice of state.
 * The `init` object provides default values and the TypeScript type brand.
 * The middleware function receives a fully typed `ctx.state`.
 */
export function withStateSlice<A extends object>(
    init: A,
    fn: (ctx: HandlerContext<A>, next: Next<A>) => Promise<HandlerResult> | HandlerResult,
): StateContributor<A> {
    return Object.assign(
        (ctx: HandlerContext<unknown>, next: Next<unknown>) => {
            const state = ctx.state as Record<string, unknown>;
            for (const key of Object.keys(init)) {
                state[key] ??= structuredClone(
                    (init as Record<string, unknown>)[key],
                );
            }
            return fn(ctx as HandlerContext<A>, next as Next<A>);
        },
        { _contributes: init },
    ) as StateContributor<A>;
}

export type ToolFn = (args: string) => Promise<unknown>;

export interface ToolDef {
    description: string;
    parameters: unknown;
    execute: ToolFn;
    retry?: RetryPolicy;
}

/** Define a tool from plain data. */
export function tool(config: {
    description: string;
    parameters: unknown;
    execute: (args: string) => unknown | Promise<unknown>;
    retry?: RetryPolicy;
}): ToolDef {
    return {
        description: config.description,
        parameters: config.parameters,
        execute: async (args: string) => config.execute(args),
        retry: config.retry,
    };
}

export function withLogging(label?: string): MiddlewareFn<unknown, unknown> {
    const prefix = label ? `[${label}]` : "[handler]";
    return async (ctx, next) => {
        const t = ctx.trigger;
        const tag = t.type === "tool_execute" ? `${t.type}:${t.name}` : t.type;
        console.log(`${prefix} ${tag}`);
        const start = performance.now();
        const result = await next(ctx);
        const ms = (performance.now() - start).toFixed(1);
        console.log(
            `${prefix} ${tag} -> ${result.actions.map((a) => a.type).join(", ")} (${ms}ms)`,
        );
        return result;
    };
}

export type MessageSelector<S> = (state: S, ctx: HandlerContext<S>) => Message[];

/**
 * Conversation history middleware. Contributes `{ messages: Message[] }` to state.
 * Records incoming messages and augments `call_llm` actions with the full history.
 */
export function withConversation(): StateContributor<{ messages: Message[] }> {
    return withStateSlice({ messages: [] as Message[] }, async (ctx, next) => {
        const history = ctx.state.messages;
        const { trigger } = ctx;

        switch (trigger.type) {
            case "user_message":
            case "llm_response":
                history.push(trigger.message);
                break;
            case "client_action":
                break;
            case "tool_result":
                history.push({
                    role: "tool",
                    content: trigger.result.content,
                    tool_call_id: trigger.result.tool_call_id,
                    name: trigger.result.name,
                });
                break;
        }

        const result = await next(ctx);

        const actions = result.actions.map((action) => {
            if (action.type !== "call_llm") return action;
            return {
                ...action,
                request: {
                    ...action.request,
                    messages: history,
                },
            };
        });

        return { actions };
    });
}

export type SystemMessageSelector<S> = (state: S, ctx: HandlerContext<S>) => string;

export function withSystemMessage<S>(
    selector: SystemMessageSelector<S>,
): MiddlewareFn<S> {
    return async (ctx, next) => {
        const systemMessage: Message = { role: "system", content: selector(ctx.state, ctx) };

        const result = await next(ctx);
        const actions = result.actions.map((action) => {
            if (action.type !== "call_llm") return action;
            return {
                ...action,
                request: {
                    ...action.request,
                    messages: [systemMessage, ...action.request.messages],
                },
            };
        });

        return { actions };
    };
}

export type ToolSelector<S> = (state: S, ctx: HandlerContext<S>) => Record<string, ToolDef>;

/**
 * Tool execution and pending-tracking middleware.
 * Contributes `{ pendingToolCalls: string[] }` to state.
 *
 * On `llm_response`: records tool_call_ids as pending.
 * On `tool_result`: removes the ID. Suppresses `call_llm` until all results are in.
 * On `tool_execute`: executes the tool and prepends the result to downstream actions.
 */
export function withTools<S>(
    selector: ToolSelector<S>,
): StateContributor<{ pendingToolCalls: string[] }> {
    return withStateSlice({ pendingToolCalls: [] as string[] },
        async (ctx, next) => {
            const tools = selector(ctx.state as S, ctx as unknown as HandlerContext<S>);

            const downstream = await next(ctx);

            // Handle tool execution
            if (ctx.trigger.type === "tool_execute") {
                const tool = tools[ctx.trigger.name];
                if (!tool) {
                    return {
                        actions: [
                            {
                                type: "return_tool_error" as const,
                                tool_call_id: ctx.trigger.tool_call_id,
                                error: `Unknown tool: ${ctx.trigger.name}`,
                                retryable: false,
                                attempt: ctx.trigger.attempt,
                            },
                            ...downstream.actions,
                        ],
                    };
                }

                try {
                    const output = await tool.execute(ctx.trigger.arguments);
                    return {
                        actions: [
                            {
                                type: "return_tool_result" as const,
                                tool_call_id: ctx.trigger.tool_call_id,
                                result:
                                    typeof output === "string"
                                        ? output
                                        : JSON.stringify(output),
                                attempt: ctx.trigger.attempt,
                            },
                            ...downstream.actions,
                        ],
                    };
                } catch (error: unknown) {
                    return {
                        actions: [
                            {
                                type: "return_tool_error" as const,
                                tool_call_id: ctx.trigger.tool_call_id,
                                error: error instanceof Error ? error.message : String(error),
                                retryable: false,
                                attempt: ctx.trigger.attempt,
                            },
                            ...downstream.actions,
                        ],
                    };
                }
            }

            // Track pending tool calls from LLM response and emit call_tool actions
            if (ctx.trigger.type === "llm_response") {
                const toolCalls = ctx.trigger.message.tool_calls;
                if (toolCalls && toolCalls.length > 0) {
                    ctx.state.pendingToolCalls = toolCalls.map((tc) => tc.id);

                    const callToolActions: WorkerAction[] = toolCalls.map((tc) => {
                        const def = tools[tc.function.name];
                        return {
                            type: "call_tool" as const,
                            tool_call_id: tc.id,
                            name: tc.function.name,
                            arguments: tc.function.arguments,
                            handler: "worker" as const,
                            retry: def?.retry ?? {
                                timeout_secs: 120,
                                max_retries: 3,
                                backoff_base_secs: 1,
                                backoff_max_secs: 10,
                            },
                        };
                    });

                    return {
                        actions: [...callToolActions, ...downstream.actions],
                    };
                }
            }

            // Augment actions with tool definitions and retry policies
            const actions = downstream.actions.map((action) => {
                if (action.type === "call_llm") {
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
                if (action.type === "call_tool") {
                    const def = tools[action.name];
                    if (def?.retry) {
                        return { ...action, retry: def.retry };
                    }
                }
                return action;
            });

            // Suppress call_llm until all tool results are in
            if (ctx.trigger.type === "tool_result") {
                const resultId = ctx.trigger.result.tool_call_id;
                ctx.state.pendingToolCalls =
                    ctx.state.pendingToolCalls.filter(
                        (id) => id !== resultId,
                    );
                if (ctx.state.pendingToolCalls.length > 0) {
                    return {
                        actions: actions.filter(
                            (a) => a.type !== "call_llm",
                        ),
                    };
                }
            }

            return { actions };
        },
    );
}

export interface CallLlmSelection {
    request: Omit<LlmRequest, "messages"> & { messages?: Message[] };
    llm_client: string;
    retry: RetryPolicy;
    stream?: boolean;
    toolRetries?: Record<string, RetryPolicy>;
}

export function withCallLLM<S>(
    selector: (state: S, ctx: HandlerContext<S>) => CallLlmSelection,
): MiddlewareFn<S> {
    return async (ctx, next) => {
        const selection = selector(ctx.state, ctx);
        const downstream = await next(ctx);

        const { trigger } = ctx;
        switch (trigger.type) {
            case "user_message":
            case "client_action":
            case "tool_result": {
                return {
                    actions: [
                        {
                            type: "call_llm",
                            request: {
                                ...selection.request,
                                // it is expected middleware will populate this
                                messages: [],
                            },
                            llm_client: selection.llm_client,
                            retry: selection.retry,
                            stream: selection.stream ?? false,
                        },
                        ...downstream.actions,
                    ],
                };
            }
            case "llm_response": {
                if (
                    !trigger.message.tool_calls ||
                    trigger.message.tool_calls.length === 0
                ) {
                    return {
                        actions: [
                            { type: "done", data: trigger.message.content },
                            ...downstream.actions,
                        ],
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
export function withSubAgents<S>(config: {
    delegates: Handler[];
    retry: RetryPolicy;
}): StateContributor<{ subAgentTracker: Record<string, SubAgentTrack> }> {
    const subAgents: Record<string, { agentId: string }> = {};
    for (const handler of config.delegates) {
        subAgents[handler.agentId] = { agentId: handler.agentId };
    }

    return withStateSlice(
        { subAgentTracker: {} as Record<string, SubAgentTrack> },
        async (ctx, next) => {
            const tracker = ctx.state.subAgentTracker;

            switch (ctx.trigger.type) {
                case "llm_response": {
                    const downstream = await next(ctx);
                    const actions: WorkerAction[] = [];
                    for (const action of downstream.actions) {
                        if (action.type === "call_tool") {
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
                                        type: "spawn_sub_agent",
                                        session_id: childSessionId,
                                        agent_id: sub.agentId,
                                        retry: config.retry,
                                    },
                                    {
                                        type: "send_message",
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
                        if (action.type === "call_llm") {
                            actions.push({
                                ...action,
                                request: {
                                    ...action.request,
                                    tools: mergeTools(
                                        action.request.tools,
                                        handlersToLlmTools(config.delegates),
                                    ),
                                },
                            });
                            continue;
                        }
                        actions.push(action);
                    }
                    return { actions };
                }

                case "sub_agent_turn_complete": {
                    const tracked = tracker[ctx.trigger.session_id];
                    if (!tracked) {
                        return next(ctx);
                    }
                    delete tracker[ctx.trigger.session_id];

                    const content =
                        typeof ctx.trigger.data === "string"
                            ? ctx.trigger.data
                            : JSON.stringify(ctx.trigger.data);
                    const result: ToolResult = {
                        tool_call_id: tracked.toolCallId,
                        name: tracked.name,
                        content,
                        is_error: false,
                    };

                    return next({
                        ...ctx,
                        trigger: {
                            type: "tool_result",
                            result,
                        },
                    });
                }

                case "sub_agent_error": {
                    const tracked = tracker[ctx.trigger.session_id];
                    if (!tracked) {
                        return next(ctx);
                    }
                    delete tracker[ctx.trigger.session_id];

                    const result: ToolResult = {
                        tool_call_id: tracked.toolCallId,
                        name: tracked.name,
                        content: `Sub-agent ${ctx.trigger.agent_id} failed: ${ctx.trigger.error}`,
                        is_error: true,
                    };

                    return next({
                        ...ctx,
                        trigger: {
                            type: "tool_result",
                            result,
                        },
                    });
                }

                default: {
                    const downstream = await next(ctx);
                    const actions = downstream.actions.map((action) => {
                        if (action.type !== "call_llm") return action;
                        return {
                            ...action,
                            request: {
                                ...action.request,
                                tools: mergeTools(
                                    action.request.tools,
                                    handlersToLlmTools(config.delegates),
                                ),
                            },
                        };
                    });
                    return { actions };
                }
            }
        },
    );
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

function mergeTools(
    existing: LlmTool[] | undefined,
    added: LlmTool[],
): LlmTool[] {
    const byName = new Map<string, LlmTool>();
    for (const tool of existing ?? []) {
        byName.set(tool.function.name, tool);
    }
    for (const tool of added) {
        byName.set(tool.function.name, tool);
    }
    return Array.from(byName.values());
}
