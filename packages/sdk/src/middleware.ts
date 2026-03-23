import type {
    Message,
    LlmTool,
    LlmRequest,
    RetryPolicy,
    WorkerAction,
    ToolResult,
} from "./types";
import type * as z from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import type {
    Handler,
    HandlerContext,
    MiddlewareFn,
    Next,
    StateContributor,
} from "./worker";

export type ToolFn = (args: unknown) => Promise<unknown> | unknown;

export interface ToolDef {
    description: string;
    parameters: unknown;
    execute: ToolFn;
    retry?: RetryPolicy;
}

export interface SubAgentTrack {
    toolCallId: string;
    name: string;
}

export interface MessagesAdapter<S> {
    getMessages: (state: S) => Message[];
    setMessages: (state: S, messages: Message[]) => void;
}

export interface SubAgentTrackerAdapter<S> {
    getSubAgentTracker: (state: S) => Record<string, SubAgentTrack>;
    setSubAgentTracker: (state: S, tracker: Record<string, SubAgentTrack>) => void;
}

export interface CallLlmSelection {
    request: Omit<LlmRequest, "messages"> & { messages?: Message[] };
    llm_client: string;
    retry: RetryPolicy;
    stream?: boolean;
    toolRetries?: Record<string, RetryPolicy>;
}

export type MessageSelector<S> =
    | Message[]
    | ((state: S, ctx: HandlerContext<S>) => Message[]);

export type SystemMessageSelector<S> =
    | string
    | Message
    | ((state: S, ctx: HandlerContext<S>) => string | Message);

export type ToolSelector<S> =
    | Record<string, ToolDef>
    | ((state: S, ctx: HandlerContext<S>) => Record<string, ToolDef>);

/**
 * JSON + base64 state serialization middleware.
 * Decodes `request.worker_state` into `ctx.state` (falling back to `init`),
 * runs the chain, then encodes `ctx.state` back.
 */
export function withState<S extends object>(init: S): StateContributor<S> {
    const mw: StateContributor<S> = Object.assign(
        async (ctx: HandlerContext<unknown>, next: Next<unknown>) => {
            const raw = ctx.request.worker_state;

            ctx.state =
                raw && raw !== ""
                    ? JSON.parse(Buffer.from(raw, "base64").toString("utf-8"))
                    : { ...init };

            const result = await next(ctx);
            ctx.request.worker_state = Buffer.from(JSON.stringify(ctx.state), "utf-8",).toString("base64");
            return result;
        },
        { _contributes: init },
    );
    return mw;
}

/** Define a tool with a zod schema for type-safe parameters. */
export function tool<T extends z.ZodTypeAny>(config: {
    description: string;
    parameters: T;
    execute: (args: z.infer<T>) => unknown | Promise<unknown>;
    retry?: RetryPolicy;
}): ToolDef {
    return {
        description: config.description,
        parameters: zodToJsonSchema(config.parameters as never, { target: "openAi" }),
        execute: (args: unknown) => config.execute(config.parameters.parse(args)),
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

export function withMessageHistory<S>(adapter: MessagesAdapter<S>): MiddlewareFn<S> {
    return async (ctx, next) => {
        const messages = adapter.getMessages(ctx.state);
        const { trigger } = ctx;
        switch (trigger.type) {
            case "user_message":
            case "llm_response":
                messages.push(trigger.message);
                break;
            case "client_action":
                break;
            case "tool_result":
                messages.push({
                    role: "tool",
                    content: trigger.result.content,
                    tool_call_id: trigger.result.tool_call_id,
                    name: trigger.result.name,
                });
                break;
        }
        adapter.setMessages(ctx.state, messages);
        return next(ctx);
    };
}

export function withMessages<S>(selector: MessageSelector<S>): MiddlewareFn<S> {
    return async (ctx, next) => {
        const messages = resolveMessages(selector, ctx);
        const result = await next(ctx);
        const actions = result.actions.map((action) => {
            if (action.type !== "call_llm") return action;
            return {
                ...action,
                request: {
                    ...action.request,
                    messages,
                },
            };
        });
        return { actions };
    };
}

export function withConversation<S>(adapter: MessagesAdapter<S>): MiddlewareFn<S> {
    return async (ctx, next) => {
        const history = adapter.getMessages(ctx.state);
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

        adapter.setMessages(ctx.state, history);

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
    };
}

export function withSystemMessage<S>(selector: SystemMessageSelector<S>): MiddlewareFn<S> {
    return async (ctx, next) => {
        const selected =
            typeof selector === "function"
                ? selector(ctx.state, ctx)
                : selector;
        const systemMessage = normalizeSystemMessage(selected);

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

export function withTools<S>(selector: ToolSelector<S>): MiddlewareFn<S> {
    return async (ctx, next) => {
        const tools = resolveTools(selector, ctx);

        if (ctx.trigger.type === "tool_execute") {
            const tool = tools[ctx.trigger.name];
            if (!tool) {
                return {
                    actions: [
                        {
                            type: "return_tool_error",
                            tool_call_id: ctx.trigger.tool_call_id,
                            error: `Unknown tool: ${ctx.trigger.name}`,
                            retryable: false,
                            attempt: ctx.trigger.attempt,
                        },
                    ],
                };
            }

            try {
                const args = JSON.parse(ctx.trigger.arguments);
                const output = await tool.execute(args);
                return {
                    actions: [
                        {
                            type: "return_tool_result",
                            tool_call_id: ctx.trigger.tool_call_id,
                            result: typeof output === "string" ? output : JSON.stringify(output),
                            attempt: ctx.trigger.attempt,
                        },
                    ],
                };
            } catch (error: unknown) {
                return {
                    actions: [
                        {
                            type: "return_tool_error",
                            tool_call_id: ctx.trigger.tool_call_id,
                            error: errorMessage(error),
                            retryable: false,
                            attempt: ctx.trigger.attempt,
                        },
                    ],
                };
            }
        }

        const result = await next(ctx);
        const actions = result.actions.map((action) => {
            if (action.type === "call_llm") {
                return {
                    ...action,
                    request: {
                        ...action.request,
                        tools: mergeTools(action.request.tools, toolsToLlmTools(tools)),
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
        return { actions };
    };
}

export function withCallLLM<S>(
    selector: (state: S, ctx: HandlerContext<S>) => CallLlmSelection,
): MiddlewareFn<S> {
    return async (ctx, next) => {
        const selection = selector(ctx.state, ctx);

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
                                messages: []
                            },
                            llm_client: selection.llm_client,
                            retry: selection.retry,
                            stream: selection.stream ?? false,
                        },
                    ],
                }
            }
            case "llm_response": {
                if (!trigger.message.tool_calls || trigger.message.tool_calls.length === 0) {
                    return { actions: [{ type: "done", data: trigger.message.content }] }
                }
            }

            default:
                return next(ctx);
        }
    };
}

export function withSubAgents<S>(config: {
    delegates: Handler[];
    tracker: SubAgentTrackerAdapter<S>;
    retry: RetryPolicy;
}): MiddlewareFn<S> {
    const subAgents = handlersToSubAgents(config.delegates);

    return async (ctx, next) => {
        const state = ctx.state;
        const tracker = config.tracker.getSubAgentTracker(state);

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
                                    message: { role: "user", content: message },
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
                                tools: mergeTools(action.request.tools, handlersToLlmTools(config.delegates)),
                            },
                        });
                        continue;
                    }
                    actions.push(action);
                }
                config.tracker.setSubAgentTracker(state, tracker);
                return { actions };
            }

            case "sub_agent_turn_complete": {
                const tracked = tracker[ctx.trigger.session_id];
                if (!tracked) {
                    return next(ctx);
                }
                delete tracker[ctx.trigger.session_id];
                config.tracker.setSubAgentTracker(state, tracker);

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
                config.tracker.setSubAgentTracker(state, tracker);

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
                            tools: mergeTools(action.request.tools, handlersToLlmTools(config.delegates)),
                        },
                    };
                });
                return { actions };
            }
        }
    };
}

function resolveTools<S>(
    selector: ToolSelector<S>,
    ctx: HandlerContext<S>,
): Record<string, ToolDef> {
    return typeof selector === "function" ? selector(ctx.state, ctx) : selector;
}

function resolveMessages<S>(
    selector: MessageSelector<S>,
    ctx: HandlerContext<S>,
): Message[] {
    return typeof selector === "function" ? selector(ctx.state, ctx) : selector;
}

function toolsToLlmTools(tools: Record<string, ToolDef>): LlmTool[] {
    return Object.entries(tools).map(([name, def]) => ({
        function: { name, description: def.description, parameters: def.parameters },
    }));
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

function handlersToSubAgents(handlers: Handler[]): Record<string, { agentId: string }> {
    const out: Record<string, { agentId: string }> = {};
    for (const handler of handlers) {
        out[handler.agentId] = { agentId: handler.agentId };
    }
    return out;
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

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function normalizeSystemMessage(input: string | Message): Message {
    if (typeof input === "string") {
        return { role: "system", content: input };
    }
    return input;
}
