import type {
    Agent,
    DecisionRequest,
    LlmGenerator,
    StopCondition,
    StopInfo,
    ToolDef,
    ToolExecutionContext,
} from "./core";
import { DEFAULT_RETRY, DEFERRED, serverGenerate, stamp } from "./core";
import type {
    DecisionTrigger,
    LlmRequest,
    LlmResponse,
    LlmTokenDeltaInput,
    LlmTool,
    Message,
    ReasoningConfig,
    RetryPolicy,
    StreamPart,
    ToolHandler,
    WorkerAction,
} from "./types";

// ── Model ────────────────────────────────────────────────────────────────────

/** A bound LLM backend: where the call runs (`handler`) and the request shape it
 *  carries. `server(...)` runs on the Substructure server; adapters (e.g.
 *  `anthropicGenerate`) run on your worker and supply `run`. */
export type Model = LlmGenerator;

/** The server's configured provider makes the call. `model` is its provider id,
 *  e.g. "anthropic/claude-sonnet-4-6". */
export function server(
    model: string,
    opts?: { temperature?: number; maxTokens?: number; reasoning?: ReasoningConfig; stream?: boolean },
): Model {
    return serverGenerate({ model, ...opts });
}

// ── Action builders (pure) ───────────────────────────────────────────────────

export function callLlm(opts: {
    model: Model;
    messages: Message[];
    tools?: LlmTool[];
    stream?: boolean;
    retry?: RetryPolicy;
}): WorkerAction {
    const request: LlmRequest = { ...opts.model.request, messages: opts.messages };
    if (opts.tools && opts.tools.length > 0) request.tools = opts.tools;
    return {
        type: "call.llm",
        request,
        handler: opts.model.handler,
        stream: opts.stream ?? opts.model.stream ?? false,
        retry: opts.retry ?? DEFAULT_RETRY,
    };
}

export function done(data?: unknown): WorkerAction {
    return { type: "done", data: data ?? null };
}

export function callTool(opts: {
    toolCallId: string;
    name: string;
    arguments: string;
    handler?: ToolHandler;
    retry?: RetryPolicy;
}): WorkerAction {
    return {
        type: "call.tool",
        tool_call_id: opts.toolCallId,
        name: opts.name,
        arguments: opts.arguments,
        handler: opts.handler ?? "worker",
        retry: opts.retry ?? DEFAULT_RETRY,
    };
}

export function toolResult(toolCallId: string, result: string, attempt: number): WorkerAction {
    return { type: "return.tool.result", tool_call_id: toolCallId, result, attempt };
}

export function toolError(toolCallId: string, error: string, attempt: number, retryable = false): WorkerAction {
    return { type: "return.tool.error", tool_call_id: toolCallId, error, retryable, attempt };
}

export function spawn(opts: {
    sessionId: string;
    agentId: string;
    toolCallId: string;
    retry?: RetryPolicy;
}): WorkerAction {
    return {
        type: "spawn.sub_agent",
        session_id: opts.sessionId,
        agent_id: opts.agentId,
        tool_call_id: opts.toolCallId,
        retry: opts.retry ?? DEFAULT_RETRY,
    };
}

export function sendMessage(sessionId: string, message: Message): WorkerAction {
    return { type: "send.message", session_id: sessionId, message };
}

// ── agent(config): the default tool/sub-agent loop ───────────────────────────

/** The default loop's configuration. */
export interface LoopConfig<S = unknown> {
    model: Model;
    instructions?: string | (() => string | Promise<string>);
    tools?: ToolDef[];
    /** Named agents the model can delegate to (reference them by value). */
    // biome-ignore lint/suspicious/noExplicitAny: sub-agents carry their own state type
    subAgents?: Agent<any>[];
    stopWhen?: StopCondition<S>;
    stream?: boolean;
    retry?: RetryPolicy;
}

/** An agent: a `name` plus a decision function. `decide` is `toolLoop(...)` for
 *  the default loop, or your own function. The `name` is the id the engine routes
 *  to (for `worker([...])` and sub-agent delegation). */
export interface AgentConfig<S = unknown> {
    name: string;
    decide: Agent<S>;
}

// biome-ignore lint/suspicious/noExplicitAny: sub-agents carry their own state type
function subAgentId(sub: Agent<any>): string {
    if (!sub.agentName) {
        throw new Error("a sub-agent must be a named agent — give it agent({ name, ... })");
    }
    return sub.agentName;
}

function toolSchema(def: ToolDef): LlmTool {
    return { function: { name: def.name, description: def.description, parameters: def.parameters } };
}

function subAgentSchema(agentId: string): LlmTool {
    return {
        function: {
            name: agentId,
            description: `Delegate to ${agentId}`,
            parameters: {
                type: "object",
                properties: { message: { type: "string", description: "The message to send to the agent" } },
                required: ["message"],
            },
        },
    };
}

function stopInfo<S>(d: DecisionRequest<S>, history: Message[]): StopInfo<S> {
    let steps = 0;
    let lastResponse: Message | undefined;
    for (let i = history.length - 1; i >= 0; i--) {
        const m = history[i];
        if (m.role === "user") break;
        if (m.role === "assistant") {
            steps++;
            lastResponse ??= m;
        }
    }
    return { steps, lastResponse, history, state: d.state };
}

/** The default tool/sub-agent loop, as a decision function. Plug it into an
 *  agent's `decide`, or call it from a custom `decide` to delegate. It echoes the
 *  decision's `state`, so a wrapping agent threads its own state through with
 *  `toolLoop(cfg)({ ...req, state })`. */
export function toolLoop<S = unknown>(config: LoopConfig<S>): Agent<S> {
    const toolList = config.tools ?? [];
    const toolMap: Record<string, ToolDef> = {};
    for (const t of toolList) toolMap[t.name] = t;

    const subIds = new Set((config.subAgents ?? []).map(subAgentId));
    const schemas: LlmTool[] = [...toolList.map(toolSchema), ...[...subIds].map(subAgentSchema)];

    return async (d) => {
        const { trigger, state } = d;
        const history = d.transcript ?? [];
        const ask = (messages: Message[]): WorkerAction =>
            callLlm({ model: config.model, messages, tools: schemas, stream: config.stream, retry: config.retry });

        const instructions =
            typeof config.instructions === "function" ? await config.instructions() : config.instructions;
        const withSystem = (base: Message[]): Message[] =>
            instructions && base[0]?.role !== "system"
                ? [{ role: "system", content: instructions, id: crypto.randomUUID() } as Message, ...base]
                : base;

        switch (trigger.type) {
            case "user.message": {
                const transcript = [...withSystem(history), trigger.message];
                return { transcript, actions: [ask(transcript)], state };
            }
            case "user.transcript": {
                const sys: Message | undefined =
                    history[0]?.role === "system"
                        ? history[0]
                        : instructions
                          ? { role: "system", content: instructions, id: crypto.randomUUID() }
                          : undefined;
                const body = trigger.messages.filter((m) => m.role !== "system");
                const transcript = sys ? [sys, ...body] : body;
                return { transcript, actions: [ask(transcript)], state };
            }
            case "client.action": {
                const base = withSystem(history);
                return { transcript: base, actions: [ask(base)], state };
            }
            case "llm.response": {
                const assistant = stamp(trigger.message);
                const transcript = [...history, assistant];
                const calls = assistant.tool_calls ?? [];
                if (calls.length === 0) return { transcript, actions: [done(assistant.content)], state };

                const actions: WorkerAction[] = [];
                for (const tc of calls) {
                    if (subIds.has(tc.function.name)) {
                        const childId = crypto.randomUUID();
                        let message = tc.function.arguments;
                        try {
                            const args = JSON.parse(tc.function.arguments);
                            if (typeof args?.message === "string") message = args.message;
                        } catch {
                            // leave raw arguments as the message
                        }
                        actions.push(
                            spawn({
                                sessionId: childId,
                                agentId: tc.function.name,
                                toolCallId: tc.id,
                                retry: config.retry,
                            }),
                            sendMessage(childId, { role: "user", content: message }),
                        );
                    } else if (toolMap[tc.function.name]) {
                        const def = toolMap[tc.function.name];
                        actions.push(
                            callTool({
                                toolCallId: tc.id,
                                name: tc.function.name,
                                arguments: tc.function.arguments,
                                handler: def.handler,
                                retry: def.retry,
                            }),
                        );
                    }
                }
                return { transcript, actions, state };
            }
            case "tool.execute": {
                const def = toolMap[trigger.name];
                if (!def)
                    return {
                        actions: [toolError(trigger.tool_call_id, `Unknown tool: ${trigger.name}`, trigger.attempt)],
                        state,
                    };
                const ctx: ToolExecutionContext = {
                    sessionId: d.session_id,
                    toolCallId: trigger.tool_call_id,
                    attempt: trigger.attempt,
                    request: d,
                    defer: () => DEFERRED,
                };
                try {
                    const out = await def.execute(trigger.arguments, ctx);
                    if (out === DEFERRED) return { state };
                    return {
                        actions: [
                            toolResult(trigger.tool_call_id, typeof out === "string" ? out : "", trigger.attempt),
                        ],
                        state,
                    };
                } catch (error) {
                    const message = error instanceof Error ? error.message : String(error);
                    return { actions: [toolError(trigger.tool_call_id, message, trigger.attempt)], state };
                }
            }
            case "tool.result": {
                const node: Message = {
                    id: crypto.randomUUID(),
                    role: "tool",
                    content: trigger.result,
                    tool_call_id: trigger.tool_call_id,
                    name: trigger.name,
                };
                const transcript = [...history, node];
                const roundComplete = (d.pending?.tool_calls ?? 0) + (d.pending?.sub_agents ?? 0) === 0;
                if (!roundComplete) return { transcript, state };
                if (config.stopWhen) {
                    const info = stopInfo(d, history);
                    if (await config.stopWhen(info)) {
                        return { transcript, actions: [done(info.lastResponse?.content ?? "")], state };
                    }
                }
                return { transcript, actions: [ask(transcript)], state };
            }
            case "llm.request": {
                const action = await runWorkerLlm(trigger, config.model.run, d.emitDelta);
                return { transcript: history, actions: [action], state };
            }
            default:
                return { state };
        }
    };
}

/** Name a decision function so it can be deployed (`worker([...])`) or used as a
 *  sub-agent. `decide` is `toolLoop(...)` for the default loop, or your own. */
export function agent<S = unknown>(config: AgentConfig<S>): Agent<S> {
    config.decide.agentName = config.name;
    return config.decide;
}

// ── Worker-run LLM (handler: "worker") ───────────────────────────────────────

function flattenDelta(part: StreamPart): LlmTokenDeltaInput | null {
    switch (part.type) {
        case "text-delta":
            return { text: part.delta };
        case "reasoning-delta":
            return { reasoning: part.delta };
        case "tool-input-start":
            return { tool_calls: [{ id: part.toolCallId, name: part.toolName }] };
        case "tool-input-delta":
            return { tool_calls: [{ id: part.toolCallId, arguments: part.inputTextDelta }] };
        case "finish":
            return part.finishReason ? { finish_reason: part.finishReason } : null;
        default:
            return null;
    }
}

async function runWorkerLlm(
    trigger: Extract<DecisionTrigger, { type: "llm.request" }>,
    run:
        | ((request: LlmRequest, ctx: { emitDelta?: (part: StreamPart) => Promise<void> }) => Promise<LlmResponse>)
        | undefined,
    emitDelta?: (delta: LlmTokenDeltaInput) => Promise<void>,
): Promise<WorkerAction> {
    if (!run) {
        return {
            type: "return.llm.error",
            call_id: trigger.call_id,
            error: 'received an "llm.request" trigger but this model has no worker-side `run` (use server(...) to let the engine call its provider)',
            retryable: false,
            attempt: trigger.attempt,
        };
    }
    const emitPart = emitDelta
        ? async (part: StreamPart) => {
              const flat = flattenDelta(part);
              if (flat) await emitDelta(flat);
          }
        : undefined;
    try {
        const response = await run(trigger.request, { emitDelta: emitPart });
        return { type: "return.llm.result", call_id: trigger.call_id, response, attempt: trigger.attempt };
    } catch (error) {
        return {
            type: "return.llm.error",
            call_id: trigger.call_id,
            error: error instanceof Error ? error.message : String(error),
            retryable: false,
            attempt: trigger.attempt,
        };
    }
}
