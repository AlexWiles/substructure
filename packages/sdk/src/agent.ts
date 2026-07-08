import type { Agent, Llm, NamedAgent, ToolDef } from "./core";
import type { LlmTool, Message, MessageInput, RetryPolicy, WorkerAction } from "./types";

// ── agent(config): the default tool/sub-agent loop ───────────────────────────

/** The default loop's configuration. */
export interface LoopConfig {
    llm: Llm;
    instructions?: string | (() => string | Promise<string>);
    tools?: ToolDef[];
    /** Named agents the model can delegate to (reference them by value). */
    // biome-ignore lint/suspicious/noExplicitAny: sub-agents carry their own state type
    subAgents?: NamedAgent<any>[];
    retry?: RetryPolicy;
}

/** An agent config: a `name` and a `decide` function (`toolLoop(...)` or your own). */
export interface AgentConfig<S = unknown> {
    name: string;
    decide: Agent<S>;
}

function toolSchema(def: ToolDef): LlmTool {
    return { name: def.name, description: def.description, input: def.input, output: def.output };
}

function subAgentSchema(agentId: string): LlmTool {
    return {
        name: agentId,
        description: `Delegate to ${agentId}`,
        input: {
            type: "object",
            properties: {
                message: { type: "string", description: "The message to send to the agent" },
            },
            required: ["message"],
        },
    };
}

/** The default tool/sub-agent loop, as a decision function. Expresses no `state`;
 *  wrap it (`toolLoop(cfg)({ ...req, state })`) to thread your own. */
export function toolLoop<S = unknown>(config: LoopConfig): Agent<S> {
    const toolList = config.tools ?? [];
    const toolMap = toolList.reduce<Record<string, ToolDef>>((map, t) => {
        map[t.name] = t;
        return map;
    }, {});

    const subIds = new Set((config.subAgents ?? []).map((sub) => sub.agentName));

    const toolSchemas: LlmTool[] = [...toolList.map(toolSchema), ...[...subIds].map(subAgentSchema)];

    const { handler, run: _run, stream, ...llmParams } = config.llm;

    const ask = (messages: MessageInput[]): WorkerAction => {
        return {
            type: "llm.call",
            id: crypto.randomUUID(),
            request: { ...llmParams, messages, tools: toolSchemas },
            handler: handler ?? "server",
            stream: stream ?? false,
            retry: config.retry,
        };
    };

    return async (d) => {
        const history = d.messages ?? [];

        const instructions =
            typeof config.instructions === "function" ? await config.instructions() : config.instructions;
        // Lead with the system message, reusing the stored one (stable id) when present.
        const withSystem = (messages: MessageInput[]): MessageInput[] => {
            if (messages[0]?.role === "system") return messages;
            const existing = history[0]?.role === "system" ? history[0] : undefined;
            const system =
                existing ??
                (instructions ? { role: "system", content: instructions, id: crypto.randomUUID() } : undefined);
            return system ? [system, ...messages] : messages;
        };

        switch (d.trigger.type) {
            case "client.messages": {
                const messages = withSystem(d.trigger.messages.filter((m) => m.role !== "system"));
                // Client news arriving mid-step (calls still in flight): record
                // the proposal now, prompt once the open step settles.
                if (d.pending_calls > 0) return { messages };
                return { messages, actions: [ask(messages)] };
            }
            case "client.action": {
                const base = withSystem(history);
                return { messages: base, actions: [ask(base)] };
            }
            case "llm.execute": {
                try {
                    // biome-ignore lint/style/noNonNullAssertion: the Llm union guarantees a worker LLM has `run`
                    const response = await config.llm.run!(d.trigger.request, { emitDelta: d.emitDelta });
                    const action: WorkerAction = {
                        type: "llm.result",
                        id: d.trigger.id,
                        response,
                        attempt: d.trigger.attempt,
                    };
                    return { messages: history, actions: [action] };
                } catch (error) {
                    const message = error instanceof Error ? error.message : String(error);
                    const action: WorkerAction = {
                        type: "llm.error",
                        id: d.trigger.id,
                        error: message,
                        retryable: false,
                        attempt: d.trigger.attempt,
                    };
                    return { messages: history, actions: [action] };
                }
            }
            case "tool.execute": {
                const def = toolMap[d.trigger.name];
                if (!def)
                    return {
                        actions: [
                            {
                                type: "tool.error",
                                id: d.trigger.id,
                                error: `Unknown tool: ${d.trigger.name}`,
                                retryable: false,
                                attempt: d.trigger.attempt,
                            },
                        ],
                    };
                try {
                    const out = await def.execute(d.trigger.arguments, d);
                    // Deferred: execute only starts the work; result arrives via settleEffect.
                    if (def.deferred) return {};
                    return {
                        actions: [
                            {
                                type: "tool.result",
                                id: d.trigger.id,
                                result: out ?? "",
                                attempt: d.trigger.attempt,
                            },
                        ],
                    };
                } catch (error) {
                    const message = error instanceof Error ? error.message : String(error);
                    return {
                        actions: [
                            {
                                type: "tool.error",
                                id: d.trigger.id,
                                error: message,
                                retryable: false,
                                attempt: d.trigger.attempt,
                            },
                        ],
                    };
                }
            }
            case "llm.finished": {
                if (!d.trigger.ok || d.trigger.message === undefined) return {};
                const assistant = d.trigger.message;
                const messages = [...history, assistant];
                const calls = assistant.tool_calls ?? [];
                if (calls.length === 0)
                    return {
                        messages,
                        actions: [{ type: "done", data: assistant.content ?? null }],
                    };

                const actions: WorkerAction[] = [];
                for (const tc of calls) {
                    if (subIds.has(tc.function.name)) {
                        const childId = crypto.randomUUID();
                        let message = tc.function.arguments;
                        try {
                            const args = JSON.parse(tc.function.arguments);
                            if (typeof args?.message === "string") message = args.message;
                        } catch {}
                        actions.push(
                            {
                                type: "sub_agent.spawn",
                                session_id: childId,
                                agent_id: tc.function.name,
                                tool_call_id: tc.id,
                                retry: config.retry,
                            },
                            {
                                type: "message.send",
                                session_id: childId,
                                message: { role: "user", content: message },
                            },
                        );
                    } else if (toolMap[tc.function.name]) {
                        const def = toolMap[tc.function.name];
                        actions.push({
                            type: "tool.call",
                            id: tc.id,
                            name: tc.function.name,
                            arguments: tc.function.arguments,
                            handler: def.handler ?? "worker",
                            retry: def.retry,
                        });
                    }
                }
                return { messages, actions };
            }
            case "sub_agent.finished": {
                const node: Message = {
                    id: crypto.randomUUID(),
                    role: "tool",
                    content: (d.trigger.ok ? d.trigger.result : d.trigger.error) ?? "",
                    tool_call_id: d.trigger.id,
                    name: d.trigger.agent_id,
                };
                const messages = [...history, node];
                if (d.pending_calls > 0) return { messages };
                return { messages, actions: [ask(messages)] };
            }

            case "tool.finished": {
                const node: Message = {
                    id: crypto.randomUUID(),
                    role: "tool",
                    content: (d.trigger.ok ? d.trigger.result : d.trigger.error) ?? "",
                    tool_call_id: d.trigger.id,
                    name: d.trigger.name,
                };
                const messages = [...history, node];
                if (d.pending_calls > 0) return { messages };
                return { messages, actions: [ask(messages)] };
            }
            default:
                return {};
        }
    };
}

/** Name a decision function so it can be deployed (`worker([...])`) or delegated to as a sub-agent. */
export function agent<S = unknown>(config: AgentConfig<S>): NamedAgent<S> {
    return Object.assign(config.decide, { agentName: config.name });
}
