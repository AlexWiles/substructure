import type { Agent, Llm, NamedAgent, ToolDef } from "./core";
import type { LlmTool, Message, RetryPolicy, WorkerAction } from "./types";

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

/** An agent: a `name` plus a decision function. `decide` is `toolLoop(...)` for
 *  the default loop, or your own function. The `name` is the id the engine routes
 *  to (for `worker([...])` and sub-agent delegation). */
export interface AgentConfig<S = unknown> {
    name: string;
    decide: Agent<S>;
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
                properties: {
                    message: { type: "string", description: "The message to send to the agent" },
                },
                required: ["message"],
            },
        },
    };
}

/** The default tool/sub-agent loop, as a decision function. It echoes the decision's
 *  `state`, so a wrapping agent can thread its own via `toolLoop(cfg)({ ...req, state })`. */
export function toolLoop<S = unknown>(config: LoopConfig): Agent<S> {
    const toolList = config.tools ?? [];
    const toolMap = toolList.reduce<Record<string, ToolDef>>((map, t) => {
        map[t.name] = t;
        return map;
    }, {});

    const subIds = new Set((config.subAgents ?? []).map((sub) => sub.agentName));

    const toolSchemas: LlmTool[] = [...toolList.map(toolSchema), ...[...subIds].map(subAgentSchema)];

    // Split loop-fixed request params from routing (handler/stream/run).
    const { handler, run: _run, stream, ...llmParams } = config.llm;

    const ask = (messages: Message[]): WorkerAction => {
        return {
            type: "call.llm",
            id: crypto.randomUUID(),
            request: { ...llmParams, messages, tools: toolSchemas },
            handler: handler ?? "server",
            stream: stream ?? false,
            retry: config.retry,
        };
    };

    return async (d) => {
        const history = d.transcript ?? [];

        const instructions =
            typeof config.instructions === "function" ? await config.instructions() : config.instructions;
        // Guarantee the transcript starts with the system message, reusing the
        // stored one (stable id) when the branch already has it.
        const withSystem = (messages: Message[]): Message[] => {
            if (messages[0]?.role === "system") return messages;
            const existing = history[0]?.role === "system" ? history[0] : undefined;
            const system =
                existing ??
                (instructions ? { role: "system", content: instructions, id: crypto.randomUUID() } : undefined);
            return system ? [system, ...messages] : messages;
        };

        switch (d.trigger.type) {
            case "user.message": {
                const transcript = [...withSystem(history), d.trigger.message];
                return { transcript, actions: [ask(transcript)], state: d.state };
            }
            case "user.transcript": {
                const transcript = withSystem(d.trigger.messages.filter((m) => m.role !== "system"));
                return { transcript, actions: [ask(transcript)], state: d.state };
            }
            case "client.action": {
                const base = withSystem(history);
                return { transcript: base, actions: [ask(base)], state: d.state };
            }
            case "effect.execute": {
                switch (d.trigger.kind) {
                    case "llm_call": {
                        try {
                            // biome-ignore lint/style/noNonNullAssertion: the Llm union guarantees a worker LLM has `run`
                            const response = await config.llm.run!(d.trigger.request, { emitDelta: d.emitDelta });
                            const action: WorkerAction = {
                                type: "effect.result",
                                kind: "llm_call",
                                id: d.trigger.id,
                                response,
                                attempt: d.trigger.attempt,
                            };
                            return { transcript: history, actions: [action], state: d.state };
                        } catch (error) {
                            const message = error instanceof Error ? error.message : String(error);
                            const action: WorkerAction = {
                                type: "effect.error",
                                kind: "llm_call",
                                id: d.trigger.id,
                                error: message,
                                retryable: false,
                                attempt: d.trigger.attempt,
                            };
                            return { transcript: history, actions: [action], state: d.state };
                        }
                    }
                    case "tool_call": {
                        const def = toolMap[d.trigger.name];
                        if (!def)
                            return {
                                actions: [
                                    {
                                        type: "effect.error",
                                        kind: "tool_call",
                                        id: d.trigger.id,
                                        error: `Unknown tool: ${d.trigger.name}`,
                                        retryable: false,
                                        attempt: d.trigger.attempt,
                                    },
                                ],
                                state: d.state,
                            };
                        try {
                            const out = await def.execute(d.trigger.arguments, d);
                            // A deferred tool's execute only starts the work; the result arrives via settleEffect.
                            if (def.deferred) return { state: d.state };
                            return {
                                actions: [
                                    {
                                        type: "effect.result",
                                        kind: "tool_call",
                                        id: d.trigger.id,
                                        result: out ?? "",
                                        attempt: d.trigger.attempt,
                                    },
                                ],
                                state: d.state,
                            };
                        } catch (error) {
                            const message = error instanceof Error ? error.message : String(error);
                            return {
                                actions: [
                                    {
                                        type: "effect.error",
                                        kind: "tool_call",
                                        id: d.trigger.id,
                                        error: message,
                                        retryable: false,
                                        attempt: d.trigger.attempt,
                                    },
                                ],
                                state: d.state,
                            };
                        }
                    }
                    default:
                        return { state: d.state };
                }
            }
            case "effect.settled": {
                switch (d.trigger.kind) {
                    case "llm_call": {
                        if (!d.trigger.ok || d.trigger.message === undefined) return { state: d.state };
                        const assistant = d.trigger.message;
                        const transcript = [...history, assistant];
                        const calls = assistant.tool_calls ?? [];
                        if (calls.length === 0)
                            return {
                                transcript,
                                actions: [{ type: "done", data: assistant.content ?? null }],
                                state: d.state,
                            };

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
                                    {
                                        type: "spawn.sub_agent",
                                        session_id: childId,
                                        agent_id: tc.function.name,
                                        tool_call_id: tc.id,
                                        retry: config.retry,
                                    },
                                    {
                                        type: "send.message",
                                        session_id: childId,
                                        message: { role: "user", content: message },
                                    },
                                );
                            } else if (toolMap[tc.function.name]) {
                                const def = toolMap[tc.function.name];
                                actions.push({
                                    type: "call.tool",
                                    id: tc.id,
                                    name: tc.function.name,
                                    arguments: tc.function.arguments,
                                    handler: def.handler ?? "worker",
                                    retry: def.retry,
                                });
                            }
                        }
                        return { transcript, actions, state: d.state };
                    }
                    case "sub_agent":
                    case "tool_call": {
                        const node: Message =
                            d.trigger.kind === "sub_agent"
                                ? {
                                      id: crypto.randomUUID(),
                                      role: "tool",
                                      content: d.trigger.result,
                                      tool_call_id: d.trigger.tool_call_id,
                                      name: d.trigger.agent_id,
                                  }
                                : {
                                      id: crypto.randomUUID(),
                                      role: "tool",
                                      content: d.trigger.result,
                                      tool_call_id: d.trigger.id,
                                      name: d.trigger.name,
                                  };
                        const transcript = [...history, node];
                        if (d.pending_effects > 0) return { transcript, state: d.state };
                        return { transcript, actions: [ask(transcript)], state: d.state };
                    }
                    default:
                        return { state: d.state };
                }
            }
            default:
                return { state: d.state };
        }
    };
}

/** Name a decision function so it can be deployed (`worker([...])`) or used as a
 *  sub-agent. `decide` is `toolLoop(...)` for the default loop, or your own. */
export function agent<S = unknown>(config: AgentConfig<S>): NamedAgent<S> {
    return Object.assign(config.decide, { agentName: config.name });
}
