import type { Agent, Llm, NamedAgent, ToolDef, ToolExecutionContext } from "./core";
import { DEFERRED, stamp } from "./core";
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

/** The default tool/sub-agent loop, as a decision function. Plug it into an
 *  agent's `decide`, or call it from a custom `decide` to delegate. It echoes the
 *  decision's `state`, so a wrapping agent threads its own state through with
 *  `toolLoop(cfg)({ ...req, state })`. */
export function toolLoop<S = unknown>(config: LoopConfig): Agent<S> {
    const toolList = config.tools ?? [];
    const toolMap = toolList.reduce<Record<string, ToolDef>>((map, t) => {
        map[t.name] = t;
        return map;
    }, {});

    const subIds = new Set((config.subAgents ?? []).map((sub) => sub.agentName));

    const toolSchemas: LlmTool[] = [...toolList.map(toolSchema), ...[...subIds].map(subAgentSchema)];

    // A `call.llm` for the current messages. Request params (model, temperature, …)
    // are fixed for the loop; split routing (handler/stream/run) off them once.
    const { handler, run: _run, stream, ...llmParams } = config.llm;

    const ask = (messages: Message[]): WorkerAction => {
        return {
            type: "call.llm",
            request: { ...llmParams, messages, tools: toolSchemas },
            handler: handler ?? "server",
            stream: stream ?? false,
            retry: config.retry,
        };
    };

    return async (d) => {
        const { trigger, state } = d;
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

        switch (trigger.type) {
            case "user.message": {
                const transcript = [...withSystem(history), trigger.message];
                return { transcript, actions: [ask(transcript)], state };
            }
            case "user.transcript": {
                const transcript = withSystem(trigger.messages.filter((m) => m.role !== "system"));
                return { transcript, actions: [ask(transcript)], state };
            }
            case "client.action": {
                const base = withSystem(history);
                return { transcript: base, actions: [ask(base)], state };
            }
            case "effect.execute": {
                const { id, attempt } = trigger;
                if (trigger.kind === "llm_call") {
                    try {
                        // biome-ignore lint/style/noNonNullAssertion: the Llm union guarantees a worker LLM has `run`
                        const response = await config.llm.run!(trigger.request, { emitDelta: d.emitDelta });
                        const action: WorkerAction = {
                            type: "effect.result",
                            kind: "llm_call",
                            id,
                            response,
                            attempt,
                        };
                        return { transcript: history, actions: [action], state };
                    } catch (error) {
                        const message = error instanceof Error ? error.message : String(error);
                        const action: WorkerAction = {
                            type: "effect.error",
                            kind: "llm_call",
                            id,
                            error: message,
                            retryable: false,
                            attempt,
                        };
                        return { transcript: history, actions: [action], state };
                    }
                }
                const def = toolMap[trigger.name];
                if (!def)
                    return {
                        actions: [
                            {
                                type: "effect.error",
                                kind: "tool_call",
                                id,
                                error: `Unknown tool: ${trigger.name}`,
                                retryable: false,
                                attempt,
                            },
                        ],
                        state,
                    };
                const ctx: ToolExecutionContext = {
                    sessionId: d.session_id,
                    toolCallId: id,
                    attempt,
                    request: d,
                    defer: () => DEFERRED,
                };
                try {
                    const out = await def.execute(trigger.arguments, ctx);
                    if (out === DEFERRED) return { state };
                    return {
                        actions: [
                            {
                                type: "effect.result",
                                kind: "tool_call",
                                id,
                                result: typeof out === "string" ? out : "",
                                attempt,
                            },
                        ],
                        state,
                    };
                } catch (error) {
                    const message = error instanceof Error ? error.message : String(error);
                    return {
                        actions: [
                            {
                                type: "effect.error",
                                kind: "tool_call",
                                id,
                                error: message,
                                retryable: false,
                                attempt,
                            },
                        ],
                        state,
                    };
                }
            }
            case "effect.settled": {
                if (trigger.kind === "llm_call") {
                    if (!trigger.ok || trigger.message === undefined) return { state };
                    const assistant = stamp(trigger.message);
                    const transcript = [...history, assistant];
                    const calls = assistant.tool_calls ?? [];
                    if (calls.length === 0)
                        return { transcript, actions: [{ type: "done", data: assistant.content ?? null }], state };

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
                    return { transcript, actions, state };
                }
                const node: Message =
                    trigger.kind === "sub_agent"
                        ? {
                              id: crypto.randomUUID(),
                              role: "tool",
                              content: trigger.result,
                              tool_call_id: trigger.tool_call_id,
                              name: trigger.agent_id,
                          }
                        : {
                              id: crypto.randomUUID(),
                              role: "tool",
                              content: trigger.result,
                              tool_call_id: trigger.id,
                              name: trigger.name,
                          };
                const transcript = [...history, node];
                if (d.pending_effects > 0) return { transcript, state };
                return { transcript, actions: [ask(transcript)], state };
            }
            default:
                return { state };
        }
    };
}

/** Name a decision function so it can be deployed (`worker([...])`) or used as a
 *  sub-agent. `decide` is `toolLoop(...)` for the default loop, or your own. */
export function agent<S = unknown>(config: AgentConfig<S>): NamedAgent<S> {
    return Object.assign(config.decide, { agentName: config.name });
}
