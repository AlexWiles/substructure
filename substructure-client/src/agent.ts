import type {
  WorkerDecisionRequestWire,
  WorkerAction,
  Message,
  LlmTool,
  RetryPolicy,
  Uuid,
} from "./types";
import type { DecisionHandler } from "./worker";

// ── Types ────────────────────────────────────────────────────────────────────

export interface AgentOptions {
  id: string;
  model: string;
  systemPrompt?: string;
  llmClient?: string;
  retry?: RetryPolicy;
}

export type ToolFn = (args: Record<string, unknown>) => Promise<unknown> | unknown;

interface RegisteredTool {
  definition: LlmTool;
  handler: ToolFn;
}

interface RegisteredSubAgent {
  definition: LlmTool;
  agentId: string;
}

const DEFAULT_RETRY: RetryPolicy = {
  timeout_secs: 30,
  max_retries: 0,
  backoff_base_secs: 0,
  backoff_max_secs: 0,
};

// ── State ────────────────────────────────────────────────────────────────────

interface PendingSubAgent {
  toolCallId: string;
  name: string;
}

interface AgentState {
  messages: Message[];
  pendingSubAgents: Record<string, PendingSubAgent>;
}

function decodeState(raw: string): AgentState {
  if (!raw || raw === "") return { messages: [], pendingSubAgents: {} };
  const parsed = JSON.parse(Buffer.from(raw, "base64").toString("utf-8"));
  return { messages: parsed.messages ?? [], pendingSubAgents: parsed.pendingSubAgents ?? {} };
}

function encodeState(state: AgentState): string {
  return Buffer.from(JSON.stringify(state), "utf-8").toString("base64");
}

// ── Agent ────────────────────────────────────────────────────────────────────

export class Agent {
  readonly id: string;
  private model: string;
  private systemPrompt?: string;
  private llmClient: string;
  private retry: RetryPolicy;
  private tools: Map<string, RegisteredTool> = new Map();
  private subAgents: Map<string, RegisteredSubAgent> = new Map();

  constructor(options: AgentOptions) {
    this.id = options.id;
    this.model = options.model;
    this.systemPrompt = options.systemPrompt;
    this.llmClient = options.llmClient ?? "openrouter";
    this.retry = options.retry ?? DEFAULT_RETRY;
  }

  tool(
    name: string,
    description: string,
    parameters: unknown,
    handler: ToolFn,
  ): this {
    this.tools.set(name, {
      definition: { function: { name, description, parameters } },
      handler,
    });
    return this;
  }

  subAgent(agent: Agent, description?: string): this {
    this.subAgents.set(agent.id, {
      agentId: agent.id,
      definition: {
        function: {
          name: agent.id,
          description: description ?? agent.systemPrompt ?? `Delegate to ${agent.id}`,
          parameters: {
            type: "object",
            properties: {
              message: { type: "string", description: "The message to send to the agent" },
            },
            required: ["message"],
          },
        },
      },
    });
    return this;
  }

  handler(): DecisionHandler {
    return async (request: WorkerDecisionRequestWire) => {
      const { trigger } = request;
      const state = decodeState(request.worker_state);

      switch (trigger.type) {
        case "user_message": {
          state.messages.push(trigger.message);
          return { actions: this.callLlmActions(state), state: encodeState(state) };
        }

        case "llm_response": {
          state.messages.push(trigger.message);

          if (trigger.message.tool_calls?.length) {
            const actions: WorkerAction[] = [];

            for (const tc of trigger.message.tool_calls) {
              const sub = this.subAgents.get(tc.function.name);
              if (sub) {
                const childSessionId = crypto.randomUUID() as Uuid;
                const args = JSON.parse(tc.function.arguments);

                // Track the pending sub-agent so we can resolve the tool call when it completes
                state.pendingSubAgents[childSessionId] = {
                  toolCallId: tc.id,
                  name: tc.function.name,
                };

                actions.push(
                  {
                    type: "spawn_sub_agent",
                    session_id: childSessionId,
                    agent_id: sub.agentId,
                    retry: this.retry,
                  },
                  {
                    type: "send_message",
                    session_id: childSessionId,
                    message: { role: "user", content: args.message },
                  },
                );
              } else {
                actions.push({
                  type: "call_tool",
                  tool_call_id: tc.id,
                  name: tc.function.name,
                  arguments: tc.function.arguments,
                  handler: "worker",
                  retry: this.retry,
                });
              }
            }

            return { actions, state: encodeState(state) };
          }

          // Pack last assistant message into artifacts
          const lastMsg = state.messages[state.messages.length - 1];
          const artifacts = lastMsg?.content
            ? [{ parts: [{ kind: "text" as const, text: lastMsg.content }] }]
            : [];

          return {
            actions: [{ type: "done" as const, artifacts }],
            state: encodeState(state),
          };
        }

        case "tool_execute": {
          const tool = this.tools.get(trigger.name);
          if (!tool) {
            return {
              actions: [{
                type: "return_tool_error" as const,
                tool_call_id: trigger.tool_call_id,
                error: `Unknown tool: ${trigger.name}`,
                retryable: false,
                attempt: trigger.attempt,
              }],
              state: encodeState(state),
            };
          }

          try {
            const args = JSON.parse(trigger.arguments);
            const result = await tool.handler(args);
            return {
              actions: [{
                type: "return_tool_result" as const,
                tool_call_id: trigger.tool_call_id,
                result: typeof result === "string" ? result : JSON.stringify(result),
                attempt: trigger.attempt,
              }],
              state: encodeState(state),
            };
          } catch (e: any) {
            return {
              actions: [{
                type: "return_tool_error" as const,
                tool_call_id: trigger.tool_call_id,
                error: e.message,
                retryable: false,
                attempt: trigger.attempt,
              }],
              state: encodeState(state),
            };
          }
        }

        case "tool_result": {
          state.messages.push({
            role: "tool",
            content: trigger.result.content,
            tool_call_id: trigger.result.tool_call_id,
            name: trigger.result.name,
          });
          return { actions: this.callLlmActions(state), state: encodeState(state) };
        }

        case "sub_agent_turn_complete": {
          const pending = state.pendingSubAgents[trigger.session_id];
          if (!pending) {
            return {
              actions: [{ type: "done" as const, artifacts: [] }],
              state: encodeState(state),
            };
          }

          delete state.pendingSubAgents[trigger.session_id];

          // Extract text from artifacts for the tool result
          const parts = trigger.artifacts.flatMap((a) => a.parts);
          const text = parts
            .map((p) => (p.kind === "text" ? p.text : JSON.stringify(p.data)))
            .join("\n");

          state.messages.push({
            role: "tool",
            content: text || `Sub-agent ${trigger.agent_id} completed with no output.`,
            tool_call_id: pending.toolCallId,
            name: pending.name,
          });

          return { actions: this.callLlmActions(state), state: encodeState(state) };
        }

        case "sub_agent_error": {
          const pending = state.pendingSubAgents[trigger.session_id];
          if (pending) {
            delete state.pendingSubAgents[trigger.session_id];
            state.messages.push({
              role: "tool",
              content: `Sub-agent ${trigger.agent_id} failed: ${trigger.error}`,
              tool_call_id: pending.toolCallId,
              name: pending.name,
            });
            return { actions: this.callLlmActions(state), state: encodeState(state) };
          }
          return {
            actions: [{ type: "done" as const, artifacts: [] }],
            state: encodeState(state),
          };
        }

        case "llm_error":
        default: {
          return {
            actions: [{ type: "done" as const, artifacts: [] }],
            state: encodeState(state),
          };
        }
      }
    };
  }

  private callLlmActions(state: AgentState): WorkerAction[] {
    const messages: Message[] = [];
    if (this.systemPrompt) {
      messages.push({ role: "system", content: this.systemPrompt });
    }
    messages.push(...state.messages);

    const toolDefs = [
      ...Array.from(this.tools.values()).map((t) => t.definition),
      ...Array.from(this.subAgents.values()).map((s) => s.definition),
    ];

    return [{
      type: "call_llm",
      request: {
        model: this.model,
        messages,
        ...(toolDefs.length > 0 ? { tools: toolDefs } : {}),
      },
      stream: false,
      llm_client: this.llmClient,
      retry: this.retry,
    }];
  }
}
