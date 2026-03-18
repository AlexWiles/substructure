import type {
  WorkerAction,
  Message,
  LlmTool,
  RetryPolicy,
  Uuid,
} from "./types";
import type { HandlerContext, HandlerResult, Composable, StateContributor, MiddlewareFn, ToolDef } from "./worker";

// ── Types ────────────────────────────────────────────────────────────────────

export interface LlmConfig {
  model: string;
  client?: string;
  retry?: RetryPolicy;
}

export interface AgentOptions {
  id: string;
  description?: string;
  llm: LlmConfig;
  systemPrompt?: string;
  tools?: Record<string, ToolDef>;
  subAgents?: Agent[];
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

// ── Agent ────────────────────────────────────────────────────────────────────

interface RegisteredTool {
  definition: LlmTool;
  execute: (args: Record<string, unknown>) => Promise<unknown> | unknown;
}

interface RegisteredSubAgent {
  definition: LlmTool;
  agentId: string;
}

export class Agent implements Composable<AgentState> {
  readonly id: string;
  readonly description?: string;
  private llm: LlmConfig & { client: string; retry: RetryPolicy };
  private systemPrompt?: string;
  private tools: Map<string, RegisteredTool> = new Map();
  private subAgents: Map<string, RegisteredSubAgent> = new Map();

  constructor(options: AgentOptions) {
    this.id = options.id;
    this.description = options.description;
    this.systemPrompt = options.systemPrompt;
    this.llm = {
      model: options.llm.model,
      client: options.llm.client ?? "openrouter",
      retry: options.llm.retry ?? DEFAULT_RETRY,
    };

    if (options.tools) {
      for (const [name, def] of Object.entries(options.tools)) {
        this.tools.set(name, {
          definition: { function: { name, description: def.description, parameters: def.parameters } },
          execute: def.execute,
        });
      }
    }

    if (options.subAgents) {
      for (const agent of options.subAgents) {
        this.subAgents.set(agent.id, {
          agentId: agent.id,
          definition: {
            function: {
              name: agent.id,
              description: agent.description ?? agent.systemPrompt ?? `Delegate to ${agent.id}`,
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
      }
    }
  }

  toMiddleware(): StateContributor<AgentState> {
    const self = this;
    const mw: MiddlewareFn<any, any> = (ctx, next) => {
      if (ctx.request.agent_id !== self.id) return next(ctx);
      if (!ctx.state.messages) ctx.state.messages = [];
      if (!ctx.state.pendingSubAgents) ctx.state.pendingSubAgents = {};
      return self.handle(ctx as HandlerContext<AgentState>);
    };
    return Object.assign(mw, { _contributes: undefined as unknown as AgentState });
  }

  private handle(ctx: HandlerContext<AgentState>): HandlerResult | Promise<HandlerResult> {
    const { trigger, state } = ctx;

    switch (trigger.type) {
      case "user_message": {
        state.messages.push(trigger.message);
        return { actions: this.callLlmActions(state) };
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

              state.pendingSubAgents[childSessionId] = {
                toolCallId: tc.id,
                name: tc.function.name,
              };

              actions.push(
                {
                  type: "spawn_sub_agent",
                  session_id: childSessionId,
                  agent_id: sub.agentId,
                  retry: this.llm.retry,
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
                retry: this.llm.retry,
              });
            }
          }

          return { actions };
        }

        const lastMsg = state.messages[state.messages.length - 1];
        const artifacts = lastMsg?.content
          ? [{ parts: [{ kind: "text" as const, text: lastMsg.content }] }]
          : [];

        return { actions: [{ type: "done", artifacts }] };
      }

      case "tool_execute": {
        const tool = this.tools.get(trigger.name);
        if (!tool) {
          return {
            actions: [{
              type: "return_tool_error",
              tool_call_id: trigger.tool_call_id,
              error: `Unknown tool: ${trigger.name}`,
              retryable: false,
              attempt: trigger.attempt,
            }],
          };
        }

        return (async () => {
          try {
            const args = JSON.parse(trigger.arguments);
            const result = await tool.execute(args);
            return {
              actions: [{
                type: "return_tool_result" as const,
                tool_call_id: trigger.tool_call_id,
                result: typeof result === "string" ? result : JSON.stringify(result),
                attempt: trigger.attempt,
              }],
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
            };
          }
        })();
      }

      case "tool_result": {
        state.messages.push({
          role: "tool",
          content: trigger.result.content,
          tool_call_id: trigger.result.tool_call_id,
          name: trigger.result.name,
        });
        return { actions: this.callLlmActions(state) };
      }

      case "sub_agent_turn_complete": {
        const pending = state.pendingSubAgents[trigger.session_id];
        if (!pending) {
          return { actions: [{ type: "done", artifacts: [] }] };
        }

        delete state.pendingSubAgents[trigger.session_id];

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

        return { actions: this.callLlmActions(state) };
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
          return { actions: this.callLlmActions(state) };
        }
        return { actions: [{ type: "done", artifacts: [] }] };
      }

      case "llm_error":
      default: {
        return { actions: [{ type: "done", artifacts: [] }] };
      }
    }
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
        model: this.llm.model,
        messages,
        ...(toolDefs.length > 0 ? { tools: toolDefs } : {}),
      },
      stream: false,
      llm_client: this.llm.client,
      retry: this.llm.retry,
    }];
  }
}
