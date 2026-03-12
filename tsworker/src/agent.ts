/** Agent definition with tool registration and decide loop. */

import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import type {
  Artifact,
  DecisionTrigger,
  LlmRequest,
  LlmTool,
  Message,
  RetryPolicy,
  SpanContext,
  ToolCall,
  WorkerAction,
} from "./types.js";
import { RetryPolicies } from "./types.js";

// --- Run context ---

export interface RunContext {
  sessionId: string;
  tenantId: string;
  agentId: string;
  decisionId: string;
  attempts: number;
  span?: SpanContext;
}

// --- Tool definition ---

interface ToolDef<T extends z.ZodTypeAny = z.ZodTypeAny> {
  description: string;
  parameters: T;
  execute: (ctx: RunContext, params: z.infer<T>) => Promise<string>;
  retry?: RetryPolicy;
}

interface InternalToolDef {
  spec: LlmTool;
  execute: (ctx: RunContext, args: string) => Promise<string>;
  retry: RetryPolicy;
}

// --- Agent state ---

interface AgentStateData {
  messages: Message[];
}

function stateFromBytes(data: Uint8Array): AgentStateData {
  if (data.length === 0) return { messages: [] };
  const text = new TextDecoder().decode(data);
  const parsed = JSON.parse(text) as { messages?: Message[] };
  return { messages: parsed.messages ?? [] };
}

function stateToBytes(state: AgentStateData): Uint8Array {
  return new TextEncoder().encode(JSON.stringify({ messages: state.messages }));
}

function cleanMessage(msg: Message): Message {
  const clean: Message = { role: msg.role };
  if (msg.content != null) clean.content = msg.content;
  if (msg.tool_calls != null) clean.tool_calls = msg.tool_calls;
  if (msg.tool_call_id != null) clean.tool_call_id = msg.tool_call_id;
  if (msg.name != null) clean.name = msg.name;
  return clean;
}

// --- Agent ---

export interface AgentOptions {
  model?: string;
  instructions?: string;
  llmClient?: string;
  temperature?: number;
  maxCompletionTokens?: number;
  retry?: RetryPolicy;
}

export class Agent {
  readonly name: string;
  readonly model: string;
  readonly instructions: string;
  readonly llmClient: string;
  readonly temperature?: number;
  readonly maxCompletionTokens?: number;
  readonly retry: RetryPolicy;

  private _tools = new Map<string, InternalToolDef>();

  constructor(name: string, options: AgentOptions = {}) {
    this.name = name;
    this.model = options.model ?? "gpt-4o";
    this.instructions = options.instructions ?? "";
    this.llmClient = options.llmClient ?? "";
    this.temperature = options.temperature;
    this.maxCompletionTokens = options.maxCompletionTokens;
    this.retry = options.retry ?? RetryPolicies.default();
  }

  tool<T extends z.ZodObject<z.ZodRawShape>>(
    name: string,
    def: ToolDef<T>,
  ): void {
    const jsonSchema = zodToJsonSchema(def.parameters, { target: "openAi" });
    const spec: LlmTool = {
      function: {
        name,
        description: def.description,
        parameters: jsonSchema,
      },
    };
    this._tools.set(name, {
      spec,
      retry: def.retry ?? RetryPolicies.noRetry(),
      execute: async (ctx, argsJson) => {
        const parsed = def.parameters.parse(
          argsJson ? JSON.parse(argsJson) : {},
        );
        return def.execute(ctx, parsed);
      },
    });
  }

  // --- Decision entry point ---

  async decide(
    trigger: DecisionTrigger,
    workerState: Uint8Array,
    ctx: RunContext,
  ): Promise<[WorkerAction[], Uint8Array]> {
    const state = stateFromBytes(workerState);

    switch (trigger.type) {
      case "user_message": {
        state.messages.push(cleanMessage(trigger.message));
        return [this._requestLlm(state), stateToBytes(state)];
      }

      case "llm_completed": {
        state.messages.push(cleanMessage(trigger.message));
        if (trigger.message.tool_calls?.length) {
          const names = trigger.message.tool_calls.map(
            (tc) => tc.function.name,
          );
          await this._executeTools(trigger.message.tool_calls, state, ctx);
          return [this._requestLlm(state), stateToBytes(state)];
        }
        const text = trigger.message.content ?? "";
        return [this._done(text), stateToBytes(state)];
      }

      case "llm_failed": {
        return [this._done(`LLM error: ${trigger.error}`), stateToBytes(state)];
      }

      case "tool_resolved": {
        state.messages.push({
          role: "tool",
          content: trigger.result.content,
          tool_call_id: trigger.result.tool_call_id,
        });
        return [this._requestLlm(state), stateToBytes(state)];
      }

      case "stall": {
        return [this._requestLlm(state), stateToBytes(state)];
      }

      default: {
        return [this._requestLlm(state), stateToBytes(state)];
      }
    }
  }

  // --- Internals ---

  private async _executeTools(
    toolCalls: ToolCall[],
    state: AgentStateData,
    ctx: RunContext,
  ): Promise<void> {
    const results = await Promise.allSettled(
      toolCalls.map((tc) => this._executeOneTool(tc, ctx)),
    );

    for (let i = 0; i < toolCalls.length; i++) {
      const tc = toolCalls[i];
      const result = results[i];
      let content: string;

      if (result.status === "fulfilled") {
        content = result.value;
      } else {
        content = `Error: ${result.reason}`;
      }

      state.messages.push({
        role: "tool",
        content,
        tool_call_id: tc.id,
      });
    }
  }

  private async _executeOneTool(
    tc: ToolCall,
    ctx: RunContext,
  ): Promise<string> {
    const toolDef = this._tools.get(tc.function.name);
    if (!toolDef) throw new Error(`Unknown tool: ${tc.function.name}`);

    const { max_retries, backoff_base_secs, backoff_max_secs } = toolDef.retry;
    for (let attempt = 0; ; attempt++) {
      try {
        return await toolDef.execute(ctx, tc.function.arguments);
      } catch (err) {
        if (attempt >= max_retries) throw err;
        const delay = Math.min(
          backoff_base_secs ** (attempt + 1) * 1000,
          backoff_max_secs * 1000,
        );
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }

  private _llmTools(): LlmTool[] | undefined {
    if (this._tools.size === 0) return undefined;
    return Array.from(this._tools.values()).map((t) => t.spec);
  }

  private _requestLlm(state: AgentStateData): WorkerAction[] {
    const messages: Message[] = [...state.messages];
    if (this.instructions) {
      messages.unshift({ role: "system", content: this.instructions });
    }

    const request: LlmRequest = {
      model: this.model,
      messages,
      tools: this._llmTools(),
      temperature: this.temperature,
      max_completion_tokens: this.maxCompletionTokens,
    };

    // Strip undefined fields
    const cleanRequest = JSON.parse(JSON.stringify(request)) as LlmRequest;

    return [
      {
        type: "request_llm",
        request: cleanRequest,
        stream: false,
        llm_client: this.llmClient,
        retry: this.retry,
      },
    ];
  }

  private _done(text: string): WorkerAction[] {
    const artifacts: Artifact[] = [];
    if (text) {
      artifacts.push({ parts: [{ kind: "text", text }] });
    }
    return [{ type: "done", artifacts }];
  }
}
