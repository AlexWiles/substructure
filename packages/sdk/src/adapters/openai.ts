// ── OpenAI adapter (`@substructure.ai/sdk/adapters/openai`) ───────────────────────────
//
// Lets an OpenAI user run an existing agent on Substructure. You author with the
// OpenAI Agents SDK (`new Agent({...})` + `tool()`) and execution runs through
// the core `openai` Responses API. Substructure always owns the loop: each
// `llm.request` runs exactly one `responses.create` step (tools are passed as
// definitions only, so the model returns function calls instead of executing
// them) and Substructure executes the tools and iterates.
//
// Two authoring surfaces, one execution path:
//   const assistant = new OpenAIAgent({ model, instructions, tools });
//   const assistant = new OpenAIAgent(existingAgentsSdkAgent);  // an @openai/agents Agent
//   agent({ id }).use(assistant);
//
// Like `./ai`, this is the only entry importing `openai`/`@openai/agents`, kept
// off the main `@substructure.ai/sdk` entry so consumers who don't use it never
// pull it into their bundle.

import { Agent, RunContext } from "@openai/agents";
import type { ModelSettings, ModelSettingsToolChoice, Tool } from "@openai/agents";
import OpenAI from "openai";

import { llmLoop, messageHistory, tools } from "../middleware";
import type { LlmLoopSelection, ToolDef, ToolExecutionContext } from "../middleware";
import { contentText } from "../types";
import type { Message, StreamPart, ToolCall } from "../types";
import type { MiddlewareFn, MiddlewareSource, Next } from "../worker";

type ResponsesParams = OpenAI.Responses.ResponseCreateParamsStreaming;
type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type ResponseStreamEvent = OpenAI.Responses.ResponseStreamEvent;
type ResponseOutputItem = OpenAI.Responses.ResponseOutputItem;

export interface OpenAIAgentSettings {
    model: string;
    instructions?: string;
    tools?: Tool[];
    modelSettings?: ModelSettings;
    toolChoice?: ModelSettingsToolChoice;
    /** Existing OpenAI client to reuse. Defaults to `new OpenAI()` (which reads
     *  `OPENAI_API_KEY`), or `new OpenAI({ apiKey })` when `apiKey` is set. */
    client?: OpenAI;
    apiKey?: string;
    /** Run context passed to each tool's `execute` (the OpenAI Agents SDK
     *  `RunContext` local context). */
    context?: unknown;
}

interface ResolvedSettings {
    client: OpenAI;
    model: string;
    instructions?: string | (() => string | Promise<string>);
    tools: Tool[];
    modelSettings?: ModelSettings;
    toolChoice?: ModelSettingsToolChoice;
    context?: unknown;
}

// An OpenAI agent you drop into a Substructure handler chain. Accepts either an
// `OpenAIAgentSettings` object or an existing `@openai/agents` `Agent` instance;
// either way Substructure drives the loop and runs the tools, and each
// `llm.request` is one `responses.create` step.
export class OpenAIAgent implements MiddlewareSource {
    private readonly settings: ResolvedSettings;

    constructor(input: OpenAIAgentSettings | Agent, options?: { client?: OpenAI; apiKey?: string; context?: unknown }) {
        this.settings = input instanceof Agent ? fromAgent(input, options) : resolveSettings(input);
    }

    get tools(): Tool[] {
        return this.settings.tools;
    }

    toMiddleware(): MiddlewareFn<unknown> {
        return openAIAgent(this.settings);
    }
}

function resolveSettings(settings: OpenAIAgentSettings): ResolvedSettings {
    return {
        client: settings.client ?? new OpenAI(settings.apiKey ? { apiKey: settings.apiKey } : {}),
        model: settings.model,
        instructions: settings.instructions,
        tools: settings.tools ?? [],
        modelSettings: settings.modelSettings,
        toolChoice: settings.toolChoice ?? settings.modelSettings?.toolChoice,
        context: settings.context,
    };
}

function fromAgent(agent: Agent, options?: { client?: OpenAI; apiKey?: string; context?: unknown }): ResolvedSettings {
    if (typeof agent.model !== "string") {
        throw new Error(
            "OpenAIAgent executes via the OpenAI Responses API and needs a model id string. Pass `model` as a string on the Agent, or use `new OpenAIAgent({ model, ... })`.",
        );
    }
    const context = options?.context;
    const instructions =
        typeof agent.instructions === "function"
            ? () =>
                  (agent.instructions as (rc: RunContext, a: Agent) => string | Promise<string>)(
                      new RunContext(context),
                      agent,
                  )
            : agent.instructions;
    return {
        client: options?.client ?? new OpenAI(options?.apiKey ? { apiKey: options.apiKey } : {}),
        model: agent.model,
        instructions,
        tools: agent.tools,
        modelSettings: agent.modelSettings,
        toolChoice: agent.modelSettings?.toolChoice,
        context,
    };
}

// Middleware that wires an OpenAI agent into a Substructure handler chain. It
// composes messageHistory + tools + llmLoop, so it behaves exactly as if those
// three were `.use()`d in order — Substructure drives the loop and runs the
// tools; each `llm.request` is one `responses.create` step.
export function openAIAgent(settings: ResolvedSettings): MiddlewareFn<unknown> {
    const ms = settings.modelSettings;
    const request: LlmLoopSelection["request"] = { model: settings.model };
    if (ms?.temperature !== undefined) request.temperature = ms.temperature;
    if (ms?.maxTokens !== undefined) request.max_completion_tokens = ms.maxTokens;

    const chain: MiddlewareFn<any, any>[] = [
        messageHistory(settings.instructions),
        tools(openAITools(settings.tools, settings.context)),
        llmLoop({ request, stream: true, handler: "worker", caller: openAICaller(settings) }),
    ];

    return (ctx, next) => {
        let fn: Next<unknown> = next;
        for (let i = chain.length - 1; i >= 0; i--) {
            const mw = chain[i];
            const downstream = fn;
            fn = (c) => mw(c, downstream);
        }
        return fn(ctx);
    };
}

// ── Tools (Substructure-side execution) ───────────────────────────────────────

export function openAITools(toolset: Tool[], context?: unknown): ToolDef[] {
    return toolset.flatMap((t): ToolDef[] => {
        if (t.type !== "function") return [];
        return [
            {
                name: t.name,
                description: t.description ?? "",
                parameters: t.parameters,
                execute: async (args: string, _state?: unknown, _ctx?: ToolExecutionContext) => {
                    const result = await t.invoke(new RunContext(context), args || "{}");
                    return typeof result === "string" ? result : JSON.stringify(result);
                },
            } satisfies ToolDef,
        ];
    });
}

// Function-tool definitions for the model: name/description/parameters/strict,
// with no `invoke` — the SDK returns function calls instead of running them
// (Substructure runs them).
function modelTools(toolset: Tool[]): OpenAI.Responses.Tool[] {
    return toolset.flatMap((t): OpenAI.Responses.Tool[] => {
        if (t.type !== "function") return [];
        return [
            {
                type: "function",
                name: t.name,
                description: t.description ?? "",
                // agents normalizes `parameters` to JSON Schema; widen to the
                // looser JSON-object shape the Responses tool param expects.
                parameters: t.parameters as Record<string, unknown>,
                strict: t.strict,
            },
        ];
    });
}

// ── Caller (one responses.create step per llm.request) ────────────────────────

function openAICaller(settings: ResolvedSettings): NonNullable<LlmLoopSelection["caller"]> {
    const { client } = settings;
    return async (request, ctx) => {
        const ms = settings.modelSettings;
        const modelToolList = modelTools(settings.tools);
        const params: ResponsesParams = {
            model: request.model,
            input: toResponsesInput(request.messages),
            stream: true,
            tools: modelToolList.length ? modelToolList : undefined,
            tool_choice: settings.toolChoice ? toResponsesToolChoice(settings.toolChoice) : undefined,
            temperature: request.temperature ?? ms?.temperature,
            max_output_tokens: request.max_completion_tokens ?? ms?.maxTokens,
            top_p: ms?.topP,
            parallel_tool_calls: ms?.parallelToolCalls,
            store: ms?.store,
            truncation: ms?.truncation,
            ...ms?.providerData,
        };

        const stream = await client.responses.create(params);
        const callIdByItem = new Map<string, string>();
        let final: OpenAI.Responses.Response | undefined;

        for await (const event of stream as AsyncIterable<ResponseStreamEvent>) {
            const part = toStreamPart(event, callIdByItem);
            if (part) await ctx.emitDelta?.(part);
            if (event.type === "response.completed") final = event.response;
        }

        const output = final?.output ?? [];
        const toolCalls = output
            .filter((i): i is Extract<ResponseOutputItem, { type: "function_call" }> => i.type === "function_call")
            .map(
                (i): ToolCall => ({
                    id: i.call_id,
                    type: "function",
                    function: { name: i.name, arguments: i.arguments },
                }),
            );

        const content = outputText(output);
        await ctx.emitDelta?.({ type: "finish", finishReason: toolCalls.length ? "tool_calls" : "stop" });
        return {
            model: request.model,
            content: content || undefined,
            tool_calls: toolCalls,
            finish_reason: toolCalls.length ? "tool_calls" : "stop",
            usage: final?.usage as Record<string, unknown> | undefined,
        };
    };
}

// ── Conversions ──────────────────────────────────────────────────────────────

export function toStreamPart(event: ResponseStreamEvent, callIdByItem: Map<string, string>): StreamPart | null {
    switch (event.type) {
        case "response.output_text.delta":
            return { type: "text-delta", delta: event.delta };
        case "response.reasoning_summary_text.delta":
            return { type: "reasoning-delta", delta: event.delta };
        case "response.output_item.added":
            if (event.item.type === "function_call") {
                callIdByItem.set(event.item.id ?? event.item.call_id, event.item.call_id);
                return { type: "tool-input-start", toolCallId: event.item.call_id, toolName: event.item.name };
            }
            return null;
        case "response.function_call_arguments.delta":
            return {
                type: "tool-input-delta",
                toolCallId: callIdByItem.get(event.item_id) ?? event.item_id,
                inputTextDelta: event.delta,
            };
        default:
            return null;
    }
}

export function toResponsesInput(messages: Message[]): ResponseInputItem[] {
    const out: ResponseInputItem[] = [];
    for (const m of messages) {
        switch (m.role) {
            case "system":
                out.push({ role: "system", content: contentText(m.content) });
                break;
            case "user":
                out.push({ role: "user", content: contentText(m.content) });
                break;
            case "assistant": {
                const text = contentText(m.content);
                if (text) out.push({ role: "assistant", content: text });
                for (const tc of m.tool_calls ?? []) {
                    out.push({
                        type: "function_call",
                        call_id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    });
                }
                break;
            }
            case "tool":
                out.push({
                    type: "function_call_output",
                    call_id: m.tool_call_id ?? "",
                    output: contentText(m.content),
                });
                break;
        }
    }
    return out;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function outputText(output: ResponseOutputItem[]): string {
    return output
        .filter((i): i is Extract<ResponseOutputItem, { type: "message" }> => i.type === "message")
        .flatMap((i) => i.content)
        .filter((c): c is OpenAI.Responses.ResponseOutputText => c.type === "output_text")
        .map((c) => c.text)
        .join("");
}

const TOOL_CHOICE_OPTIONS = ["auto", "required", "none"] as const;

function toResponsesToolChoice(choice: ModelSettingsToolChoice): ResponsesParams["tool_choice"] {
    const option = TOOL_CHOICE_OPTIONS.find((o) => o === choice);
    return option ?? { type: "function", name: choice };
}
