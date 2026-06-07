// ── AI SDK adapter (`@substructure.ai/sdk/ai`) ───────────────────────────────
//
// Lets a Vercel AI SDK user run their existing agent on Substructure by
// swapping `new ToolLoopAgent(settings)` for `new SubstructureAgent(settings)`.
// The result is a normal Substructure worker agent (a `Handler`): it answers
// decisions and is agnostic to whether an embedded or remote engine drives the
// loop. Substructure always owns the loop; each `llm.request` runs exactly one
// `streamText` step (tools are passed without `execute` so the model returns
// tool calls instead of executing them), and Substructure executes the tools
// and iterates.
//
// This is the only entry that imports `ai`, kept off the main `@substructure.ai/sdk`
// entry so consumers who don't use it never pull it into their bundle.

import { asSchema, streamText } from "ai";
import type { LanguageModel, ModelMessage, TextStreamPart, Tool, ToolChoice, ToolSet } from "ai";

import { llmLoop, messageHistory, tools } from "./middleware";
import type { LlmLoopSelection, ToolDef, ToolExecutionContext } from "./middleware";
import { contentText } from "./types";
import type { Message, StreamPart, ToolCall } from "./types";
import type { MiddlewareFn, MiddlewareSource, Next } from "./worker";

type StreamTextOptions = Parameters<typeof streamText>[0];
type ProviderOptions = NonNullable<StreamTextOptions["providerOptions"]>;
type ToolResultOutput = Awaited<ReturnType<NonNullable<Tool["toModelOutput"]>>>;

export interface SubstructureAgentSettings<TOOLS extends ToolSet = ToolSet> {
    model: LanguageModel;
    instructions?: string;
    tools?: TOOLS;
    toolChoice?: ToolChoice<TOOLS>;
    temperature?: number;
    maxOutputTokens?: number;
    topP?: number;
    topK?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
    stopSequences?: string[];
    seed?: number;
    providerOptions?: ProviderOptions;
    experimental_context?: unknown;
}

// An AI SDK agent you drop into a Substructure handler chain:
//
//   const assistant = new ToolLoopAgent({ model, instructions, tools });
//   agent({ id }).use(assistant);
//
// Mirrors the AI SDK `ToolLoopAgent` constructor, but instead of driving its own
// loop it hands a middleware to the builder. Substructure drives the loop and
// runs the tools; each `llm.request` is one `streamText` step.
export class ToolLoopAgent<TOOLS extends ToolSet = ToolSet> implements MiddlewareSource {
    constructor(private readonly settings: SubstructureAgentSettings<TOOLS>) {}

    get tools(): TOOLS {
        return (this.settings.tools ?? ({} as TOOLS)) as TOOLS;
    }

    toMiddleware(): MiddlewareFn<unknown> {
        return aiSdkAgent(this.settings);
    }
}

// Middleware that wires an AI SDK agent into a Substructure handler chain:
//
//   agent({ id }).use(aiSdkAgent({ model, instructions, tools }))
//
// It composes messageHistory + tools + llmLoop, so it behaves exactly as if
// those three were `.use()`d in order — Substructure drives the loop and runs
// the tools; each `llm.request` is one `streamText` step.
export function aiSdkAgent<TOOLS extends ToolSet>(settings: SubstructureAgentSettings<TOOLS>): MiddlewareFn<unknown> {
    const request: LlmLoopSelection["request"] = { model: modelId(settings.model) };
    if (settings.temperature !== undefined) request.temperature = settings.temperature;
    if (settings.maxOutputTokens !== undefined) request.max_completion_tokens = settings.maxOutputTokens;

    const chain: MiddlewareFn<any, any>[] = [
        messageHistory(settings.instructions),
        tools(settings.tools ? aiSdkTools(settings.tools, settings.experimental_context) : []),
        llmLoop({ request, stream: true, handler: "worker", caller: aiSdkCaller(settings) }),
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

export function aiSdkTools(toolset: ToolSet, experimentalContext?: unknown): ToolDef[] {
    return Object.entries(toolset).flatMap(([name, t]): ToolDef[] => {
        if (t.type === "provider") return [];

        const description = t.description ?? "";
        const parameters = asSchema(t.inputSchema).jsonSchema;
        const execute = t.execute;

        if (!execute) {
            return [
                {
                    name,
                    description,
                    parameters,
                    handler: "client",
                    execute: async (_args: string, _state?: unknown, ctx?: ToolExecutionContext) =>
                        ctx ? ctx.defer() : "",
                } satisfies ToolDef,
            ];
        }

        return [
            {
                name,
                description,
                parameters,
                execute: async (args: string, _state?: unknown, ctx?: ToolExecutionContext) => {
                    const input = args ? JSON.parse(args) : {};
                    const options = {
                        toolCallId: ctx?.toolCallId ?? "",
                        messages: [] as ModelMessage[],
                        experimental_context: experimentalContext,
                    };
                    const output = await execute(input, options);
                    if (t.toModelOutput) {
                        return modelOutputToString(
                            await t.toModelOutput({ toolCallId: options.toolCallId, input, output }),
                        );
                    }
                    return typeof output === "string" ? output : JSON.stringify(output);
                },
            } satisfies ToolDef,
        ];
    });
}

// Pass tools to the model with `execute`/`needsApproval` removed so the SDK
// returns tool calls instead of running them (Substructure runs them) and never
// pauses for approval — but keeps every model-facing field and the
// `onInput*` streaming callbacks intact.
function modelTools<T extends ToolSet>(toolset: T): T {
    const out: Record<string, Tool> = {};
    for (const [name, t] of Object.entries(toolset)) {
        const { execute: _execute, needsApproval: _needsApproval, ...rest } = t as Tool;
        out[name] = rest;
    }
    return out as T;
}

// ── Caller (one streamText step per llm.request) ──────────────────────────────

function aiSdkCaller<TOOLS extends ToolSet>(
    settings: SubstructureAgentSettings<TOOLS>,
): NonNullable<LlmLoopSelection["caller"]> {
    return async (request, ctx) => {
        const result = streamText({
            model: settings.model,
            messages: toModelMessages(request.messages),
            tools: settings.tools ? modelTools(settings.tools) : undefined,
            toolChoice: settings.toolChoice,
            temperature: request.temperature ?? settings.temperature,
            maxOutputTokens: request.max_completion_tokens ?? settings.maxOutputTokens,
            topP: settings.topP,
            topK: settings.topK,
            presencePenalty: settings.presencePenalty,
            frequencyPenalty: settings.frequencyPenalty,
            stopSequences: settings.stopSequences,
            seed: settings.seed,
            providerOptions: settings.providerOptions,
            experimental_context: settings.experimental_context,
        });

        for await (const part of result.fullStream) {
            const mapped = toStreamPart(part);
            if (mapped) await ctx.emitDelta?.(mapped);
        }

        const toolCalls = (await result.toolCalls).filter(
            (tc) => !(tc as { providerExecuted?: boolean }).providerExecuted,
        );
        return {
            model: request.model,
            content: (await result.text) || undefined,
            tool_calls: toolCalls.map(
                (tc): ToolCall => ({
                    id: tc.toolCallId,
                    type: "function",
                    function: { name: tc.toolName, arguments: JSON.stringify(tc.input) },
                }),
            ),
            finish_reason: await result.finishReason,
            usage: (await result.usage) as Record<string, unknown>,
        };
    };
}

// ── Conversions ──────────────────────────────────────────────────────────────

export function toStreamPart<T extends ToolSet>(part: TextStreamPart<T>): StreamPart | null {
    switch (part.type) {
        case "text-delta":
            return { type: "text-delta", delta: part.text };
        case "reasoning-delta":
            return { type: "reasoning-delta", delta: part.text };
        case "tool-input-start":
            return { type: "tool-input-start", toolCallId: part.id, toolName: part.toolName };
        case "tool-input-delta":
            return { type: "tool-input-delta", toolCallId: part.id, inputTextDelta: part.delta };
        case "finish":
            return { type: "finish", finishReason: part.finishReason };
        default:
            return null;
    }
}

export function toModelMessages(messages: Message[]): ModelMessage[] {
    const out: ModelMessage[] = [];
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
                const parts: unknown[] = [];
                if (text) parts.push({ type: "text", text });
                for (const tc of m.tool_calls ?? []) {
                    parts.push({
                        type: "tool-call",
                        toolCallId: tc.id,
                        toolName: tc.function.name,
                        input: safeParseJson(tc.function.arguments),
                    });
                }
                out.push({ role: "assistant", content: (parts.length ? parts : text) as never });
                break;
            }
            case "tool":
                out.push({
                    role: "tool",
                    content: [
                        {
                            type: "tool-result",
                            toolCallId: m.tool_call_id ?? "",
                            toolName: m.name ?? "",
                            output: { type: "text", value: contentText(m.content) },
                        },
                    ],
                });
                break;
        }
    }
    return out;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function modelOutputToString(output: ToolResultOutput): string {
    if (output.type === "text" || output.type === "error-text") return output.value;
    if ("value" in output) return JSON.stringify(output.value);
    return JSON.stringify(output);
}

function modelId(model: LanguageModel): string {
    return typeof model === "string" ? model : model.modelId;
}

function safeParseJson(text: string): unknown {
    try {
        return JSON.parse(text);
    } catch {
        return {};
    }
}
