// AI SDK adapter (`@substructure.ai/sdk/adapters/ai`). `aiGenerate` is an
// `LlmGenerator` backed by `streamText`; `ToolLoopAgent` wires an AI SDK agent into
// a handler chain. Substructure owns the loop; each `llm.request` runs one
// `streamText` step.

import { asSchema, jsonSchema, streamText, tool } from "ai";
import type { LanguageModel, ModelMessage, TextStreamPart, Tool, ToolChoice, ToolSet } from "ai";

import { llmToolLoop, messageHistory, tools } from "../middleware";
import type { LlmGenerate, LlmGenerator, ToolDef, ToolExecutionContext } from "../middleware";
import { contentText } from "../types";
import type { LlmTool, Message, StreamPart, ToolCall } from "../types";
import type { MiddlewareFn, MiddlewareSource, Next } from "../worker";

type StreamTextOptions = Parameters<typeof streamText>[0];
type ToolResultOutput = Awaited<ReturnType<NonNullable<Tool["toModelOutput"]>>>;

// `streamText`'s own options minus what the loop supplies: `messages`/`prompt`/
// `system` come from the transcript, and tools are declared in `tools()`.
export type AIGenerateSettings = Omit<StreamTextOptions, "messages" | "prompt" | "system" | "tools">;

export function aiGenerate(settings: AIGenerateSettings): LlmGenerator {
    const request: LlmGenerator["request"] = { model: modelId(settings.model) };
    if (settings.temperature !== undefined) request.temperature = settings.temperature;
    if (settings.maxOutputTokens !== undefined) request.max_completion_tokens = settings.maxOutputTokens;

    const run: LlmGenerate = async (req, ctx) => {
        const result = streamText({
            ...settings,
            model: settings.model,
            messages: toModelMessages(req.messages),
            // The system prompt is the agent's own instructions, not user input.
            allowSystemInMessages: true,
            tools: modelTools(req.tools),
            temperature: req.temperature ?? settings.temperature,
            maxOutputTokens: req.max_completion_tokens ?? settings.maxOutputTokens,
        });

        for await (const part of result.fullStream) {
            const mapped = toStreamPart(part);
            if (mapped) await ctx.emitDelta?.(mapped);
        }

        const toolCalls = (await result.toolCalls).filter(
            (tc) => !(tc as { providerExecuted?: boolean }).providerExecuted,
        );
        return {
            model: req.model,
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

    return { request, handler: "worker", stream: true, run };
}

// Model-facing tools from `request.tools` — schema only, no `execute`, so the SDK
// returns tool calls instead of running them (Substructure runs them).
function modelTools(toolList: LlmTool[] | undefined): ToolSet {
    const out: ToolSet = {};
    for (const t of toolList ?? []) {
        out[t.function.name] = tool({
            description: t.function.description || undefined,
            inputSchema: jsonSchema(t.function.parameters as Parameters<typeof jsonSchema>[0]),
        });
    }
    return out;
}

export type SubstructureAgentSettings<TOOLS extends ToolSet = ToolSet> = Omit<
    AIGenerateSettings,
    "tools" | "toolChoice"
> & {
    instructions?: string;
    tools?: TOOLS;
    toolChoice?: ToolChoice<TOOLS>;
};

// Mirrors the AI SDK `ToolLoopAgent`, but hands a middleware to the builder
// instead of driving its own loop.
export class ToolLoopAgent<TOOLS extends ToolSet = ToolSet> implements MiddlewareSource {
    constructor(private readonly settings: SubstructureAgentSettings<TOOLS>) {}

    get tools(): TOOLS {
        return (this.settings.tools ?? ({} as TOOLS)) as TOOLS;
    }

    toMiddleware(): MiddlewareFn<unknown> {
        return aiSdkAgent(this.settings);
    }
}

// Composes `messageHistory` + `tools` + `llmToolLoop`, behaving as if those three
// were `.use()`d in order.
export function aiSdkAgent<TOOLS extends ToolSet>(settings: SubstructureAgentSettings<TOOLS>): MiddlewareFn<unknown> {
    const { instructions, tools: toolset, ...generateSettings } = settings;
    const generator = aiGenerate(generateSettings);

    const chain: MiddlewareFn<any, any>[] = [
        messageHistory(instructions),
        tools(toolset ? aiSdkTools(toolset, settings.experimental_context) : []),
        llmToolLoop({ generator }),
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

// AI SDK tools as Substructure-executed `ToolDef`s.
export function aiSdkTools(toolset: ToolSet, experimentalContext?: unknown): ToolDef[] {
    return Object.entries(toolset).flatMap(([name, t]): ToolDef[] => {
        if (t.type === "provider") return [];

        const description = t.description ?? "";
        const parameters = asSchema(t.inputSchema).jsonSchema;
        const execute = t.execute;

        // No `execute` means a client tool: the worker never runs it (the frontend
        // completes it via `submitToolCallResult`), so this only satisfies the type.
        if (!execute) {
            return [
                {
                    name,
                    description,
                    parameters,
                    handler: "client",
                    execute: async () => {
                        throw new Error(
                            `Client tool "${name}" has no server-side execute and must be completed by the frontend.`,
                        );
                    },
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
