// AI SDK adapter (`@substructure.ai/sdk/adapters/ai`). `aiGenerate` is an
// `Llm` backed by `streamText`; `aiSdkAgent` builds a `toolLoop` from an
// AI SDK toolset. Substructure owns the loop; each `llm.request` runs one
// `streamText` step.

import type { LanguageModel, ModelMessage, TextStreamPart, Tool, ToolChoice, ToolSet } from "ai";
import { asSchema, jsonSchema, streamText, tool } from "ai";
import { toolLoop } from "../agent";
import type { Agent, Llm, LlmGenerate, ToolDef } from "../core";
import type { LlmParams, LlmTokenDeltaInput, LlmTool, Message, ToolCall } from "../types";
import { contentText } from "../types";

type StreamTextOptions = Parameters<typeof streamText>[0];
type ToolResultOutput = Awaited<ReturnType<NonNullable<Tool["toModelOutput"]>>>;

// `streamText`'s own options minus what the loop supplies: `messages`/`prompt`/
// `system` come from the transcript, and tools are declared in `tools()`.
export type AIGenerateSettings = Omit<StreamTextOptions, "messages" | "prompt" | "system" | "tools">;

export function aiGenerate(settings: AIGenerateSettings): Llm {
    const request: LlmParams = { model: modelId(settings.model) };
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
            const delta = toDelta(part);
            if (delta) await ctx.emitDelta?.(delta);
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

    return { ...request, handler: "worker", stream: true, run };
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

export type AiAgentSettings<TOOLS extends ToolSet = ToolSet> = Omit<AIGenerateSettings, "tools" | "toolChoice"> & {
    instructions?: string;
    tools?: TOOLS;
    toolChoice?: ToolChoice<TOOLS>;
};

/** The loop for running an AI SDK toolset on Substructure: `aiGenerate` as the
 *  model, the toolset converted to worker-executed tools. Name and deploy it with
 *  `worker([agent({ name, decide: aiSdkAgent(...) })])`. */
export function aiSdkAgent<TOOLS extends ToolSet>(settings: AiAgentSettings<TOOLS>): Agent {
    const { instructions, tools: toolset, ...generateSettings } = settings;
    return toolLoop({
        llm: aiGenerate(generateSettings),
        instructions,
        tools: toolset ? aiSdkTools(toolset, settings.experimental_context) : [],
    });
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
                execute: async (args, ctx) => {
                    const input = args ? JSON.parse(args) : {};
                    const options = {
                        toolCallId: ctx.toolCallId,
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

export function toDelta<T extends ToolSet>(part: TextStreamPart<T>): LlmTokenDeltaInput | null {
    switch (part.type) {
        case "text-delta":
            return { text: part.text };
        case "reasoning-delta":
            return { reasoning: part.text };
        case "tool-input-start":
            return { tool_calls: [{ id: part.id, name: part.toolName }] };
        case "tool-input-delta":
            return { tool_calls: [{ id: part.id, arguments: part.delta }] };
        case "finish":
            return part.finishReason ? { finish_reason: part.finishReason } : null;
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
