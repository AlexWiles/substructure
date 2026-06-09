// Anthropic Messages API generator for `llmToolLoop`. Each `llm.request` runs one
// `messages.stream` call.

import Anthropic from "@anthropic-ai/sdk";

import type { LlmGenerate, LlmGenerator } from "../middleware";
import { contentText } from "../types";
import type { LlmTool, Message, StreamPart, ToolCall } from "../types";

// `MessageCreateParams` minus what the loop supplies: `messages`/`system` and
// `tools` come from the request; `stream` is set on `llmToolLoop`.
export type AnthropicGenerateSettings = Omit<
    Anthropic.MessageCreateParamsNonStreaming,
    "messages" | "system" | "tools" | "stream"
> & {
    /** Defaults to `new Anthropic()` (reads `ANTHROPIC_API_KEY`). */
    client?: Anthropic;
};

export function anthropicGenerate(settings: AnthropicGenerateSettings): LlmGenerator {
    const client = settings.client ?? new Anthropic();
    const { client: _client, ...params } = settings;

    const requestDefaults: LlmGenerator["request"] = {
        model: settings.model,
        max_completion_tokens: settings.max_tokens,
    };
    if (settings.temperature !== undefined) requestDefaults.temperature = settings.temperature;

    const run: LlmGenerate = async (request, ctx) => {
        const { system, messages } = toMessages(request.messages);
        const modelToolList = modelTools(request.tools);

        const stream = client.messages.stream({
            ...params,
            model: request.model,
            max_tokens: request.max_completion_tokens ?? params.max_tokens,
            temperature: request.temperature ?? params.temperature,
            system,
            messages,
            tools: modelToolList.length ? modelToolList : undefined,
        });

        const blocks = new Map<number, BlockInfo>();
        for await (const event of stream) {
            const part = toStreamPart(event, blocks);
            if (part) await ctx.emitDelta?.(part);
        }

        const final = await stream.finalMessage();
        const toolCalls = final.content
            .filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use")
            .map(
                (b): ToolCall => ({
                    id: b.id,
                    type: "function",
                    function: { name: b.name, arguments: JSON.stringify(b.input) },
                }),
            );
        const content = textOf(final.content);
        const finishReason = toFinishReason(final.stop_reason, toolCalls.length > 0);

        await ctx.emitDelta?.({ type: "finish", finishReason });
        return {
            model: final.model,
            content: content || undefined,
            tool_calls: toolCalls,
            finish_reason: finishReason,
            usage: final.usage as unknown as Record<string, unknown>,
        };
    };

    return { request: requestDefaults, handler: "worker", stream: true, run };
}

function modelTools(tools: LlmTool[] | undefined): Anthropic.Tool[] {
    return (tools ?? []).map((t) => ({
        name: t.function.name,
        description: t.function.description || undefined,
        input_schema: t.function.parameters as Anthropic.Tool.InputSchema,
    }));
}

type BlockKind = "text" | "thinking" | "tool";
interface BlockInfo {
    kind: BlockKind;
    id?: string;
}

export function toStreamPart(
    event: Anthropic.RawMessageStreamEvent,
    blocks: Map<number, BlockInfo>,
): StreamPart | null {
    switch (event.type) {
        case "content_block_start": {
            const block = event.content_block;
            if (block.type === "text") {
                blocks.set(event.index, { kind: "text" });
                return { type: "text-start" };
            }
            if (block.type === "thinking" || block.type === "redacted_thinking") {
                blocks.set(event.index, { kind: "thinking" });
                return { type: "reasoning-start" };
            }
            if (block.type === "tool_use") {
                blocks.set(event.index, { kind: "tool", id: block.id });
                return { type: "tool-input-start", toolCallId: block.id, toolName: block.name };
            }
            return null;
        }
        case "content_block_delta": {
            const delta = event.delta;
            if (delta.type === "text_delta") return { type: "text-delta", delta: delta.text };
            if (delta.type === "thinking_delta") return { type: "reasoning-delta", delta: delta.thinking };
            if (delta.type === "input_json_delta") {
                const info = blocks.get(event.index);
                return {
                    type: "tool-input-delta",
                    toolCallId: info?.id ?? String(event.index),
                    inputTextDelta: delta.partial_json,
                };
            }
            return null;
        }
        case "content_block_stop": {
            const info = blocks.get(event.index);
            if (info?.kind === "text") return { type: "text-end" };
            if (info?.kind === "thinking") return { type: "reasoning-end" };
            return null;
        }
        default:
            return null;
    }
}

// Anthropic has no `system`/`tool` role: the system prompt is a top-level param and
// tool results ride inside a `user` message as `tool_result` blocks (merged when
// consecutive).
export function toMessages(messages: Message[]): { system?: string; messages: Anthropic.MessageParam[] } {
    const systemParts: string[] = [];
    const out: Anthropic.MessageParam[] = [];

    for (const m of messages) {
        switch (m.role) {
            case "system": {
                const text = contentText(m.content);
                if (text) systemParts.push(text);
                break;
            }
            case "user":
                out.push({ role: "user", content: contentText(m.content) });
                break;
            case "assistant": {
                const text = contentText(m.content);
                const parts: Anthropic.ContentBlockParam[] = [];
                if (text) parts.push({ type: "text", text });
                for (const tc of m.tool_calls ?? []) {
                    parts.push({
                        type: "tool_use",
                        id: tc.id,
                        name: tc.function.name,
                        input: safeParseJson(tc.function.arguments),
                    });
                }
                out.push({ role: "assistant", content: parts.length ? parts : text });
                break;
            }
            case "tool": {
                const block: Anthropic.ToolResultBlockParam = {
                    type: "tool_result",
                    tool_use_id: m.tool_call_id ?? "",
                    content: contentText(m.content),
                };
                const last = out[out.length - 1];
                if (last && last.role === "user" && Array.isArray(last.content)) {
                    last.content.push(block);
                } else {
                    out.push({ role: "user", content: [block] });
                }
                break;
            }
        }
    }

    return { system: systemParts.length ? systemParts.join("\n\n") : undefined, messages: out };
}

function textOf(content: Anthropic.ContentBlock[]): string {
    return content
        .filter((b): b is Anthropic.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("");
}

function toFinishReason(stopReason: Anthropic.Message["stop_reason"], hasToolCalls: boolean): string {
    if (hasToolCalls) return "tool_calls";
    switch (stopReason) {
        case "tool_use":
            return "tool_calls";
        case "max_tokens":
            return "length";
        case "end_turn":
        case "stop_sequence":
        case "pause_turn":
        case null:
            return "stop";
        default:
            return stopReason ?? "stop";
    }
}

function safeParseJson(text: string): unknown {
    try {
        return JSON.parse(text);
    } catch {
        return {};
    }
}
