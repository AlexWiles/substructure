// OpenAI adapter: `openaiGenerate` is an `Llm` backed by the Responses API;
// `openaiAgent` builds a `toolLoop` from an `@openai/agents` Agent.

import type { ModelSettings, ModelSettingsToolChoice, Tool } from "@openai/agents";
import { Agent, RunContext } from "@openai/agents";
import OpenAI from "openai";
import { toolLoop } from "../agent";
import type { Llm, LlmGenerate, Agent as SdkAgent, ToolDef } from "../core";
import type { LlmParams, LlmTokenDeltaInput, LlmTool, MessageInput, ToolCall } from "../types";
import { contentText } from "../types";

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type ResponseStreamEvent = OpenAI.Responses.ResponseStreamEvent;
type ResponseOutputItem = OpenAI.Responses.ResponseOutputItem;

// Responses create params minus what the loop supplies (input/tools/stream);
// `model` is re-required (optional upstream).
export type OpenAIGenerateSettings = Omit<
    OpenAI.Responses.ResponseCreateParamsStreaming,
    "input" | "tools" | "stream" | "model"
> & {
    model: NonNullable<OpenAI.Responses.ResponseCreateParams["model"]>;
    /** Defaults to `new OpenAI()` (reads `OPENAI_API_KEY`). */
    client?: OpenAI;
};

export function openaiGenerate(settings: OpenAIGenerateSettings): Llm {
    const client = settings.client ?? new OpenAI();
    const { client: _client, ...params } = settings;

    const request: LlmParams = { model: String(settings.model) };
    if (settings.temperature != null) request.temperature = settings.temperature;
    if (settings.max_output_tokens != null) request.max_completion_tokens = settings.max_output_tokens;

    const run: LlmGenerate = async (req, ctx) => {
        const modelToolList = modelTools(req.tools);
        const stream = await client.responses.create({
            ...params,
            model: req.model,
            input: toResponsesInput(req.messages),
            tools: modelToolList.length ? modelToolList : undefined,
            stream: true,
            temperature: req.temperature ?? params.temperature,
            max_output_tokens: req.max_completion_tokens ?? params.max_output_tokens,
        });

        const callIdByItem = new Map<string, string>();
        let final: OpenAI.Responses.Response | undefined;
        for await (const event of stream) {
            const delta = toDelta(event, callIdByItem);
            if (delta) await ctx.emitDelta?.(delta);
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
        await ctx.emitDelta?.({ finish_reason: toolCalls.length ? "tool_calls" : "stop" });
        return {
            model: req.model,
            content: content || undefined,
            tool_calls: toolCalls,
            finish_reason: toolCalls.length ? "tool_calls" : "stop",
            usage: final?.usage as Record<string, unknown> | undefined,
        };
    };

    return { ...request, handler: "worker", stream: true, run };
}

function modelTools(toolList: LlmTool[] | undefined): OpenAI.Responses.Tool[] {
    return (toolList ?? []).map(
        (t): OpenAI.Responses.Tool => ({
            type: "function",
            name: t.name,
            description: t.description || undefined,
            parameters: (t.input ?? { type: "object", properties: {} }) as Record<string, unknown>,
            strict: null,
        }),
    );
}

export interface OpenAIAgentSettings {
    model: string;
    instructions?: string;
    tools?: Tool[];
    modelSettings?: ModelSettings;
    /** Defaults to `new OpenAI()` (reads `OPENAI_API_KEY`). */
    client?: OpenAI;
    /** Run context passed to each tool's `execute` (the Agents SDK `RunContext`). */
    context?: unknown;
}

interface ResolvedSettings {
    client: OpenAI;
    model: string;
    instructions?: string | (() => string | Promise<string>);
    tools: Tool[];
    modelSettings?: ModelSettings;
    context?: unknown;
}

/** Build a `toolLoop` from an `@openai/agents` Agent (or `OpenAIAgentSettings`),
 *  using `openaiGenerate` as the model and the Agent's tools run by the worker. */
export function openaiAgent(
    input: OpenAIAgentSettings | Agent,
    options?: { client?: OpenAI; context?: unknown },
): SdkAgent {
    const settings = input instanceof Agent ? fromAgent(input, options) : resolveSettings(input);
    return toolLoop({
        llm: openaiGenerate(toGenerateSettings(settings)),
        instructions: settings.instructions,
        tools: openAITools(settings.tools, settings.context),
    });
}

function resolveSettings(settings: OpenAIAgentSettings): ResolvedSettings {
    return {
        client: settings.client ?? new OpenAI(),
        model: settings.model,
        instructions: settings.instructions,
        tools: settings.tools ?? [],
        modelSettings: settings.modelSettings,
        context: settings.context,
    };
}

function fromAgent(oaiAgent: Agent, options?: { client?: OpenAI; context?: unknown }): ResolvedSettings {
    if (typeof oaiAgent.model !== "string") {
        throw new Error(
            "openaiAgent executes via the OpenAI Responses API and needs a model id string. Pass `model` as a string on the Agent, or use `openaiAgent({ model, ... })`.",
        );
    }
    const context = options?.context;
    const instructions =
        typeof oaiAgent.instructions === "function"
            ? () =>
                  (oaiAgent.instructions as (rc: RunContext, a: Agent) => string | Promise<string>)(
                      new RunContext(context),
                      oaiAgent,
                  )
            : oaiAgent.instructions;
    return {
        client: options?.client ?? new OpenAI(),
        model: oaiAgent.model,
        instructions,
        tools: oaiAgent.tools,
        modelSettings: oaiAgent.modelSettings,
        context,
    };
}

// Map the Agents SDK `ModelSettings` onto Responses create params; `providerData`
// passes through to the request unchanged.
function toGenerateSettings(s: ResolvedSettings): OpenAIGenerateSettings {
    const ms = s.modelSettings;
    const out: OpenAIGenerateSettings = { model: s.model, client: s.client };
    if (ms?.temperature != null) out.temperature = ms.temperature;
    if (ms?.maxTokens != null) out.max_output_tokens = ms.maxTokens;
    if (ms?.topP != null) out.top_p = ms.topP;
    if (ms?.parallelToolCalls != null) out.parallel_tool_calls = ms.parallelToolCalls;
    if (ms?.store != null) out.store = ms.store;
    if (ms?.truncation != null) out.truncation = ms.truncation;
    if (ms?.toolChoice != null) out.tool_choice = toResponsesToolChoice(ms.toolChoice);
    return Object.assign(out, ms?.providerData);
}

// Substructure-side execution of `@openai/agents` function tools as `ToolDef`s.
export function openAITools(toolset: Tool[], context?: unknown): ToolDef[] {
    return toolset.flatMap((t): ToolDef[] => {
        if (t.type !== "function") return [];
        return [
            {
                name: t.name,
                description: t.description ?? "",
                input: t.parameters,
                execute: async (args) => {
                    const result = await t.invoke(new RunContext(context), args || "{}");
                    return typeof result === "string" ? result : JSON.stringify(result);
                },
            } satisfies ToolDef,
        ];
    });
}

export function toDelta(event: ResponseStreamEvent, callIdByItem: Map<string, string>): LlmTokenDeltaInput | null {
    switch (event.type) {
        case "response.output_text.delta":
            return { text: event.delta };
        case "response.reasoning_summary_text.delta":
            return { reasoning: event.delta };
        case "response.output_item.added":
            if (event.item.type === "function_call") {
                callIdByItem.set(event.item.id ?? event.item.call_id, event.item.call_id);
                return { tool_calls: [{ id: event.item.call_id, name: event.item.name }] };
            }
            return null;
        case "response.function_call_arguments.delta":
            return { tool_calls: [{ id: callIdByItem.get(event.item_id) ?? event.item_id, arguments: event.delta }] };
        default:
            return null;
    }
}

export function toResponsesInput(messages: MessageInput[]): ResponseInputItem[] {
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

function outputText(output: ResponseOutputItem[]): string {
    return output
        .filter((i): i is Extract<ResponseOutputItem, { type: "message" }> => i.type === "message")
        .flatMap((i) => i.content)
        .filter((c): c is OpenAI.Responses.ResponseOutputText => c.type === "output_text")
        .map((c) => c.text)
        .join("");
}

const TOOL_CHOICE_OPTIONS = ["auto", "required", "none"] as const;

function toResponsesToolChoice(choice: ModelSettingsToolChoice): OpenAIGenerateSettings["tool_choice"] {
    const option = TOOL_CHOICE_OPTIONS.find((o) => o === choice);
    return option ?? { type: "function", name: choice };
}
