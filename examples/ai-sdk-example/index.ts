import Substructure, { type LlmRequest, type LlmResponse, type StreamPart, type ToolCall } from "@substructure.ai/sdk";
import { serve } from "@hono/node-server";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { jsonSchema, streamText, tool, type ModelMessage, type TextStreamPart, type ToolSet } from "ai";

const sub = new Substructure();
const { agent } = sub;

const openrouter = createOpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });

type EmitPart = (part: StreamPart) => Promise<void>;

function toStreamPart<T extends ToolSet>(part: TextStreamPart<T>): StreamPart | null {
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

function toModelMessages(messages: LlmRequest["messages"]): ModelMessage[] {
    return messages.map((m) => ({
        role: m.role,
        content: typeof m.content === "string" ? m.content : (m.content ?? ""),
    })) as ModelMessage[];
}

async function callOpenRouter(request: LlmRequest, ctx: { emitDelta?: EmitPart }): Promise<LlmResponse> {
    const tools = request.tools?.length
        ? Object.fromEntries(
              request.tools.map((t) => [
                  t.function.name,
                  tool({ description: t.function.description, inputSchema: jsonSchema(t.function.parameters as any) }),
              ]),
          )
        : undefined;

    const result = streamText({
        model: openrouter(request.model),
        messages: toModelMessages(request.messages),
        tools,
        temperature: request.temperature,
        maxOutputTokens: request.max_completion_tokens,
        ...(request.reasoning ? { providerOptions: { openrouter: { reasoning: request.reasoning as any } } } : {}),
    });

    for await (const part of result.fullStream) {
        const mapped = toStreamPart(part);
        if (mapped) await ctx.emitDelta?.(mapped);
    }

    const toolCalls = await result.toolCalls;
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
}

const chatAgent = agent({ id: "openrouter-worker" })
    .use(agent.messageHistory("You are a concise assistant."))
    .use(
        agent.llmLoop({
            request: { model: process.env.OPENROUTER_MODEL ?? "openai/gpt-5-nano" },
            stream: true,
            handler: "worker",
            caller: callOpenRouter,
        }),
    );

const worker = sub.worker({ agents: [chatAgent] });
const handler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });

const port = Number(process.env.PORT ?? 3030);
serve({ fetch: handler, port });

console.log(`ai-sdk-example worker listening on http://localhost:${port}`);
