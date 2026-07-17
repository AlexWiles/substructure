// A chat agent whose LLM calls run on the worker against OpenRouter, served
// with Hono. Types are generated from schemas/protocol.schema.json (see README).
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import OpenAI from "openai";
import type { ChatCompletionStreamParams } from "openai/lib/ChatCompletionStream";
import type { DecisionRequest, DecisionResponse } from "./protocol.ts";

// OpenRouter speaks Chat Completions, so the OpenAI SDK works as-is.
const openrouter = new OpenAI({
    baseURL: "https://openrouter.ai/api/v1",
    apiKey: process.env.OPENROUTER_API_KEY,
});

const app = new Hono();
app.post("/", async (c) => {
    const { trigger, proposed }: DecisionRequest = await c.req.json();

    // Declare the agent; format makes the engine speak OpenAI on this wire.
    if (trigger.type === "session.start") {
        const decision: DecisionResponse = {
            agent: {
                model: "anthropic/claude-haiku-4.5",
                stream: true,
                handler: "worker",
                format: "openai",
            },
        };
        return c.json(decision);
    }

    // The request is already a Chat Completions body; raw stream chunks and
    // the final completion go straight back.
    if (trigger.type === "llm.execute")
        return streamSSE(c, async (sse) => {
            const stream = openrouter.chat.completions.stream(trigger.request as ChatCompletionStreamParams);
            for await (const chunk of stream)
                await sse.writeSSE({ event: "llm.token.delta", data: JSON.stringify(chunk) });
            await sse.writeSSE({
                event: "decision.result",
                data: JSON.stringify({
                    actions: [{ type: "llm.result", response: await stream.finalChatCompletion() }],
                }),
            });
        });

    // Accept the engine's proposal for every other decision.
    return c.json(proposed);
});

serve({ fetch: app.fetch, port: 4444 }, () => console.log("worker listening on http://localhost:4444"));
