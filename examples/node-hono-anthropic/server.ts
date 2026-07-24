// A chat agent whose LLM calls run on the worker, served with Hono. Types are
// generated from schemas/protocol.schema.json (see README).
import Anthropic from "@anthropic-ai/sdk";
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import type { DecisionRequest, DecisionResponse } from "./protocol.ts";

const anthropic = new Anthropic();

const app = new Hono();
app.post("/", async (c) => {
    const { trigger, proposed }: DecisionRequest = await c.req.json();

    // Declare the agent; format makes the engine speak Anthropic on this wire.
    if (trigger.type === "session.start") {
        const decision: DecisionResponse = {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true,
                handler: "worker",
                format: "anthropic",
                system: "Reply using slack compatible mrkdwn",
            },
        };
        return c.json(decision);
    }

    // The request is already a Messages API body; raw stream events and the
    // final message go straight back.
    if (trigger.type === "llm.execute")
        return streamSSE(c, async (sse) => {
            const stream = anthropic.messages.stream(trigger.request as Anthropic.MessageStreamParams);
            for await (const event of stream)
                await sse.writeSSE({ event: "llm.token.delta", data: JSON.stringify(event) });
            await sse.writeSSE({
                event: "decision.result",
                data: JSON.stringify({
                    actions: [{ type: "llm.result", response: await stream.finalMessage() }],
                }),
            });
        });

    // Accept the engine's proposal for every other decision.
    return c.json(proposed);
});

serve({ fetch: app.fetch, port: 4444 }, () => console.log("worker listening on http://localhost:4444"));
