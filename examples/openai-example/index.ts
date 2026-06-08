import Substructure from "@substructure.ai/sdk";
import { OpenAIAgent } from "@substructure.ai/sdk/adapters/openai";
import { serve } from "@hono/node-server";
import { Agent, tool } from "@openai/agents";
import { z } from "zod";

const assistant = new OpenAIAgent(
    new Agent({
        name: "Assistant",
        model: "gpt-5-nano",
        instructions: "You are a concise assistant.",
        tools: [
            tool({
                name: "getWeather",
                description: "Get the current weather for a city.",
                parameters: z.object({ city: z.string().describe("City name") }),
                execute: async ({ city }) => `It is 22°C and sunny in ${city}.`,
            }),
        ],
    }),
);

const sub = new Substructure();

const chatAgent = sub.agent({ id: "openai-agent" }).use(assistant);

const worker = sub.worker({ agents: [chatAgent] });

const handler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });

const port = Number(process.env.PORT ?? 3030);
serve({ fetch: handler, port });

console.log(`openai-example worker listening on http://localhost:${port}`);
