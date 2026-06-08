import Substructure from "@substructure.ai/sdk";
import { ToolLoopAgent } from "@substructure.ai/sdk/adapters/ai";
import { serve } from "@hono/node-server";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { tool } from "ai";
import { z } from "zod";

const openrouter = createOpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });

const assistant = new ToolLoopAgent({
    model: openrouter("openai/gpt-5-nano"),
    instructions: "You are a concise assistant.",
    tools: {
        getWeather: tool({
            description: "Get the current weather for a city.",
            inputSchema: z.object({ city: z.string().describe("City name") }),
            execute: async ({ city }) => `It is 22°C and sunny in ${city}.`,
        }),
    },
});

const sub = new Substructure();

const chatAgent = sub.agent({ id: "ai-sdk-agent" }).use(sub.agent.logging()).use(assistant);

const worker = sub.worker({ agents: [chatAgent] });

const handler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });

const port = Number(process.env.PORT ?? 3030);

serve({ fetch: handler, port });

console.log(`ai-sdk-example worker listening on http://localhost:${port}`);
