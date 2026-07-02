// A minimal agent. The Substructure runtime runs in-process and persists the
// session + event log to a SQLite file (`agent.db`); the OpenAI generator drives
// each step and Substructure runs the tool loop. Add tools and turns from here.

import { agent, tool, toolLoop } from "@substructure.ai/sdk";
import { openaiGenerate } from "@substructure.ai/sdk/adapters/openai";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";

const getCurrentTime = tool({
    name: "get_current_time",
    description: "Get the current date and time.",
    parameters: { type: "object", properties: {} },
    execute: () => new Date().toString(),
});

const SYSTEM_PROMPT = "You are a concise assistant. Use tools when relevant.";

const assistant = agent({
    name: "assistant",
    decide: toolLoop({
        llm: openaiGenerate({ model: "gpt-5-nano" }),
        instructions: SYSTEM_PROMPT,
        tools: [getCurrentTime],
    }),
});

const embedded = await SubstructureEmbedded.create({
    agents: [assistant],
    db: "agent.db",
});

const scope = await embedded.startTurn({
    agentId: "assistant",
    payload: {
        type: "message",
        message: { role: "user", content: "What time is it right now?" },
    },
    identity: { tenant_id: "default", id: "demo" },
});

const { data } = await embedded.turnResult(scope);
console.log(data);

await embedded.shutdown();
