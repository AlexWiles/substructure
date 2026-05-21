// Submit a turn to the agent hosted by the Hono server.
//
// Run after a Substructure backend has been started with
// `--worker-url http://localhost:3000/agent`.

import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const client = sub.backend.client({
    url: process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000",
    apiKey: process.env.SUBSTRUCTURE_API_KEY ?? "dev-worker-key",
});

const scope = await client.startTurn({
    agentId: "weather",
    payload: {
        type: "message",
        message: { role: "user", content: "What's the weather in San Francisco?" },
    },
    identity: { id: "demo" },
});

const { data } = await client.turnResult(scope);
console.log(data);
