// Submit a turn to the agent hosted by this Vercel function.
//
// Run after the Substructure backend has been pointed at the deployment URL.
// Set SUBSTRUCTURE_URL and SUBSTRUCTURE_API_KEY from the dashboard.

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
