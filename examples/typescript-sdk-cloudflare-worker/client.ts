// Submit a turn to the agent hosted by this Worker.
//
// Run after the Substructure backend has been pointed at the Worker URL.
// Set and SUBSTRUCTURE_API_KEY from the dashboard.

import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();

const client = sub.backend.client({
    url: "https://api.substructure.ai",
    apiKey: process.env.SUBSTRUCTURE_API_KEY!,
});

const scope = await client.startTurn({
    agentId: "todo",
    payload: {
        type: "message",
        message: { role: "user", content: "Add 'buy groceries' and list my todos" },
    },
    identity: { id: "demo" },
});

const { data } = await client.turnResult(scope);
console.log(data);
