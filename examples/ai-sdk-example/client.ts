import Substructure, { isTokenDelta } from "@substructure.ai/sdk";

const sub = new Substructure();
const client = sub.backend.client({
    url: process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000",
    apiKey: process.env.SUBSTRUCTURE_API_KEY ?? "dev-worker-key",
});

const scope = await client.startTurn({
    agentId: "openrouter-worker",
    payload: {
        type: "message",
        message: { role: "user", content: "Write a three sentence explanation of worker-side LLM streaming." },
    },
    identity: { id: "demo" },
});

for await (const event of client.stream(scope, { tokens: true })) {
    if (isTokenDelta(event)) {
        if (event.text) process.stdout.write(event.text);
        continue;
    }

    if (event.payload.type === "turn.completed") {
        process.stdout.write("\n");
        console.log(event.payload.data);
    }
}
