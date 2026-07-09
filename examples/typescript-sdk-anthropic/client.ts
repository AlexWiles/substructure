import Substructure, { isTokenDelta } from "@substructure.ai/sdk";

const sub = new Substructure();
const client = sub.backend.client({
    url: process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000",
    apiKey: process.env.SUBSTRUCTURE_API_KEY ?? "dev-worker-key",
});

const scope = await client.startTurn({
    agentId: "anthropic-agent",
    payload: {
        type: "client.message",
        message: {
            role: "user",
            content: "What's the weather in Paris? write a story about it in the style of tolstoy.",
        },
    },
    identity: { id: "demo" },
});

for await (const event of client.stream(scope, { tokens: true })) {
    if (isTokenDelta(event)) {
        if (event.text) process.stdout.write(event.text);
        continue;
    }
}
