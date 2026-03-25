import { BackendClient } from "@substructure.ai/sdk";

const backend = new BackendClient({
    url: "http://localhost:8080",
    apiKey: "dev-worker-key",
});

const stream = backend.submit({
    agentId: "web-agent",
    payload: {
        type: "message",
        message: {
            role: "user",
            content: "What is the price of Brent Crude right now?",
        },
    },
    auth: { tenant_id: "default", sub: "test-user" },
});

for await (const event of stream) {
    if (event.payload.type === "message.new") {
        console.log(event.payload.message.role, event.payload.message.content);
    } else if (event.payload.type === "llm.call.errored") {
        console.log("LLM ERROR:", event.payload.error);
    }
}

const result = await stream.result;
console.log("\nTurn result:", result.data);
