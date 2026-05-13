import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const backend = sub.backend.client({
    url: "http://localhost:8080",
    apiKey: "dev-worker-key",
});

const stream = backend.submitAndListen({
    agentId: "web-agent",
    payload: {
        type: "message",
        message: {
            role: "user",
            content: "What are the headlines today?",
        },
    },
    identity: { id: "test-user" },
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
