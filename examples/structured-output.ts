import Substructure from "@substructure.ai/sdk";
import type { MiddlewareFn } from "@substructure.ai/sdk";
import { randomUUID } from "crypto";

const sub = new Substructure();
const { agent } = sub;

const extractContact = agent.tool({
    name: "extract_contact",
    description: "Save the extracted contact information. You must call this tool with the structured data.",
    parameters: {
        type: "object",
        properties: {
            name: { type: "string", description: "Full name" },
            email: { type: "string", description: "Email address" },
            company: { type: "string", description: "Company or organization" },
            role: { type: "string", description: "Job title or role" },
        },
        required: ["name", "email", "company", "role"],
    },
    execute: (args: string) => JSON.parse(args),
});

// Short-circuit: when the tool result comes back, emit done with the
// parsed data instead of sending it back to the LLM for another turn.
const doneOnExtraction: MiddlewareFn<unknown> = (req, next) => {
    if (req.trigger.type === "tool.result" && req.trigger.result.name === extractContact.name) {
        const data = JSON.parse(req.trigger.result.content);
        return { actions: [{ type: "done", data }], state: req.state };
    }
    return next(req);
};

const extractor = agent({ id: "contact-extractor" })
    .use(agent.logging())
    .use(agent.jsonState())
    .use(agent.messageHistory())
    .use(
        agent.systemMessage(
            "You extract contact information from text. When given text, identify the person's name, email, company, and role. Always call extract_contact with the structured data.",
        ),
    )
    .use(agent.tools([extractContact]))
    .use(doneOnExtraction)
    .use(
        agent.llmLoop({
            request: { model: "arcee-ai/trinity-large-preview:free" },
            llm_client: "openrouter",
        }),
    );

const embedded = await sub.embedded({
    agents: [extractor],
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const input = `Hey! Just met Sarah Chen at the conference. She's a Senior Engineer
at Acme Corp. Shoot her a note at schen@acme.io about the integration work.`;

const stream = embedded.submit({
    agentId: extractor.agentId,
    payload: {
        type: "message",
        message: { role: "user", content: `Extract the contact info from this text:\n\n${input}` },
    },
    sessionId: randomUUID(),
    auth: { tenant_id: "default", sub: "example-user" },
    turnId: randomUUID(),
});

for await (const event of stream) {
    if (event.payload.type === "tool.call.completed") {
        const result = JSON.parse(event.payload.result);
        console.log("Extracted:", result);
    } else if (event.payload.type === "llm.call.errored") {
        console.log("LLM ERROR:", event.payload.error);
    }
}

const result = await stream.result;
console.log(result.data);

await embedded.shutdown();
