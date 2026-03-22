import { Substructure, defineAgent, withState, withLogging, retry, contentText } from "@substructure.ai/sdk/substructure";
import type { MiddlewareFn } from "@substructure.ai/sdk/substructure";
import type { WorkerAction, Message } from "@substructure.ai/sdk/types";
import { z } from "zod";
import { randomUUID } from "crypto";

// ── Schema ───────────────────────────────────────────────────────────────────
//
const sub = new Substructure({
    db: "data.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const ContactSchema = z.object({
    name: z.string().describe("Full name"),
    email: z.string().describe("Email address"),
    company: z.string().describe("Company or organization"),
    role: z.string().describe("Job title or role"),
});


const RETRY = retry().timeout(120).retries(3).backoff(1, 10);
const SYSTEM_MESSAGE: Message = {
    role: "system",
    content: "You extract contact information from text. When given text, identify the person's name, email, company, and role. Always call extract_contact with the structured data."
}

interface State {
    messages: Message[];
}

function callLlm(state: State): WorkerAction {
    return {
        type: "call_llm",
        llm_client: "openrouter",
        request: {
            model: "arcee-ai/trinity-large-preview:free",
            messages: [SYSTEM_MESSAGE, ...state.messages],
            tools: [
                {
                    function: {
                        name: "extract_contact",
                        description: "Save the extracted contact information. You must call this tool with the structured data.",
                        parameters: z.toJSONSchema(ContactSchema, { target: "draft-2020-12" }),
                    }
                }
            ],
        },
        stream: false,
        retry: RETRY,
    };
}

const structuredOutput: MiddlewareFn<State> = (ctx, next) => {
    const { trigger, state } = ctx;

    switch (trigger.type) {
        case "user_message": {
            state.messages.push(trigger.message);
            return { actions: [callLlm(state)] };
        }

        case "llm_response": {
            state.messages.push(trigger.message);

            if (trigger.message.tool_calls?.length) {
                return {
                    actions: trigger.message.tool_calls.map((tc) => ({
                        type: "call_tool" as const,
                        tool_call_id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                        handler: "worker" as const,
                        retry: RETRY,
                    })),
                };
            }

            const text = contentText(trigger.message.content);
            return { actions: [{ type: "done", artifacts: text ? [{ parts: [{ kind: "text", text }] }] : [] }] };
        }

        case "tool_execute": {
            try {
                const args = JSON.parse(trigger.arguments);
                const contact = ContactSchema.parse(args);
                return {
                    actions: [{
                        type: "done",
                        artifacts: [{ parts: [{ kind: "data" as const, data: contact }] }],
                    }],
                };
            } catch (e: any) {
                return {
                    actions: [{
                        type: "return_tool_error",
                        tool_call_id: trigger.tool_call_id,
                        error: e.message,
                        retryable: false,
                        attempt: trigger.attempt,
                    }],
                };
            }
        }

        default:
            return { actions: [{ type: "done", artifacts: [] }] };
    }
};

const extractor = defineAgent()
    .use(withLogging())
    .use(withState<State>({ messages: [] }))
    .use(structuredOutput)

sub.agent("contact-extractor", extractor);

const input = `Hey! Just met Sarah Chen at the conference. She's a Senior Engineer
at Acme Corp. Shoot her a note at schen@acme.io about the integration work.`;

console.log("Extracting contact from text...\n");

const stream = sub.run(
    "contact-extractor",
    `Extract the contact info from this text:\n\n${input}`,
    { sessionId: randomUUID(), turnId: randomUUID() },
);

for await (const event of stream) {
    if (event.payload.type === "tool.call.completed") {
        const result = JSON.parse(event.payload.result);
        console.log("Extracted:", result);
    } else if (event.payload.type === "llm.call.errored") {
        console.log("LLM ERROR:", event.payload.error);
    }
}

const result = await stream.result;
console.log("\nArtifacts:", JSON.stringify(result.artifacts, null, 2));

await sub.shutdown();
