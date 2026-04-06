import { defineAgent, state, logging } from "@substructure.ai/sdk/agent";
import type { MiddlewareFn } from "@substructure.ai/sdk/agent";
import { Substructure, contentText } from "@substructure.ai/sdk";
import type { ClientIdentity, WorkerAction, Message } from "@substructure.ai/sdk";
import { EmbeddedRuntime } from "@substructure.ai/runtime";
import { z } from "zod";
import { randomUUID } from "crypto";

// ── Schema ───────────────────────────────────────────────────────────────────
//
const runtime = new EmbeddedRuntime({
    db: "data.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});
const sub = new Substructure({ runtime });

const auth: ClientIdentity = {
    tenant_id: "default",
    sub: "example-user",
};

const ContactSchema = z.object({
    name: z.string().describe("Full name"),
    email: z.string().describe("Email address"),
    company: z.string().describe("Company or organization"),
    role: z.string().describe("Job title or role"),
});


const RETRY = {
    timeout_secs: 120,
    max_retries: 3,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

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

const withMessageHistory = (): MiddlewareFn<unknown> => (req, next) => {
    const { trigger } = req;
    const state = req.state as State;
    state.messages ??= [];

    switch (trigger.type) {
        case "user_message": {
            state.messages.push(trigger.message);
            break;
        }
        case "llm_response": {
            state.messages.push(trigger.message);
            break;
        }
        case "tool_result": {
            state.messages.push({
                role: "tool",
                content: trigger.result.content,
                tool_call_id: trigger.result.tool_call_id,
                name: trigger.result.name,
            });
            break;
        }
    }

    return next({ ...req, state });
};

const withStructuredOutputFlow = (): MiddlewareFn<unknown> => (req, next) => {
    const { trigger } = req;
    const state = req.state as State;

    switch (trigger.type) {
        case "user_message":
            return { actions: [callLlm(state)], state };

        case "llm_response": {
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
                    state,
                };
            }

            const text = contentText(trigger.message.content);
            return { actions: [{ type: "done", data: text ?? null }], state };
        }

        case "tool_execute": {
            try {
                const args = JSON.parse(trigger.arguments);
                const contact = ContactSchema.parse(args);
                return {
                    actions: [{
                        type: "done",
                        data: contact,
                    }],
                    state,
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
                    state,
                };
            }
        }

        default:
            return next({ ...req, state });
    }
};

const extractor = defineAgent("contact-extractor")
    .use(logging())
    .use(state())
    .use(withMessageHistory())
    .use(withStructuredOutputFlow())

sub.agent(extractor);

const input = `Hey! Just met Sarah Chen at the conference. She's a Senior Engineer
at Acme Corp. Shoot her a note at schen@acme.io about the integration work.`;

console.log("Extracting contact from text...\n");

const stream = sub.submit({
    agentId: "contact-extractor",
    payload: {
        type: "message",
        message: {
            role: "user",
            content: `Extract the contact info from this text:\n\n${input}`,
        },
    },
    sessionId: randomUUID(),
    auth,
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

await sub.shutdown();
