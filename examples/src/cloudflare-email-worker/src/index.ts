import Substructure from "@substructure.ai/sdk";
import { EmailMessage } from "cloudflare:email";
import { createMimeMessage } from "mimetext";
import PostalMime from "postal-mime";
import { AgentState, durableObjectState } from "./state";
import { Database } from "./database";
import { inReplyToOf, sessionIdForThread, threadRoot } from "./email";

export { AgentState, Database };

interface WorkerEnv extends Env {
    AGENT_STATE: DurableObjectNamespace<AgentState>;
    DATABASE: DurableObjectNamespace<Database>;
    EMAIL: SendEmail;
    SUBSTRUCTURE_URL: string;
    SUBSTRUCTURE_API_KEY: string;
    SIGNING_SECRET?: string;
}

let workerEnv: WorkerEnv;

const WHITELIST = new Set<string>(["alice@example.com", "bob@example.com"]);

const db = () => workerEnv.DATABASE.get(workerEnv.DATABASE.idFromName("global"));

// ── Agent ──────────────────────────────────────────────────────────────────

const sub = new Substructure();
const { agent } = sub;

const recordExpense = agent.tool({
    name: "record_expense",
    description:
        "Record one expense extracted from the current email. Amount is in " +
        "dollars; stored as integer cents. Call once per line item — the " +
        "same email can have multiple expenses.",
    parameters: {
        type: "object",
        properties: {
            email_id: { type: "number", description: "Row id of the current email" },
            amount: { type: "number", description: "Dollars, e.g. 12.50" },
            description: { type: "string", description: "What was bought" },
        },
        required: ["email_id", "amount", "description"],
    },
    execute: async (args) => {
        const { email_id, amount, description } = JSON.parse(args) as {
            email_id: number;
            amount: number;
            description: string;
        };
        await db().recordExpense({
            emailId: email_id,
            amountCents: Math.round(amount * 100),
            description,
        });
        return { ok: true, email_id, amount, description };
    },
});

const sendReplyTool = agent.tool({
    name: "send_reply",
    description:
        "Send a plain-text reply to the sender of the given email. Threads " +
        "correctly via In-Reply-To. Call once with the body you want " +
        "delivered. Omit if no reply is needed.",
    parameters: {
        type: "object",
        properties: {
            emailId: {
                type: "number",
                description: "Row id of the email to reply to",
            },
            body: { type: "string", description: "Plain-text reply body" },
        },
        required: ["email_id", "body"],
    },
    execute: async (args) => {
        const { emailId, body } = JSON.parse(args) as {
            emailId: number;
            body: string;
        };

        const email = await db().getEmail(emailId);
        if (!email) throw new Error(`email_id ${emailId} not found`);

        const mime = createMimeMessage();
        if (email.message_id) mime.setHeader("In-Reply-To", `<${email.message_id}>`);
        mime.setSender({ name: "Inbox agent", addr: email.to_addr });
        mime.setRecipient(email.from_addr);
        mime.setSubject(`Re: ${email.subject}`);
        mime.addMessage({ contentType: "text/plain", data: body });

        const message = new EmailMessage(email.to_addr, email.from_addr, mime.asRaw());
        await workerEnv.EMAIL.send(message);

        return "message sent";
    },
});

const querySql = agent.tool({
    name: "query_sql",
    description:
        "Run any SQL statement against the database and return the rows.\n" +
        "Schema:\n" +
        "  emails(id INTEGER PRIMARY KEY, message_id TEXT UNIQUE, thread_id TEXT,\n" +
        "         in_reply_to TEXT, from_addr TEXT, to_addr TEXT, subject TEXT,\n" +
        "         body TEXT, received_at INTEGER)\n" +
        "  expenses(id INTEGER PRIMARY KEY,\n" +
        "           email_id INTEGER REFERENCES emails(id),\n" +
        "           amount_cents INTEGER, description TEXT, created_at INTEGER)\n" +
        "Amounts are integer cents. Timestamps are Unix epoch in milliseconds. " +
        "Use ? placeholders for parameters. Prefer record_expense for " +
        "inserting new expense rows.",
    parameters: {
        type: "object",
        properties: {
            sql: { type: "string", description: "Any SQL statement" },
            params: {
                type: "array",
                description: "Positional parameters for ? placeholders",
                items: {},
            },
        },
        required: ["sql"],
    },
    execute: async (args) => {
        const { sql, params = [] } = JSON.parse(args) as {
            sql: string;
            params?: SqlStorageValue[];
        };
        const rows = await db().query(sql, params);
        return { rows, count: rows.length };
    },
});

const SYSTEM_PROMPT =
    "You are a personal-finance assistant that processes inbound emails. " +
    "Log any expenses mentioned, answer questions about past spending, and " +
    "reply when there's something useful to say back. Each user message " +
    "starts with `email_id`, `now_ms`, and `now` header lines for context.";

const emailAgent = agent({ id: "email-agent" })
    .use(agent.logging())
    .use(durableObjectState(() => workerEnv.AGENT_STATE))
    .use(agent.systemMessage(SYSTEM_PROMPT))
    .use(agent.messageHistory())
    .use(agent.tools([recordExpense, querySql, sendReplyTool]))
    .use(
        agent.llmLoop({
            request: { model: "anthropic/claude-sonnet-4-6" },
            llm_client: "openrouter",
        }),
    );

const worker = sub.worker({ agents: [emailAgent] });

export default {
    async email(message: ForwardableEmailMessage, env: WorkerEnv, ctx: ExecutionContext): Promise<void> {
        workerEnv = env;
        if (!WHITELIST.has(message.from.toLowerCase())) {
            message.setReject("Sender not allowed");
            return;
        }

        const parsed = await PostalMime.parse(message.raw);
        const subject = parsed.subject ?? "(no subject)";
        const body = (parsed.text ?? "").slice(0, 4000);
        const emailId = await db().recordEmail({
            messageId: message.headers.get("Message-ID"),
            threadId: threadRoot(message.headers),
            inReplyTo: inReplyToOf(message.headers),
            from: message.from,
            to: message.to,
            subject,
            body,
        });
        const now = Date.now();
        const userContent =
            `email_id: ${emailId}\n` +
            `now_ms: ${now}\n` +
            `now: ${new Date(now).toISOString()}\n` +
            `From: ${message.from}\nSubject: ${subject}\n\n${body}`;

        const client = sub.backend.client({
            url: env.SUBSTRUCTURE_URL,
            apiKey: env.SUBSTRUCTURE_API_KEY,
        });

        const sessionId = sessionIdForThread(message.headers);

        await client.startTurn({
            agentId: emailAgent.agentId,
            sessionId,
            payload: { type: "message", message: { role: "user", content: userContent } },
            identity: { id: message.from },
        });
    },

    async fetch(request: Request, env: WorkerEnv): Promise<Response> {
        workerEnv = env;
        const handler = worker.fetchHandler({ signingSecret: env.SIGNING_SECRET });
        try {
            return await handler(request);
        } catch (err) {
            const m = err instanceof Error ? err.message : "Internal error";
            return Response.json({ error: m }, { status: 500 });
        }
    },
} satisfies ExportedHandler<WorkerEnv>;
