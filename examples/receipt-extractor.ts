import {
    Substructure,
    defineAgent,
    withState,
    withLogging,
    tool,
    withConversation,
    withSystemMessage,
    withTools,
    withCallLLM,
} from "@substructure.ai/sdk/substructure";
import type { Message } from "@substructure.ai/sdk/types";
import { z } from "zod";
import { appendFileSync, existsSync } from "fs";
import { randomUUID } from "crypto";

const CSV_PATH = "receipts.csv";

const receiptRetry = {
    timeout_secs: 120,
    max_retries: 3,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

const saveToCSV = tool({
    description: "Append an extracted receipt to the CSV file",
    parameters: z.object({
        date: z.string().describe("Payment date (YYYY-MM-DD)"),
        vendor: z.string().describe("Vendor or merchant name"),
        amount: z.number().describe("Payment amount"),
        currency: z.string().describe("Currency code, e.g. USD"),
    }),
    execute: ({ date, vendor, amount, currency }) => {
        if (!existsSync(CSV_PATH)) {
            appendFileSync(CSV_PATH, "date,vendor,amount,currency\n");
        }
        const row = `${date},"${vendor.replace(/"/g, '""')}",${amount},${currency}\n`;
        appendFileSync(CSV_PATH, row);
        return { saved: true, row: { date, vendor, amount, currency } };
    },
});

const RECEIPT_AGENT_ID = "receipt-extractor";

type State = {
    messages: Message[];
};

const messagesAdapter = {
    getMessages: (state: State) => state.messages,
    setMessages: (state: State, messages: Message[]) => {
        state.messages = messages;
    },
};
const SYSTEM_PROMPT = `You extract structured data from payment receipts.
Given receipt text, identify the date, vendor, total amount, and currency.
Then call save_to_csv to record it. If the date is unclear, use your best guess.
If the currency is not stated, assume USD.`

const receiptHandler = defineAgent(RECEIPT_AGENT_ID)
    .use(withLogging())
    .use(withState<State>({ messages: [] }))
    .use(withSystemMessage<State>(SYSTEM_PROMPT))
    .use(withConversation<State>(messagesAdapter))
    .use(withTools<State>({ saveToCSV }))
    .use(withCallLLM<State>((state) => ({
        request: {
            model: "arcee-ai/trinity-large-preview:free",
        },
        llm_client: "openrouter",
        retry: receiptRetry,
    })));

const sub = new Substructure({
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

sub.agent(receiptHandler);

// Sample receipts to process
const receipts = [
    `RECEIPT #4821
     Acme Cloud Services
     Date: 2026-03-15
     Monthly subscription: $49.99
     Tax: $4.50
     Total: $54.49
     Paid via Visa ending 4242`,

    `PaymentConfirmation
     Vendor: Tokyo Ramen House
     March 18, 2026
     2x Tonkotsu Ramen  $17.00
     1x Gyoza            $8.50
     Tip                  $5.00
     Total Charged: $30.50`,

    `INVOICE PAID
     From: Figma Inc.
     Professional Plan - Annual
     Amount: €348.00
     Date: 2026-03-01
     Status: PAID`,
];

for (const [i, receipt] of receipts.entries()) {
    console.log(`\n--- Processing receipt ${i + 1} ---`);

    const stream = sub.run(
        RECEIPT_AGENT_ID,
        `Extract the payment details from this receipt:\n\n${receipt}`,
        { sessionId: randomUUID(), turnId: randomUUID() },
    );

    for await (const event of stream) {
        if (event.payload.type === "tool.call.completed") {
            console.log("Saved:", event.payload.result);
        } else if (event.payload.type === "llm.call.errored") {
            console.log("LLM ERROR:", event.payload.error);
        }
    }
}

console.log(`\nDone! Results written to ${CSV_PATH}`);
await sub.shutdown();
