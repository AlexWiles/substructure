import Substructure from "@substructure.ai/sdk";
import { appendFileSync, existsSync } from "fs";
import { randomUUID } from "crypto";

const sub = new Substructure();
const { agent } = sub;

const CSV_PATH = "receipts.csv";

const saveToCSV = agent.tool({
    name: "save_to_csv",
    description: "Append an extracted receipt to the CSV file",
    parameters: {
        type: "object",
        properties: {
            date: { type: "string", description: "Payment date (YYYY-MM-DD)" },
            vendor: { type: "string", description: "Vendor or merchant name" },
            amount: { type: "number", description: "Payment amount" },
            currency: { type: "string", description: "Currency code, e.g. USD" },
        },
        required: ["date", "vendor", "amount", "currency"],
    },
    execute: (args: string) => {
        const { date, vendor, amount, currency } = JSON.parse(args);
        if (!existsSync(CSV_PATH)) {
            appendFileSync(CSV_PATH, "date,vendor,amount,currency\n");
        }
        const row = `${date},"${vendor.replace(/"/g, '""')}",${amount},${currency}\n`;
        appendFileSync(CSV_PATH, row);
        return { saved: true, row: { date, vendor, amount, currency } };
    },
});

const receiptHandler = agent({ id: "receipt-agent" })
    .use(agent.logging())
    .use(agent.state())
    .use(
        agent.systemMessage(`You extract structured data from payment receipts.
Given receipt text, identify the date, vendor, total amount, and currency.
Then call save_to_csv to record it. If the date is unclear, use your best guess.
If the currency is not stated, assume USD.`),
    )
    .use(agent.messageHistory())
    .use(agent.tools([saveToCSV]))
    .use(
        agent.llmLoop({
            request: { model: "deepseek/deepseek-v3.2" },
            llm_client: "openrouter",
        }),
    );

const embedded = await sub.embedded({
    agents: [receiptHandler],
    db: ":memory:",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const receipt = `RECEIPT #4821
Acme Cloud Services
Date: 2026-03-15
Monthly subscription: $49.99
Tax: $4.50
Total: $54.49
Paid via Visa ending 4242`;

const stream = embedded.submit({
    auth: { tenant_id: "default", sub: "example-user" },
    agentId: receiptHandler.agentId,
    payload: {
        type: "message",
        message: { role: "user", content: receipt },
    },
    sessionId: randomUUID(),
    turnId: randomUUID(),
});

for await (const event of stream) {
    console.log(event);
}

const result = await stream.result;
console.log(`Result:\n${result}`);
console.log(`\nDone! Results written to ${CSV_PATH}`);

await embedded.shutdown();
