// A complete chat agent. No SDK, no dependencies — just Node's http server.
// The engine POSTs a decision request; this returns the next actions.
//
// The worker accepts every decision the engine has a default for (`proposed`)
// and authors only the one that is genuinely its own: the LLM request — the
// agent's identity. Everything else — model replies and model failures — flows
// through the proposed-first line at the top.
//
// Point a local Substructure server at it:
//   substructure serve --dev --provider anthropic --worker-url http://localhost:4444
import { createServer } from "node:http";

function decide({ trigger: t, proposed: proposed } = req) {
    // The engine proposes actions for a default agent tool loop
    if (proposed) return proposed;

    // The client sent the conversation → record it, prompt the model.
    if (t.type === "client.messages") {
        return {
            messages: t.messages,
            actions: [{
                type: "llm.call", stream: true,
                request: {
                    model: "claude-haiku-4-5-20251001",
                    messages: t.messages
                }
            }],
        };
    }

    return { actions: [] };
}

const PORT = 4444;

createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        res.setHeader("content-type", "application/json");
        try {
            res.end(JSON.stringify(decide(JSON.parse(body))));
        } catch (err) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: String(err) }));
        }
    });
}).listen(PORT, () => console.log(`worker listening on http://localhost:${PORT}`));
