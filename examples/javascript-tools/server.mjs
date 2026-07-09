// A complete chat agent with a tool. No SDK, no dependencies — Node's http server.
// The engine POSTs a decision request; this returns the next actions.
//
// The worker accepts every decision the engine has a default for (`proposed`)
// and authors only the two that are genuinely its own: the LLM request (the
// agent's identity) and the tool execution. Everything else — tool results,
// model replies, model failures, even broken or hallucinated tool calls —
// flows through the proposed-first line at the top.
//
// Point a local Substructure server at it:
//   substructure serve --dev --provider anthropic --worker-url http://localhost:4444
import { createServer } from "node:http";

const PORT = 4444;

const TOOLS = [
    {
        info: {
            name: "get_current_time",
            description:
                "Get the current UTC date and time. " +
                "Call this whenever the user asks what time or date it is.",
        },
        exec: () => new Date().toISOString(),
    },
];


function decide(req) {
    // The engine proposes actions for a default agent tool loop
    if (req.proposed) return req.proposed;

    const t = req.trigger;

    // The client sent the conversation → record it, prompt the model.
    if (t.type === "client.messages") {
        return {
            messages: t.messages,
            actions: [{
                type: "llm.call", stream: true,
                request: {
                    model: "claude-haiku-4-5-20251001", messages: t.messages,
                    tools: TOOLS.map((tool) => tool.info)
                }
            }],
        };
    }

    // A declared tool call with valid arguments → run it, answer.
    // Invalid tool calls will have a proposal from the engine already
    if (t.type === "tool.execute") {
        const tool = TOOLS.find((tool) => tool.info.name === t.name);
        return { actions: [{ type: "tool.result", result: tool.exec(t.input.value) }] };
    }

    return { actions: [] };
}

createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify(decide(JSON.parse(body))));
    });
}).listen(PORT, () => console.log(`worker listening on http://localhost:${PORT}`));
