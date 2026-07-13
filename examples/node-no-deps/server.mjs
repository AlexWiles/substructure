// A complete chat agent served with Node's built-in http server. No dependencies.
import { createServer } from "node:http";

function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        // The engine will use this agent config to generate proposed actions.
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true
            }
        };
    }

    // Accept the engine's proposal for every other decision.
    return proposed;
}

const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        const decision = decide(JSON.parse(body));
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(decision ?? null));
    });
});

server.listen(4444, () =>
    console.log("worker listening on http://localhost:4444"));
