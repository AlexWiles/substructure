// A chat agent with tools, served with Node's built-in http server. No dependencies.
import { createServer } from "node:http";

const tools = [
    {
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: () => new Date().toISOString()
    },
    {
        name: "get_current_time_zone",
        description: "Get the user's current timezone",
        exec: () => Intl.DateTimeFormat().resolvedOptions().timeZone
    }
];

function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        // The engine will use this agent config to generate proposed actions.
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true,
                tools: tools.map(({ name, description }) => ({ name, description }))
            }
        };
    }

    // Run our tool when the model calls it.
    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        return { actions: [{ type: "tool.result", result: tool.exec() }] };
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
