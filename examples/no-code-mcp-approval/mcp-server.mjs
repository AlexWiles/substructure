// An MCP server standing in for a real one: an issue tracker with a tool that
// reads and a tool that destroys. What matters here is the annotations — they
// are what `approve = "destructive"` reads.
import { createServer } from "node:http";

const PORT = 4500;
const VERSION = "2026-07-28";

const TOOLS = [
    {
        name: "search_issues",
        description: "Search the issue tracker.",
        inputSchema: {
            type: "object",
            properties: { q: { type: "string", description: "Words to match." } },
            required: ["q"]
        },
        annotations: { readOnlyHint: true, destructiveHint: false }
    },
    {
        name: "delete_issue",
        description: "Delete an issue for good.",
        inputSchema: {
            type: "object",
            properties: { id: { type: "string", description: "The issue to delete." } },
            required: ["id"]
        },
        annotations: { readOnlyHint: false, destructiveHint: true }
    }
];

const ISSUES = new Map([
    ["7", "the login page is blank"],
    ["9", "the export button does nothing"]
]);

function call(name, args) {
    if (name === "search_issues") {
        const q = (args.q ?? "").toLowerCase();
        const hits = [...ISSUES]
            .filter(([, title]) => title.toLowerCase().includes(q))
            .map(([id, title]) => `#${id}: ${title}`);
        return hits.length ? hits.join("\n") : "nothing matched";
    }
    if (name === "delete_issue") {
        const id = String(args.id ?? "");
        if (!ISSUES.delete(id)) return `there is no issue #${id}`;
        console.error(`[mcp] deleted #${id}`);
        return `deleted issue #${id}`;
    }
    return `unknown tool: ${name}`;
}

createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        const message = JSON.parse(body || "{}");
        // A notification: nothing to answer.
        if (message.id === undefined) {
            res.writeHead(202);
            return res.end();
        }
        const reply = (result) => {
            res.writeHead(200, { "content-type": "application/json" });
            res.end(
                JSON.stringify({
                    jsonrpc: "2.0",
                    id: message.id,
                    result: { ...result, resultType: "complete" }
                })
            );
        };
        switch (message.method) {
            // The client asks what the server speaks before it says anything.
            case "server/discover":
                return reply({
                    supportedVersions: [VERSION],
                    capabilities: { tools: {} },
                    instructions: "An issue tracker.",
                    ttlMs: 0,
                    cacheScope: "private"
                });
            case "tools/list":
                return reply({ tools: TOOLS, ttlMs: 0, cacheScope: "private" });
            case "tools/call":
                return reply({
                    content: [
                        { type: "text", text: call(message.params.name, message.params.arguments ?? {}) }
                    ]
                });
            default:
                res.writeHead(404, { "content-type": "application/json" });
                return res.end(
                    JSON.stringify({
                        jsonrpc: "2.0",
                        id: message.id,
                        error: { code: -32601, message: "no such method", data: message.method }
                    })
                );
        }
    });
}).listen(PORT, "127.0.0.1", () => console.error(`[mcp] listening on ${PORT}`));
