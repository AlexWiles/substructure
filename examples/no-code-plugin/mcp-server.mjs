// The file server the plugin's `mcp.json` names. A stand-in for wherever your
// runbooks really live: it serves the `runbooks/` directory beside this file,
// so the example runs with nothing to sign up for.
import { createServer } from "node:http";
import { readdirSync, readFileSync } from "node:fs";
import { resolve, sep } from "node:path";

const PORT = 4600;
const VERSION = "2026-07-28";
const ROOT = resolve(import.meta.dirname, "runbooks");

const TOOLS = [
    {
        name: "list_dir",
        description: "List the runbook files.",
        inputSchema: { type: "object", properties: {} },
        annotations: { readOnlyHint: true, destructiveHint: false }
    },
    {
        name: "read_file",
        description: "Read one runbook by the path `list_dir` gave.",
        inputSchema: {
            type: "object",
            properties: { path: { type: "string", description: "The file to read." } },
            required: ["path"]
        },
        annotations: { readOnlyHint: true, destructiveHint: false }
    }
];

function call(name, args) {
    if (name === "list_dir") {
        return readdirSync(ROOT).sort().join("\n");
    }
    if (name === "read_file") {
        const path = resolve(ROOT, String(args.path ?? ""));
        if (!path.startsWith(ROOT + sep)) return "that path is outside the runbooks";
        try {
            return readFileSync(path, "utf8");
        } catch {
            return `there is no ${args.path}; list_dir shows what exists`;
        }
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
                    instructions: "The on-call runbooks.",
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
