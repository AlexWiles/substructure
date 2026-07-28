// A small MCP server, so the example runs with no account anywhere.
//
// It uses the official MCP SDK rather than hand-writing the responses: the
// point of the example is to show the engine talking to a real MCP server.
// Replace it with Sentry or GitHub by editing substructure.toml — nothing in
// the worker changes.
import { randomUUID } from "node:crypto";
import express from "express";
import { z } from "zod";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";

const issues = [
    { id: "PROJ-1", status: "open", title: "Checkout times out on slow networks" },
    { id: "PROJ-2", status: "open", title: "Avatar upload rejects PNGs over 2MB" },
    { id: "PROJ-3", status: "closed", title: "Search returns archived posts" }
];

function buildServer() {
    const server = new McpServer({ name: "issues-demo", version: "1.0.0" });

    // readOnlyHint is what `read_only = true` in substructure.toml tests for.
    // A tool with no annotations fails that filter rather than passing silently.
    server.registerTool(
        "search_issues",
        {
            description: "Search issues by status.",
            inputSchema: { status: z.enum(["open", "closed"]) },
            annotations: { readOnlyHint: true }
        },
        async ({ status }) => {
            const found = issues.filter((i) => i.status === status);
            return { content: [{ type: "text", text: JSON.stringify(found, null, 2) }] };
        }
    );

    server.registerTool(
        "get_issue",
        {
            description: "Get one issue by id.",
            inputSchema: { id: z.string() },
            annotations: { readOnlyHint: true }
        },
        async ({ id }) => {
            const issue = issues.find((i) => i.id === id);
            if (!issue) {
                // isError is the tool failing, not the transport. The engine
                // settles it as a tool error the model reads, and does not retry.
                return { isError: true, content: [{ type: "text", text: `no issue ${id}` }] };
            }
            return { content: [{ type: "text", text: JSON.stringify(issue, null, 2) }] };
        }
    );

    // Not read-only, so `read_only = true` hides it from the model.
    server.registerTool(
        "close_issue",
        {
            description: "Close an issue.",
            inputSchema: { id: z.string() },
            annotations: { readOnlyHint: false, destructiveHint: true }
        },
        async ({ id }) => {
            const issue = issues.find((i) => i.id === id);
            if (issue) issue.status = "closed";
            return {
                content: [{ type: "text", text: issue ? `closed ${id}` : `no issue ${id}` }]
            };
        }
    );

    return server;
}

const app = express();
app.use(express.json());

// One transport per session, which is what the SDK expects: a shared one
// answers the first initialize and rejects every session after it.
const transports = {};

app.all("/mcp", async (req, res) => {
    const sessionId = req.headers["mcp-session-id"];
    let transport = sessionId ? transports[sessionId] : undefined;

    if (!transport) {
        if (sessionId || !isInitializeRequest(req.body)) {
            // An unknown session id gets 404, which tells the engine to shake
            // hands again rather than fail the call.
            return res.status(sessionId ? 404 : 400).json({
                jsonrpc: "2.0",
                error: { code: -32000, message: "no session" },
                id: null
            });
        }
        transport = new StreamableHTTPServerTransport({
            sessionIdGenerator: () => randomUUID(),
            onsessioninitialized: (id) => {
                transports[id] = transport;
            }
        });
        transport.onclose = () => {
            if (transport.sessionId) delete transports[transport.sessionId];
        };
        await buildServer().connect(transport);
    }

    await transport.handleRequest(req, res, req.body);
});

app.listen(4445, () => console.log("mcp server listening on http://localhost:4445/mcp"));
