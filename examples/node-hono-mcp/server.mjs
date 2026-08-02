// A chat agent whose tools come from an MCP server, served with Hono.
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

// Connect to an MCP server over stdio and pull in its tools. Here we run the
// filesystem server scoped to this directory, but any MCP server works.
const mcp = new Client({ name: "subs-mcp-example", version: "1.0.0" });
await mcp.connect(new StdioClientTransport({
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-filesystem", process.cwd()]
}));
const { tools } = await mcp.listTools();
const mcpToolNames = new Set(tools.map((t) => t.name));

async function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        // Offer every MCP tool to the model as-is.
        return {
            agent: {
                ...proposed.agent,
                tools: tools.map((t) => ({
                    name: t.name,
                    description: t.description,
                    input: t.inputSchema
                }))
            }
        };
    }

    // Forward the call to the MCP server and settle with its output.
    if (trigger.type === "tool.execute" && mcpToolNames.has(trigger.name)) {
        const res = await mcp.callTool({
            name: trigger.name,
            arguments: trigger.input.value ?? {}
        });

        const text = res.content
            .filter((c) => c.type === "text")
            .map((c) => c.text)
            .join("\n");
        if (res.isError) return { actions: [{ type: "tool.error", error: text }] };
        return { actions: [{ type: "tool.result", result: text }] };
    }

    return proposed;
}


const app = new Hono();
app.post("/", async (c) => c.json(await decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
