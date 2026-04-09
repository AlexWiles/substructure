import { EmbeddedRuntime } from "@substructure.ai/runtime";
import Substructure from "@substructure.ai/sdk";
import type { Event } from "@substructure.ai/sdk";
import { codingAgent, AGENT_ID } from "./worker";

// ── Substructure Runtime ─────────────────────────────────────────────────────

const runtime = new EmbeddedRuntime({
    db: "coding-agent.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const sub = new Substructure();
const embedded = await sub.embedded({ agents: [codingAgent], runtime });

// ── HTTP Server ──────────────────────────────────────────────────────────────

const PORT = Number(process.env.PORT ?? 3001);

const server = Bun.serve({
    port: PORT,
    idleTimeout: 255, // max: LLM calls + tool execution can take a while
    fetch: async (req) => {
        const url = new URL(req.url);

        // POST /submit — send a message, receive NDJSON event stream
        if (url.pathname === "/submit" && req.method === "POST") {
            const body = (await req.json()) as {
                message: string;
                sessionId: string;
            };

            const stream = embedded.submit({
                agentId: AGENT_ID,
                payload: {
                    type: "message",
                    message: { role: "user", content: body.message },
                },
                sessionId: body.sessionId,
                auth: { tenant_id: "default", sub: "cli-user" },
                turnId: crypto.randomUUID(),
            });

            const encoder = new TextEncoder();
            const readable = new ReadableStream({
                async start(controller) {
                    try {
                        for await (const event of stream) {
                            controller.enqueue(
                                encoder.encode(JSON.stringify(event) + "\n"),
                            );
                        }
                    } catch (err) {
                        controller.enqueue(
                            encoder.encode(
                                JSON.stringify({ error: String(err) }) + "\n",
                            ),
                        );
                    } finally {
                        controller.close();
                    }
                },
            });

            return new Response(readable, {
                headers: {
                    "Content-Type": "application/x-ndjson",
                    "Cache-Control": "no-cache",
                },
            });
        }

        // GET /health
        if (url.pathname === "/health") {
            return Response.json({ ok: true, agent: AGENT_ID });
        }

        return new Response("Not found", { status: 404 });
    },
});

console.log(`Coding agent server running on http://localhost:${server.port}`);
console.log(`Agent: ${AGENT_ID}`);
console.log(`Working directory: ${process.cwd()}`);

// Graceful shutdown
process.on("SIGINT", async () => {
    console.log("\nShutting down...");
    server.stop();
    await embedded.shutdown();
    process.exit(0);
});
