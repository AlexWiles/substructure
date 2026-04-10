import Substructure from "@substructure.ai/sdk";
import { researchAgent } from "./agent";

// ── Config ──────────────────────────────────────────────────────────────────

const SERVER_URL = process.env.SERVER_URL ?? "http://localhost:8080";
const API_KEY = process.env.API_KEY ?? "dev-worker-key";
const WORKER_PORT = Number(process.env.WORKER_PORT ?? 4444);
const API_PORT = Number(process.env.API_PORT ?? 3001);

// ── Runtime & Worker ────────────────────────────────────────────────────────

const sub = new Substructure();

const embedded = await sub.embedded({
    agents: [researchAgent],
    db: "web-ui-example.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const workerServer = Bun.serve({
    port: WORKER_PORT,
    fetch: embedded.fetchHandler(),
});

// ── Register worker with Substructure server ────────────────────────────────

const backend = sub.backend.client({ url: SERVER_URL, apiKey: API_KEY });

await backend.registerWorker({
    transport_type: "http",
    config: { endpoint_url: `http://localhost:${WORKER_PORT}` },
});

// ── API Server ──────────────────────────────────────────────────────────────

function corsHeaders(): Record<string, string> {
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    };
}

const apiServer = Bun.serve({
    port: API_PORT,
    fetch: async (req) => {
        const url = new URL(req.url);

        // CORS preflight
        if (req.method === "OPTIONS") {
            return new Response(null, { status: 204, headers: corsHeaders() });
        }

        // GET /api/token — mint a real JWT via BackendClient
        if (url.pathname === "/api/token" && req.method === "GET") {
            const { token, expiresAt } = await backend.mintClientToken({
                tenantId: "default",
                sub: "web-user",
                ttlSeconds: 600,
            });
            return Response.json({ token, expiresAt }, { headers: corsHeaders() });
        }

        // GET /api/config — tell the frontend where the Substructure server is
        if (url.pathname === "/api/config" && req.method === "GET") {
            return Response.json({ serverUrl: SERVER_URL, agentId: researchAgent.agentId }, { headers: corsHeaders() });
        }

        return new Response("Not found", { status: 404, headers: corsHeaders() });
    },
});

console.log(`Worker server running on http://localhost:${workerServer.port}`);
console.log(`API server running on http://localhost:${apiServer.port}`);
console.log(`Substructure server: ${SERVER_URL}`);
console.log(`Agent: ${researchAgent.agentId}`);

// ── Graceful shutdown ───────────────────────────────────────────────────────

process.on("SIGINT", async () => {
    console.log("\nShutting down...");
    workerServer.stop();
    apiServer.stop();
    await embedded.shutdown();
    process.exit(0);
});
