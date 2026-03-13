import { Agent } from "../src/agent";
import { Worker } from "../src/worker";
import { Client } from "../src/client";
import type { WorkerDecisionRequestWire } from "../src/types";

const RUNTIME_URL = "http://localhost:8080";
const WORKER_PORT = 4444;
const TENANT_ID = "default";

// ── Define agents ────────────────────────────────────────────────────────────

const mathAgent = new Agent({
  id: "math-agent",
  model: "openrouter/hunter-alpha",
  systemPrompt: "You are a math assistant. Use tools to compute results. Be concise.",
  llmClient: "openrouter",
});

mathAgent.tool(
  "add",
  "Add two numbers",
  {
    type: "object",
    properties: {
      a: { type: "number" },
      b: { type: "number" },
    },
    required: ["a", "b"],
  },
  ({ a, b }) => ({ result: (a as number) + (b as number) }),
);

const weatherAgent = new Agent({
  id: "weather-agent",
  model: "openrouter/hunter-alpha",
  systemPrompt: "You are a weather assistant. Use tools when appropriate. Be concise.",
  llmClient: "openrouter",
  retry: { timeout_secs: 120, max_retries: 0, backoff_base_secs: 0, backoff_max_secs: 0 },
});

weatherAgent.tool(
  "get_weather",
  "Get the current weather for a city. Returns temperature in fahrenheit.",
  {
    type: "object",
    properties: {
      city: { type: "string", description: "City name" },
    },
    required: ["city"],
  },
  async ({ city }) => ({ city, temp_f: city === "San Francisco" ? 62 : 78, condition: "sunny" }),
);

weatherAgent.subAgent(mathAgent, "A math assistant that can add numbers. Send it a math question.");

// ── Serve and register ───────────────────────────────────────────────────────

const worker = Worker.from(weatherAgent, mathAgent);

const server = Bun.serve({
  port: WORKER_PORT,
  async fetch(req) {
    const decision = (await req.json()) as WorkerDecisionRequestWire;
    const submit = await worker.handleDecision(decision);
    return Response.json(submit);
  },
});

console.log(`Worker listening on port ${WORKER_PORT}`);

const client = new Client({ baseUrl: RUNTIME_URL });
const res = await client.register({
  tenant_id: TENANT_ID,
  agent_ids: worker.agentIds,
  transport_type: "http",
  config: { endpoint_url: `http://localhost:${WORKER_PORT}` },
});

if (!res.ok) {
  console.error("Failed to register");
  process.exit(1);
}

console.log(`Registered agents: ${worker.agentIds.join(", ")}`);

// ── Send a message ───────────────────────────────────────────────────────────

console.log("\nAsking weather agent to sum temperatures...\n");

for await (const event of client.sendMessage({
  agent_id: "weather-agent",
  message: "What is the sum of the current temperatures in San Francisco and New York?",
})) {
  console.log(event.payload.type);
}

server.stop();
