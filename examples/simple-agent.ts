import { Agent } from "@substructure.ai/sdk/agent";
import { Worker } from "@substructure.ai/sdk/worker-handler";
import { UserClient } from "@substructure.ai/sdk/user";

const RUNTIME_URL = "http://localhost:8080";
const WORKER_PORT = 4444;
const TENANT_ID = "default";

// ── Define agents ────────────────────────────────────────────────────────────

const mathAgent = new Agent({
  id: "math-agent",
  systemPrompt: "You are a math assistant. Use tools to compute results. Be concise.",
  llm: {
    model: "minimax/minimax-m2.5",
    client: "openrouter",
    retry: { timeout_secs: 120, max_retries: 3, backoff_base_secs: 1, backoff_max_secs: 10 },
  },
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
  systemPrompt: "You are a weather assistant. Use tools when appropriate. Be concise.",
  llm: {
    model: "openrouter/hunter-alpha",
    client: "openrouter",
    retry: { timeout_secs: 120, max_retries: 3, backoff_base_secs: 1, backoff_max_secs: 10 },
  },
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

const server = Bun.serve({ port: WORKER_PORT, fetch: worker.fetchHandler() });
console.log(`Worker listening on port ${WORKER_PORT}`);

const res = await worker.register({
  runtimeUrl: RUNTIME_URL,
  tenantId: TENANT_ID,
  endpointUrl: `http://localhost:${WORKER_PORT}`,
});

if (!res.ok) {
  console.error("Failed to register");
  process.exit(1);
}

console.log(`Registered agents: ${worker.agentIds.join(", ")}`);

// ── Send a message ───────────────────────────────────────────────────────────

console.log("\nAsking weather agent to sum temperatures...\n");

const userClient = new UserClient({ baseUrl: RUNTIME_URL });

for await (const event of userClient.sendMessage({
  agent_id: "weather-agent",
  message: "What is the sum of the current temperatures in San Francisco and New York?",
})) {
  console.log(event.derived.agent_id, event.aggregate_type, event.aggregate_id, event.payload.type);
}

server.stop();
