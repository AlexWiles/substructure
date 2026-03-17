import { Substructure, Agent } from "@substructure.ai/sdk/substructure";

// ── Define agents ───────────────────────────────────────────────────────────

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

// ── Setup ────────────────────────────────────────────────────────────────────

const sub = new Substructure({
  db: "data.db",
  openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

await sub.agents(weatherAgent, mathAgent);

// ── Run ──────────────────────────────────────────────────────────────────────

console.log("Asking weather agent to sum temperatures...\n");

const stream = sub.run(
    "weather-agent",
    "What is the sum of the current temperatures in San Francisco and New York?"
)

for await (const event of stream) {
  console.log(event.derived?.agent_id, event.aggregate_type, event.aggregate_id, event.payload.type);
}

await sub.shutdown();
