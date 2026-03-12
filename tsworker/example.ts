/** Example substructure2 worker with a weather agent. */

import { z } from "zod";
import { Agent, Client, Worker } from "./src/index.js";

const agent = new Agent("weather", {
  model: "arcee-ai/trinity-large-preview:free",
  instructions:
    "You are a helpful weather assistant. Use the available tools to answer questions about weather.",
  llmClient: "openrouter",
  retry: { timeout_secs: 30, max_retries: 3, backoff_base_secs: 2, backoff_max_secs: 60 },
});

agent.tool("get_weather", {
  description: "Get the current weather for a city.",
  parameters: z.object({ city: z.string().describe("The city name to look up") }),
  retry: { max_retries: 2, backoff_base_secs: 1, backoff_max_secs: 5 },
  execute: async (_ctx, { city }) => {
    const temp = Math.floor(Math.random() * 45) + 50;
    const conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"][
      Math.floor(Math.random() * 5)
    ];
    return `${temp}°F and ${conditions} in ${city}`;
  },
});

agent.tool("get_forecast", {
  description: "Get a multi-day weather forecast.",
  parameters: z.object({
    city: z.string().describe("The city name to look up"),
    days: z.number().default(3).describe("Number of days to forecast"),
  }),
  retry: { max_retries: 2, backoff_base_secs: 1, backoff_max_secs: 5 },
  execute: async (_ctx, { city, days }) => {
    const lines: string[] = [];
    for (let i = 0; i < days; i++) {
      const temp = Math.floor(Math.random() * 45) + 50;
      const cond = ["sunny", "cloudy", "rainy", "partly cloudy"][
        Math.floor(Math.random() * 4)
      ];
      lines.push(`  Day ${i + 1}: ${temp}°F, ${cond}`);
    }
    return `Forecast for ${city}:\n${lines.join("\n")}`;
  },
});

agent.tool("convert_temperature", {
  description: "Convert a temperature between Fahrenheit and Celsius.",
  parameters: z.object({
    value: z.number().describe("The temperature value"),
    to_unit: z
      .string()
      .describe("Target unit, either 'celsius' or 'fahrenheit'"),
  }),
  retry: { max_retries: 1, backoff_base_secs: 1, backoff_max_secs: 3 },
  execute: async (_ctx, { value, to_unit }) => {
    if (to_unit.toLowerCase().startsWith("c")) {
      const converted = ((value - 32) * 5) / 9;
      return `${value}°F = ${converted.toFixed(1)}°C`;
    }
    const converted = (value * 9) / 5 + 32;
    return `${value}°C = ${converted.toFixed(1)}°F`;
  },
});

// --- Run ---

const client = new Client("http://localhost:8080");
const worker = new Worker(client, { agents: [agent] });
worker.run();

// Send a message and stream events
(async () => {
  // Give worker a moment to start
  await new Promise((r) => setTimeout(r, 500));

  for await (const event of client.send("weather", "What's the weather in NYC?")) {
    const { type } = event.payload;
    switch (type) {
      case "session.done":
        console.log("Done:", JSON.stringify(event.payload.artifacts));
        break;
      case "message.new":
        console.log("Message:", event.payload.message.role, event.payload.message.content?.slice(0, 120));
        break;
      default:
        console.log("Event:", type);
    }
  }
})();
