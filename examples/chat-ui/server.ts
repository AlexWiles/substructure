import { createServer } from "node:http";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNodeHttpEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080";
const PORT = Number(process.env.RUNTIME_PORT || 4111);

const server = createServer((req, res) => {
  // Parse sessionId from query string
  const url = new URL(req.url || "/", `http://localhost:${PORT}`);
  const sessionId = url.searchParams.get("sessionId");

  if (!sessionId) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "sessionId query parameter is required" }));
    return;
  }

  const runtime = new CopilotRuntime({
    agents: {
      default: new HttpAgent({
        url: `${BACKEND_URL}/sessions/${sessionId}/ag-ui`,
      }),
    },
  });

  const handler = copilotRuntimeNodeHttpEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });

  handler(req, res);
});

server.listen(PORT, () => {
  console.log(`CopilotKit runtime listening on http://localhost:${PORT}`);
});
