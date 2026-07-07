// A complete chat agent. No SDK, no dependencies — just Node's http server.
// The engine POSTs a decision request; this returns the next actions.
//
// Point a local Substructure server at it:
//   substructure serve --dev --provider openrouter --worker-url http://localhost:4444
import { createServer } from "node:http";

const PORT = 4444;

createServer((req, res) => {
  let body = "";
  req.on("data", (chunk) => (body += chunk));
  req.on("end", () => {
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify(decide(JSON.parse(body))));
  });
}).listen(PORT, () => console.log(`worker listening on http://localhost:${PORT}`));

function decide(req) {
  const t = req.trigger;
  let decision = {};

  // The client sent the conversation → prompt the model.
  if (t.type === "client.messages") {
    decision = {
      messages: t.messages,
      actions: [{ type: "llm.call", handler: "server", stream: true,
        request: { model: "anthropic/claude-sonnet-4-6", messages: t.messages } }],
    };
  }

  // The model answered → record it, end the turn.
  if (t.type === "llm.finished") {
    decision = {
      messages: [...req.messages, t.message],
      actions: [{ type: "done", data: t.message.content }],
    };
  }

  return { actions: [], ...decision };
}
