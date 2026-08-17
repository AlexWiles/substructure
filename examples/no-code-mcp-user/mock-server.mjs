// A mock OAuth-protected MCP server, for exercising `credential = "user"`
// end to end on one machine: discovery, dynamic registration, PKCE, consent
// (auto-approved), token exchange, and Bearer-authenticated tool calls.
//
// Every issued token is its own identity: the `whoami` tool answers with the
// token it was called with, so two people connecting see two identities.

import { createHash, randomBytes } from "node:crypto";
import { createServer } from "node:http";

const PORT = Number(process.env.PORT ?? 4478);
const BASE = `http://127.0.0.1:${PORT}`;

const codes = new Map(); // code -> { challenge }
const tokens = new Set();
let issued = 0;

const b64url = (buf) => buf.toString("base64url");
const json = (res, status, body) =>
  res.writeHead(status, { "content-type": "application/json" }).end(JSON.stringify(body));

const rpc = (res, id, result) => json(res, 200, { jsonrpc: "2.0", id, result });

function mcp(req, res, body) {
  const auth = req.headers.authorization ?? "";
  if (!auth.startsWith("Bearer ") || !tokens.has(auth.slice(7))) {
    return res
      .writeHead(401, {
        "www-authenticate": `Bearer resource_metadata="${BASE}/.well-known/oauth-protected-resource/mcp"`,
      })
      .end();
  }
  const request = JSON.parse(body || "{}");
  switch (request.method) {
    case "initialize":
      return rpc(res, request.id, {
        protocolVersion: "2025-11-25",
        capabilities: {},
        serverInfo: { name: "mock-mail", version: "0" },
      });
    case "notifications/initialized":
      return res.writeHead(202).end();
    case "tools/list":
      return rpc(res, request.id, {
        tools: [
          {
            name: "whoami",
            description: "Answers with the identity of the connected account.",
            inputSchema: { type: "object", properties: {} },
          },
        ],
      });
    case "tools/call":
      return rpc(res, request.id, {
        content: [{ type: "text", text: `you are connected as ${auth.slice(7)}` }],
      });
    default:
      return json(res, 404, {
        jsonrpc: "2.0",
        id: request.id,
        error: { code: -32601, message: "no such method" },
      });
  }
}

const server = createServer((req, res) => {
  const url = new URL(req.url, BASE);
  let body = "";
  req.on("data", (chunk) => (body += chunk));
  req.on("end", () => {
    switch (url.pathname) {
      case "/mcp":
        return mcp(req, res, body);
      case "/.well-known/oauth-protected-resource/mcp":
        return json(res, 200, {
          resource: `${BASE}/mcp`,
          authorization_servers: [BASE],
          scopes_supported: ["mail"],
        });
      case "/.well-known/oauth-authorization-server":
        return json(res, 200, {
          issuer: BASE,
          authorization_endpoint: `${BASE}/authorize`,
          token_endpoint: `${BASE}/token`,
          registration_endpoint: `${BASE}/register`,
          response_types_supported: ["code"],
          grant_types_supported: ["authorization_code", "refresh_token"],
          code_challenge_methods_supported: ["S256"],
        });
      case "/register":
        return json(res, 201, { client_id: `client-${randomBytes(4).toString("hex")}` });
      case "/authorize": {
        // Consent auto-approves: the mock's only user always says yes.
        const code = randomBytes(16).toString("hex");
        codes.set(code, { challenge: url.searchParams.get("code_challenge") });
        const redirect = new URL(url.searchParams.get("redirect_uri"));
        redirect.searchParams.set("code", code);
        redirect.searchParams.set("state", url.searchParams.get("state"));
        return res.writeHead(302, { location: redirect.href }).end();
      }
      case "/token": {
        const form = new URLSearchParams(body);
        const held = codes.get(form.get("code"));
        codes.delete(form.get("code"));
        const proof = b64url(createHash("sha256").update(form.get("code_verifier") ?? "").digest());
        if (!held || held.challenge !== proof) {
          return json(res, 400, { error: "invalid_grant" });
        }
        const token = `mock-account-${++issued}`;
        tokens.add(token);
        return json(res, 200, { access_token: token, token_type: "bearer", expires_in: 3600 });
      }
      default:
        return res.writeHead(404).end();
    }
  });
});

server.listen(PORT, "127.0.0.1", () => console.log(`mock mail MCP on ${BASE}/mcp`));
