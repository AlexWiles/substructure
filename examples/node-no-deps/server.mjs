// A complete chat agent served with Node's built-in http server. No dependencies.
import { createServer } from "node:http";

// The whole worker: accept the engine's proposal for every decision. The agent
// is declared in substructure.toml, and arrives as the `session.start`
// proposal, so there is nothing to author until you want to override something.
function decide({ proposed }) {
    return proposed;
}

const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        const decision = decide(JSON.parse(body));
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(decision));
    });
});

server.listen(4444, () =>
    console.log("worker listening on http://localhost:4444"));
