// A chat agent that verifies each request's signature in middleware before deciding.
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { createHmac, timingSafeEqual } from "node:crypto";


function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return { agent: { model: "claude-haiku-4-5-20251001", stream: true } };
    }

    // Accept the engine's proposal for every other decision.
    return proposed;
}


const app = new Hono();

// Hardcoded for the example only. In production load this from a secret store
// and pass the same value to the CLI with --signing-secret.
const SIGNING_SECRET = "dev-secret-not-for-production";

// Reject any request whose signature doesn't check out. The engine signs each
// request with HMAC-SHA256 over the raw body, sending a `sha256=<hex>` signature.
app.use(async (c, next) => {
    const body = await c.req.text();
    const expected = Buffer.from("sha256=" + createHmac("sha256", SIGNING_SECRET).update(body).digest("hex"));
    const received = Buffer.from(c.req.header("X-Substructure-Signature") ?? "");

    // Compare on byte length — timingSafeEqual throws if the buffers differ.
    const ok = received.length === expected.length && timingSafeEqual(received, expected);

    if (!ok) return c.text("invalid signature", 401);
    await next();
});

app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
