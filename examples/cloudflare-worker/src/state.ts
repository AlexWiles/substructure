import { DurableObject } from "cloudflare:workers";
import type { AgentRequest, Next, MiddlewareFn } from "@substructure.ai/sdk/worker-handler";

/**
 * Durable Object that stores agent state per session.
 * Each session gets its own DO instance, keyed by session_id.
 */
export class AgentState extends DurableObject {
    async fetch(request: Request): Promise<Response> {
        if (request.method === "GET") {
            const state = await this.ctx.storage.get("state") ?? {};
            return Response.json(state);
        }
        if (request.method === "PUT") {
            const state = await request.json();
            await this.ctx.storage.put("state", state);
            return new Response("ok");
        }
        return new Response("not found", { status: 404 });
    }
}

/**
 * Middleware that replaces `withState()` — stores state in a Durable Object
 * instead of encoding it as base64 in the wire format.
 *
 * The runtime receives only a minimal reference; the full state lives in the DO.
 */
export function withDurableObjectState(
    getNamespace: () => DurableObjectNamespace<AgentState>,
): MiddlewareFn<unknown> {
    return async (req: AgentRequest<unknown>, next: Next<unknown>) => {
        const sessionId = req.wire.session_id;
        const ns = getNamespace();
        const id = ns.idFromName(sessionId);
        const stub = ns.get(id);

        // Read state from DO
        const resp = await stub.fetch(new Request("https://state/"));
        const state = await resp.json() as Record<string, unknown>;

        const result = await next({ ...req, state });

        // Write state back to DO
        await stub.fetch(new Request("https://state/", {
            method: "PUT",
            body: JSON.stringify(result.state),
        }));

        // Return minimal reference to runtime
        return {
            ...result,
            workerState: btoa(JSON.stringify({ ref: sessionId })),
        };
    };
}
