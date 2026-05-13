import { DurableObject } from "cloudflare:workers";
import type { AgentRequest, MiddlewareFn, Next } from "@substructure.ai/sdk";

export class AgentState extends DurableObject {
    async fetch(request: Request): Promise<Response> {
        if (request.method === "GET") {
            const state = (await this.ctx.storage.get("state")) ?? {};
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

export function durableObjectState(getNamespace: () => DurableObjectNamespace<AgentState>): MiddlewareFn<unknown> {
    return async (req: AgentRequest<unknown>, next: Next<unknown>) => {
        const sessionId = req.wire.session_id;
        const ns = getNamespace();
        const stub = ns.get(ns.idFromName(sessionId));

        const resp = await stub.fetch(new Request("https://state/"));
        const state = (await resp.json()) as Record<string, unknown>;

        const result = await next({ ...req, state });

        await stub.fetch(
            new Request("https://state/", {
                method: "PUT",
                body: JSON.stringify(result.state),
            }),
        );

        return {
            ...result,
            workerState: btoa(JSON.stringify({ ref: sessionId })),
        };
    };
}
