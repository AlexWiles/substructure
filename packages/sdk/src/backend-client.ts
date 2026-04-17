import { WorkerClient } from "./worker-client";
import { RunStream } from "./run-stream";
import type { ClientPayload, MintClientTokenRequest, SubmitRequest, SubmitResponse } from "./types";

export { RunStream } from "./run-stream";
export type { TurnResult } from "./run-stream";

export interface BackendClientOptions {
    url: string;
    apiKey: string;
}

export interface IssueClientTokenRequest {
    tenantId: string;
    sub: string;
    attrs?: Record<string, string>;
    ttlSeconds?: number;
}

export interface IssueClientTokenResponse {
    token: string;
    expiresAt: number;
}

export interface BackendSubmitRequest {
    agentId: string;
    payload: ClientPayload;
    auth: {
        tenant_id: string;
        sub: string;
        attrs?: Record<string, string>;
    };
    sessionId?: string;
    turnId?: string;
}

export class BackendClient {
    private worker: WorkerClient;

    constructor(options: BackendClientOptions) {
        this.worker = new WorkerClient({
            baseUrl: options.url,
            headers: { Authorization: `Bearer ${options.apiKey}` },
        });
    }

    async mintClientToken(request: IssueClientTokenRequest): Promise<IssueClientTokenResponse> {
        const response = await this.worker.mintClientToken({
            tenant_id: request.tenantId,
            sub: request.sub,
            attrs: request.attrs,
            ttl_seconds: request.ttlSeconds,
        } as MintClientTokenRequest);
        return { token: response.token, expiresAt: response.expires_at };
    }

    async submitWorkerDecision(request: SubmitRequest): Promise<SubmitResponse> {
        return this.worker.submit(request);
    }

    submit(request: BackendSubmitRequest): RunStream {
        const stream = this.worker.submitClientPayload({
            agent_id: request.agentId,
            payload: request.payload,
            auth: request.auth,
            session_id: request.sessionId,
            turn_id: request.turnId,
        });
        return new RunStream(stream);
    }
}
