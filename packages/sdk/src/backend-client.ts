import { WorkerClient } from "./worker-client";
import { drainToTurnResult } from "./turn";
import type { SessionScope, TurnResult } from "./turn";
import type { ClientPayload, Event, SubmitRequest, SubmitResponse } from "./types";

export type { SessionScope, TurnResult } from "./turn";

export interface BackendClientOptions {
    url: string;
    apiKey: string;
}

export interface IssueClientTokenRequest {
    identity: {
        id: string;
        metadata?: Record<string, string>;
    };
    ttlSeconds?: number;
}

export interface IssueClientTokenResponse {
    token: string;
    expiresAt: number;
}

export interface StartTurnRequest {
    agentId: string;
    payload: ClientPayload;
    identity: {
        id: string;
        metadata?: Record<string, string>;
    };
    sessionId?: string;
    turnId?: string;
}

export interface StreamOptions {
    sequenceAfter?: number;
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
            identity: request.identity,
            ttl_seconds: request.ttlSeconds,
        });
        return { token: response.token, expiresAt: response.expires_at };
    }

    async submitWorkerDecision(request: SubmitRequest): Promise<SubmitResponse> {
        return this.worker.submit(request);
    }

    /** Fire-and-forget: enqueue a turn, return as soon as it's accepted. */
    async startTurn(request: StartTurnRequest): Promise<SessionScope> {
        const response = await this.worker.submitClientPayload({
            agent_id: request.agentId,
            payload: request.payload,
            identity: request.identity,
            session_id: request.sessionId,
            turn_id: request.turnId,
        });
        return { sessionId: response.session_id, turnId: response.turn_id };
    }

    /** Stream events for a session. If `scope.turnId` is set, the stream is
     *  filtered to that turn and auto-closes on completion. */
    stream(scope: SessionScope, options?: StreamOptions): AsyncGenerator<Event> {
        return this.worker.streamSessionEvents(scope.sessionId, {
            turn_id: scope.turnId,
            sequence_after: options?.sequenceAfter,
        });
    }

    /** Stream a turn to completion and return its result. Requires `scope.turnId`. */
    turnResult(scope: SessionScope): Promise<TurnResult> {
        if (!scope.turnId) {
            throw new Error("turnResult requires scope.turnId");
        }
        return drainToTurnResult(this.stream(scope));
    }
}
