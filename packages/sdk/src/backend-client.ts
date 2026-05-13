import { WorkerClient } from "./worker-client";
import { RunStream } from "./run-stream";
import type {
    ClientPayload,
    Event,
    StreamSessionEventsParams,
    SubmitClientPayloadResponse,
    SubmitRequest,
    SubmitResponse,
} from "./types";

export { RunStream } from "./run-stream";
export type { TurnResult } from "./run-stream";

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

export interface BackendSubmitRequest {
    agentId: string;
    payload: ClientPayload;
    identity: {
        id: string;
        metadata?: Record<string, string>;
    };
    sessionId?: string;
    turnId?: string;
}

export interface BackendSubmitResult {
    sessionId: string;
    turnId: string;
}

export interface BackendListenOptions {
    turnId?: string;
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

    /** Fire-and-forget: enqueue a payload, return as soon as it's accepted. */
    async submit(request: BackendSubmitRequest): Promise<BackendSubmitResult> {
        const response = await this.worker.submitClientPayload({
            agent_id: request.agentId,
            payload: request.payload,
            identity: request.identity,
            session_id: request.sessionId,
            turn_id: request.turnId,
        });
        return { sessionId: response.session_id, turnId: response.turn_id };
    }

    /** Stream events for a session, optionally scoped to a turn and/or
     *  replayed from a sequence cursor. */
    listen(sessionId: string, options?: BackendListenOptions): AsyncGenerator<Event> {
        return this.worker.streamSessionEvents(sessionId, {
            turn_id: options?.turnId,
            sequence_after: options?.sequenceAfter,
        });
    }

    /** Sugar: submit and immediately listen for events on the resulting turn. */
    submitAndListen(request: BackendSubmitRequest): RunStream {
        const self = this;
        const source = (async function* () {
            const { sessionId, turnId } = await self.submit(request);
            yield* self.listen(sessionId, { turnId, sequenceAfter: 0 });
        })();
        return new RunStream(source);
    }
}
