import { WorkerClient } from "./worker-client";
import { AdminClient } from "./admin-client";
import type { ListSessionsParams, SessionDetail, SessionEventsParams, SessionListResponse } from "./admin-client";
import { drainToTurnResult, persistedOnly, toSubmitToolCallResultRequest } from "./types";
import type {
    ClientPayload,
    Event,
    PersistedEvent,
    SessionScope,
    SubmitRequest,
    SubmitResponse,
    SubmitToolCallResultArgs,
    SubmitToolCallResultResponse,
    TurnResult,
} from "./types";
import type { RequestOptions } from "./base";

export type { SessionScope, TurnResult } from "./types";
export type {
    AggregateSort,
    AggregateSummary,
    ListSessionsParams,
    SessionDetail,
    SessionEventsParams,
    SessionListItem,
    SessionListResponse,
} from "./admin-client";

export interface BackendClientOptions {
    apiKey: string;
    url?: string;
}

const DEFAULT_URL = "https://api.substructure.ai";

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
    /** Include transient `llm.token.delta` events. Off by default, so
     *  `stream()` yields only persisted events and you can `switch` on
     *  `event.payload.type` without a guard. Opt in for live token
     *  rendering; deltas only arrive when streaming is enabled on the
     *  agent's `llmLoop`. */
    tokens?: boolean;
}

export class BackendClient {
    private worker: WorkerClient;
    private admin: AdminClient;

    constructor(options: BackendClientOptions) {
        const baseUrl = options.url ?? DEFAULT_URL;
        const headers = { Authorization: `Bearer ${options.apiKey}` };
        this.worker = new WorkerClient({ baseUrl, headers });
        this.admin = new AdminClient({ baseUrl, headers });
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

    async submitToolCallResult(args: SubmitToolCallResultArgs): Promise<SubmitToolCallResultResponse> {
        return this.worker.submitToolCallResult(args.sessionId, toSubmitToolCallResultRequest(args));
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
     *  filtered to that turn and auto-closes on completion. Yields only
     *  persisted events unless `{ tokens: true }` is passed. */
    stream(scope: SessionScope, options?: StreamOptions & { tokens?: false }): AsyncGenerator<PersistedEvent>;
    stream(scope: SessionScope, options: StreamOptions & { tokens: true }): AsyncGenerator<Event>;
    stream(scope: SessionScope, options: StreamOptions): AsyncGenerator<Event>;
    stream(scope: SessionScope, options?: StreamOptions): AsyncGenerator<Event> {
        const raw = this.worker.streamSessionEvents(scope.sessionId, {
            turn_id: scope.turnId,
            sequence_after: options?.sequenceAfter,
        });
        return options?.tokens ? raw : persistedOnly(raw);
    }

    /** Stream a turn to completion and return its result. Requires `scope.turnId`. */
    turnResult(scope: SessionScope): Promise<TurnResult> {
        if (!scope.turnId) {
            throw new Error("turnResult requires scope.turnId");
        }
        return drainToTurnResult(this.stream(scope));
    }

    /** List sessions for the tenant scoped by this client's API key. */
    listSessions(params?: ListSessionsParams): Promise<SessionListResponse> {
        return this.admin.listSessions(params);
    }

    /** Fetch the current state snapshot for a session. */
    getSession(sessionId: string): Promise<SessionDetail> {
        return this.admin.getSession(sessionId);
    }

    /** Fetch historical events for a session. */
    sessionEvents(sessionId: string, params?: SessionEventsParams): Promise<Event[]> {
        return this.admin.getSessionEvents(sessionId, params);
    }

    /** Stream session events as they are appended (SSE). */
    streamSessionEvents(sessionId: string, params?: SessionEventsParams, opts?: RequestOptions): AsyncGenerator<Event> {
        return this.admin.streamSessionEvents(sessionId, params, opts);
    }
}
