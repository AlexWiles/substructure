import type {
    ClientPayload,
    Event,
    PersistedEvent,
    SessionScope,
    SubmitToolCallResultArgs,
    SubmitToolCallResultResponse,
    TurnResult,
} from "./types";
import { drainToTurnResult, persistedOnly, toSubmitToolCallResultRequest } from "./types";
import { UserClient } from "./user-client";

export type { SessionScope, TurnResult } from "./types";

export interface FrontendClientOptions {
    token: string;
    url?: string;
}

const DEFAULT_URL = "https://api.substructure.ai";

export interface StartTurnRequest {
    agentId: string;
    payload: ClientPayload;
    sessionId?: string;
    turnId?: string;
}

export interface StreamOptions {
    sequenceAfter?: number;
    /** Include transient `llm.token.delta` events. Off by default, so
     *  `stream()` yields only persisted events and you can `switch` on
     *  `event.payload.type` without a guard. Opt in for live token
     *  rendering; deltas only arrive when streaming is enabled on the
     *  agent's `llmToolLoop`. */
    tokens?: boolean;
}

export class FrontendClient {
    private user: UserClient;

    constructor(options: FrontendClientOptions) {
        this.user = new UserClient({
            baseUrl: options.url ?? DEFAULT_URL,
            headers: { Authorization: `Bearer ${options.token}` },
        });
    }

    /** Fire-and-forget: enqueue a turn, return as soon as it's accepted. */
    async startTurn(request: StartTurnRequest): Promise<SessionScope> {
        const response = await this.user.submitPayload({
            agent_id: request.agentId,
            payload: request.payload,
            session_id: request.sessionId,
            turn_id: request.turnId,
        });
        return { sessionId: response.session_id, turnId: response.turn_id };
    }

    async submitToolCallResult(args: SubmitToolCallResultArgs): Promise<SubmitToolCallResultResponse> {
        return this.user.submitToolCallResult(args.sessionId, toSubmitToolCallResultRequest(args));
    }

    /** Stream events for a session. If `scope.turnId` is set, the stream is
     *  filtered to that turn and auto-closes on completion. Yields only
     *  persisted events unless `{ tokens: true }` is passed. */
    stream(scope: SessionScope, options?: StreamOptions & { tokens?: false }): AsyncGenerator<PersistedEvent>;
    stream(scope: SessionScope, options: StreamOptions & { tokens: true }): AsyncGenerator<Event>;
    stream(scope: SessionScope, options: StreamOptions): AsyncGenerator<Event>;
    stream(scope: SessionScope, options?: StreamOptions): AsyncGenerator<Event> {
        const raw = this.user.streamSessionEvents(scope.sessionId, {
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
}
