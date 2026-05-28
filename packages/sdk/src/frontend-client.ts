import type {
    ClientPayload,
    Event,
    SessionScope,
    SubmitToolCallResultArgs,
    SubmitToolCallResultResponse,
    TurnResult,
} from "./types";
import { drainToTurnResult, toSubmitToolCallResultRequest } from "./types";
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
     *  filtered to that turn and auto-closes on completion. */
    stream(scope: SessionScope, options?: StreamOptions): AsyncGenerator<Event> {
        return this.user.streamSessionEvents(scope.sessionId, {
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
