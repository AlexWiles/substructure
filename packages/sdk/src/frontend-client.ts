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
    /** Include transient `llm.token.delta` events. Off by default, so `stream()`
     *  yields only persisted events. Deltas only arrive when streaming is enabled
     *  on the agent's `llmToolLoop`. */
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

    /** Pause a running session. The agent stops between decisions and the
     *  session rejects new user input until resumed. */
    async interrupt(
        sessionId: string,
        options?: { interruptId?: string; reason?: string; payload?: unknown },
    ): Promise<{ interruptId: string }> {
        const response = await this.user.interruptSession(sessionId, {
            interrupt_id: options?.interruptId,
            reason: options?.reason,
            payload: options?.payload,
        });
        return { interruptId: response.interrupt_id };
    }

    async resume(sessionId: string, interruptId: string, payload?: unknown): Promise<void> {
        await this.user.resumeInterrupt(sessionId, { interrupt_id: interruptId, payload });
    }

    /** When `scope.turnId` is set, the stream is filtered to that turn and
     *  auto-closes on completion. */
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

    /** Requires `scope.turnId`. */
    turnResult(scope: SessionScope): Promise<TurnResult> {
        if (!scope.turnId) {
            throw new Error("turnResult requires scope.turnId");
        }
        return drainToTurnResult(this.stream(scope));
    }
}
