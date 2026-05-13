import { UserClient } from "./user-client";
import { RunStream } from "./run-stream";
import type { ClientPayload, Event } from "./types";

export { RunStream } from "./run-stream";
export type { TurnResult } from "./run-stream";

export interface FrontendClientOptions {
    url: string;
    token: string;
}

export interface FrontendSubmitRequest {
    agentId: string;
    payload: ClientPayload;
    sessionId?: string;
    turnId?: string;
}

export interface FrontendSubmitResult {
    sessionId: string;
    turnId: string;
}

export interface FrontendListenOptions {
    turnId?: string;
    sequenceAfter?: number;
}

export class FrontendClient {
    private user: UserClient;

    constructor(options: FrontendClientOptions) {
        this.user = new UserClient({
            baseUrl: options.url,
            headers: { Authorization: `Bearer ${options.token}` },
        });
    }

    /** Fire-and-forget: enqueue a payload, return as soon as it's accepted. */
    async submit(request: FrontendSubmitRequest): Promise<FrontendSubmitResult> {
        const response = await this.user.submitPayload({
            agent_id: request.agentId,
            payload: request.payload,
            session_id: request.sessionId,
            turn_id: request.turnId,
        });
        return { sessionId: response.session_id, turnId: response.turn_id };
    }

    /** Stream events for a session, optionally scoped to a turn and/or
     *  replayed from a sequence cursor. */
    listen(sessionId: string, options?: FrontendListenOptions): AsyncGenerator<Event> {
        return this.user.streamSessionEvents(sessionId, {
            turn_id: options?.turnId,
            sequence_after: options?.sequenceAfter,
        });
    }

    /** Sugar: submit and immediately listen for events on the resulting turn. */
    submitAndListen(request: FrontendSubmitRequest): RunStream {
        const self = this;
        const source = (async function* () {
            const { sessionId, turnId } = await self.submit(request);
            yield* self.listen(sessionId, { turnId, sequenceAfter: 0 });
        })();
        return new RunStream(source);
    }
}
