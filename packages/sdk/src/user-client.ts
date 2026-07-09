import { BaseClient } from "./base";
import type {
    ClientInputRequest,
    ClientInputResponse,
    Event,
    InterruptSessionRequest,
    InterruptSessionResponse,
    StreamSessionEventsParams,
} from "./types";

export class UserClient extends BaseClient {
    /** Post any client input — a submit, an interrupt resume, or a client tool settle —
     *  to the one input endpoint. */
    async send(request: ClientInputRequest): Promise<ClientInputResponse> {
        return this.post("/api/client/sessions/input", request);
    }

    async interruptSession(
        sessionId: string,
        request: InterruptSessionRequest = {},
    ): Promise<InterruptSessionResponse> {
        return this.post(`/api/client/sessions/${sessionId}/interrupt`, request);
    }

    async *streamSessionEvents(sessionId: string, params?: StreamSessionEventsParams): AsyncGenerator<Event> {
        yield* this.streamSSEGet<Event>(`/api/client/sessions/${sessionId}/events/stream`, {
            turn_id: params?.turn_id,
            sequence_after: params?.sequence_after?.toString(),
        });
    }
}
