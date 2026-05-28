import { BaseClient } from "./base";
import type {
    Event,
    StreamSessionEventsParams,
    SubmitClientPayloadResponse,
    SubmitPayloadRequest,
    SubmitToolCallResultRequest,
    SubmitToolCallResultResponse,
} from "./types";

export class UserClient extends BaseClient {
    async submitPayload(request: SubmitPayloadRequest): Promise<SubmitClientPayloadResponse> {
        return this.post("/api/client/sessions/submit", request);
    }

    async submitToolCallResult(
        sessionId: string,
        request: SubmitToolCallResultRequest,
    ): Promise<SubmitToolCallResultResponse> {
        return this.post(`/api/client/sessions/${sessionId}/tool-call-results`, request);
    }

    async *streamSessionEvents(sessionId: string, params?: StreamSessionEventsParams): AsyncGenerator<Event> {
        yield* this.streamSSEGet<Event>(`/api/client/sessions/${sessionId}/events/stream`, {
            turn_id: params?.turn_id,
            sequence_after: params?.sequence_after?.toString(),
        });
    }
}
