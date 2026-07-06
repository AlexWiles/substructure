import { BaseClient } from "./base";
import type {
    Event,
    InterruptSessionRequest,
    InterruptSessionResponse,
    ResumeInterruptRequest,
    ResumeInterruptResponse,
    SettleEffectRequest,
    SettleEffectResponse,
    StreamSessionEventsParams,
    SubmitClientPayloadResponse,
    SubmitPayloadRequest,
} from "./types";

export class UserClient extends BaseClient {
    async submitPayload(request: SubmitPayloadRequest): Promise<SubmitClientPayloadResponse> {
        return this.post("/api/client/sessions/submit", request);
    }

    async settleEffect(sessionId: string, request: SettleEffectRequest): Promise<SettleEffectResponse> {
        return this.post(`/api/client/sessions/${sessionId}/calls/settle`, request);
    }

    async interruptSession(
        sessionId: string,
        request: InterruptSessionRequest = {},
    ): Promise<InterruptSessionResponse> {
        return this.post(`/api/client/sessions/${sessionId}/interrupt`, request);
    }

    async resumeInterrupt(sessionId: string, request: ResumeInterruptRequest): Promise<ResumeInterruptResponse> {
        return this.post(`/api/client/sessions/${sessionId}/interrupt/resume`, request);
    }

    async *streamSessionEvents(sessionId: string, params?: StreamSessionEventsParams): AsyncGenerator<Event> {
        yield* this.streamSSEGet<Event>(`/api/client/sessions/${sessionId}/events/stream`, {
            turn_id: params?.turn_id,
            sequence_after: params?.sequence_after?.toString(),
        });
    }
}
