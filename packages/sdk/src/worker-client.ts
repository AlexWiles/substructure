import { BaseClient } from "./base";
import type {
    Event,
    MachineSubmitPayloadRequest,
    MintClientTokenRequest,
    MintClientTokenResponse,
    StreamSessionEventsParams,
    SubmitClientPayloadResponse,
    SubmitRequest,
    SubmitResponse,
    SubmitToolCallResultRequest,
    SubmitToolCallResultResponse,
    WorkerAuthOptions,
} from "./types";

export class WorkerClient extends BaseClient {
    async submit(request: SubmitRequest, auth?: WorkerAuthOptions): Promise<SubmitResponse> {
        return this.post("/api/machine/workers/submit", request, { headers: buildWorkerAuthHeaders(auth) });
    }

    async mintClientToken(request: MintClientTokenRequest, auth?: WorkerAuthOptions): Promise<MintClientTokenResponse> {
        return this.post("/api/machine/client-tokens", request, { headers: buildWorkerAuthHeaders(auth) });
    }

    async submitClientPayload(
        request: MachineSubmitPayloadRequest,
        auth?: WorkerAuthOptions,
    ): Promise<SubmitClientPayloadResponse> {
        return this.post("/api/machine/sessions/submit", request, {
            headers: buildWorkerAuthHeaders(auth),
        });
    }

    async submitToolCallResult(
        sessionId: string,
        request: SubmitToolCallResultRequest,
        auth?: WorkerAuthOptions,
    ): Promise<SubmitToolCallResultResponse> {
        return this.post(`/api/machine/sessions/${sessionId}/tool-call-results`, request, {
            headers: buildWorkerAuthHeaders(auth),
        });
    }

    async *streamSessionEvents(
        sessionId: string,
        params?: StreamSessionEventsParams,
        auth?: WorkerAuthOptions,
    ): AsyncGenerator<Event> {
        yield* this.streamSSEGet<Event>(
            `/api/machine/sessions/${sessionId}/events/stream`,
            {
                turn_id: params?.turn_id,
                sequence_after: params?.sequence_after?.toString(),
            },
            { headers: buildWorkerAuthHeaders(auth) },
        );
    }
}

function buildWorkerAuthHeaders(auth?: WorkerAuthOptions): Record<string, string> | undefined {
    if (!auth) {
        return undefined;
    }
    return { Authorization: `Bearer ${auth.bearerToken}` };
}
