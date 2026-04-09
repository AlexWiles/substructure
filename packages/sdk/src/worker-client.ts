import { BaseClient } from "./base";
import type {
  Event,
  MachineSubmitPayloadRequest,
  MintClientTokenRequest,
  MintClientTokenResponse,
  WorkerAuthOptions,
  SubmitRequest,
  SubmitResponse,
  RegisterRequest,
  RegisterResponse,
} from "./types";

export class WorkerClient extends BaseClient {
  async submit(request: SubmitRequest, auth?: WorkerAuthOptions): Promise<SubmitResponse> {
    return this.post("/api/machine/workers/submit", request, { headers: buildWorkerAuthHeaders(auth) });
  }

  async register(request: RegisterRequest, auth?: WorkerAuthOptions): Promise<RegisterResponse> {
    return this.post("/api/machine/workers/register", request, { headers: buildWorkerAuthHeaders(auth) });
  }

  async mintClientToken(request: MintClientTokenRequest, auth?: WorkerAuthOptions): Promise<MintClientTokenResponse> {
    return this.post("/api/machine/client-tokens", request, { headers: buildWorkerAuthHeaders(auth) });
  }

  async *submitClientPayload(request: MachineSubmitPayloadRequest, auth?: WorkerAuthOptions): AsyncGenerator<Event> {
    yield* this.streamSSE("/api/machine/sessions/submit", request, {
      headers: buildWorkerAuthHeaders(auth),
    });
  }
}

function buildWorkerAuthHeaders(auth?: WorkerAuthOptions): Record<string, string> | undefined {
  if (!auth) {
    return undefined;
  }
  return { Authorization: `Bearer ${auth.bearerToken}` };
}
