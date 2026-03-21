import { BaseClient } from "./base";
import type {
  WorkerAuthOptions,
  SubmitRequest,
  SubmitResponse,
  RegisterRequest,
  RegisterResponse,
} from "./types";

export class WorkerClient extends BaseClient {
  async submit(request: SubmitRequest, auth?: WorkerAuthOptions): Promise<SubmitResponse> {
    return this.post("/workers/submit", request, { headers: buildWorkerAuthHeaders(auth) });
  }

  async register(request: RegisterRequest, auth?: WorkerAuthOptions): Promise<RegisterResponse> {
    return this.post("/workers/register", request, { headers: buildWorkerAuthHeaders(auth) });
  }
}

function buildWorkerAuthHeaders(auth?: WorkerAuthOptions): Record<string, string> | undefined {
  if (!auth) {
    return undefined;
  }
  return { Authorization: `Bearer ${auth.bearerToken}` };
}
