import { BaseClient } from "./base";
import type {
  SubmitRequest,
  SubmitResponse,
  RegisterRequest,
  RegisterResponse,
} from "./types";

export class WorkerClient extends BaseClient {
  async submit(request: SubmitRequest): Promise<SubmitResponse> {
    return this.post("/workers/submit", request);
  }

  async register(request: RegisterRequest): Promise<RegisterResponse> {
    return this.post("/workers/register", request);
  }
}
