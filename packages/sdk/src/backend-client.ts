import { WorkerClient } from "./worker-client";
import type {
  MintClientTokenRequest,
  RegisterRequest,
  RegisterResponse,
  SubmitRequest,
  SubmitResponse,
} from "./types";

export interface BackendClientOptions {
  url: string;
  apiKey: string;
}

export interface IssueClientTokenRequest {
  tenantId: string;
  sub: string;
  attrs?: Record<string, string>;
  ttlSeconds?: number;
}

export interface IssueClientTokenResponse {
  token: string;
  expiresAt: number;
}

export class BackendClient {
  private worker: WorkerClient;

  constructor(options: BackendClientOptions) {
    this.worker = new WorkerClient({
      baseUrl: options.url,
      headers: { Authorization: `Bearer ${options.apiKey}` },
    });
  }

  async mintClientToken(request: IssueClientTokenRequest): Promise<IssueClientTokenResponse> {
    const response = await this.worker.mintClientToken({
      tenant_id: request.tenantId,
      sub: request.sub,
      attrs: request.attrs,
      ttl_seconds: request.ttlSeconds,
    } as MintClientTokenRequest);
    return { token: response.token, expiresAt: response.expires_at };
  }

  async registerWorker(request: RegisterRequest): Promise<RegisterResponse> {
    return this.worker.register(request);
  }

  async submitWorkerDecision(request: SubmitRequest): Promise<SubmitResponse> {
    return this.worker.submit(request);
  }
}
