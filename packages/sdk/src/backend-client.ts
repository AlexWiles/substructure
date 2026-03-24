import { WorkerClient } from "./worker-client";
import type {
  ClientPayload,
  Decimal,
  Event,
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

export interface BackendSubmitRequest {
  agentId: string;
  payload: ClientPayload;
  auth: {
    tenant_id: string;
    sub: string;
    attrs?: Record<string, string>;
  };
  sessionId?: string;
  turnId?: string;
}

export interface TurnResult {
  turnId: string;
  data: unknown;
  cost: Decimal;
  tokenUsage: Record<string, number>;
}

export class RunStream {
  readonly result: Promise<TurnResult>;
  private events: Event[] = [];
  private resolveResult!: (r: TurnResult) => void;
  private rejectResult!: (e: Error) => void;
  private source: AsyncGenerator<Event>;

  constructor(source: AsyncGenerator<Event>) {
    this.source = source;
    this.result = new Promise<TurnResult>((resolve, reject) => {
      this.resolveResult = resolve;
      this.rejectResult = reject;
    });
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<Event> {
    try {
      for await (const event of this.source) {
        this.events.push(event);
        yield event;
      }
      const tc = this.events.findLast(
        (e): e is Event & {
          payload: {
            type: "turn.completed";
            turn_id: string;
            data: unknown;
            turn_cost?: Decimal;
            turn_token_usage?: Record<string, number>;
          };
        } => e.payload.type === "turn.completed",
      );
      if (tc) {
        this.resolveResult({
          turnId: tc.payload.turn_id,
          data: tc.payload.data,
          cost: tc.payload.turn_cost ?? "0",
          tokenUsage: tc.payload.turn_token_usage ?? {},
        });
      } else {
        this.rejectResult(new Error("stream ended without turn.completed"));
      }
    } catch (err) {
      this.rejectResult(err instanceof Error ? err : new Error(String(err)));
      throw err;
    }
  }
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

  submit(request: BackendSubmitRequest): RunStream {
    const stream = this.worker.submitClientPayload({
      agent_id: request.agentId,
      payload: request.payload,
      auth: request.auth,
      session_id: request.sessionId,
      turn_id: request.turnId,
    });
    return new RunStream(stream);
  }
}
