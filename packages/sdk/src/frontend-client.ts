import { UserClient } from "./user-client";
import type { ClientPayload, Decimal, Event, TurnCompleted } from "./types";

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
        (e): e is Event & { payload: TurnCompleted } => e.payload.type === "turn.completed",
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

export class FrontendClient {
  private user: UserClient;

  constructor(options: FrontendClientOptions) {
    this.user = new UserClient({
      baseUrl: options.url,
      headers: { Authorization: `Bearer ${options.token}` },
    });
  }

  submit(request: FrontendSubmitRequest): RunStream {
    const sessionId = request.sessionId ?? crypto.randomUUID();
    const turnId = request.turnId;
    const stream = this.user.submitPayload({
      agent_id: request.agentId,
      payload: request.payload,
      session_id: sessionId,
      turn_id: turnId,
    });
    return new RunStream(stream);
  }
}
