import { UserClient } from "./user-client";
import { RunStream } from "./run-stream";
import type { ClientPayload } from "./types";

export { RunStream } from "./run-stream";
export type { TurnResult } from "./run-stream";

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
