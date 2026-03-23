import { BaseClient } from "./base";
import type {
  Event,
  SubmitPayloadRequest,
} from "./types";

export class UserClient extends BaseClient {
  async *submitPayload(request: SubmitPayloadRequest): AsyncGenerator<Event> {
    yield* this.streamNdjson("/api/client/sessions/submit", request);
  }
}
