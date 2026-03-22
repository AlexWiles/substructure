import { BaseClient } from "./base";
import type {
  Event,
  SendMessageRequest,
} from "./types";

export class UserClient extends BaseClient {
  async *sendMessage(request: SendMessageRequest): AsyncGenerator<Event> {
    yield* this.streamNdjson("/api/machine/sessions/send", request);
  }
}
