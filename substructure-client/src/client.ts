import type { BaseClientOptions } from "./base";
import { WorkerClient } from "./worker-client";
import { AdminClient } from "./admin-client";
import { UserClient } from "./user-client";

export type { BaseClientOptions as ClientOptions };

/**
 * Combined client for convenience — wraps all three scoped clients.
 * Prefer using WorkerClient, AdminClient, or UserClient directly.
 */
export class Client {
  readonly worker: WorkerClient;
  readonly admin: AdminClient;
  readonly user: UserClient;

  constructor(options: BaseClientOptions) {
    this.worker = new WorkerClient(options);
    this.admin = new AdminClient(options);
    this.user = new UserClient(options);
  }
}

export { WorkerClient } from "./worker-client";
export { AdminClient } from "./admin-client";
export { UserClient } from "./user-client";
export { BaseClient, type RequestOptions } from "./base";
