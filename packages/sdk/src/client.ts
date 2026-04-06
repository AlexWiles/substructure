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

// ── Clients ────────────────────────────────────────────────────────────────

export { BackendClient } from "./backend-client";
export { FrontendClient } from "./frontend-client";
export { WorkerClient } from "./worker-client";
export { AdminClient } from "./admin-client";
export { UserClient } from "./user-client";
export { BaseClient, type RequestOptions } from "./base";

// ── Worker (server) ────────────────────────────────────────────────────────

export { Worker } from "./worker";
export type { FetchHandlerOptions } from "./worker";

// ── Substructure ───────────────────────────────────────────────────────────

export { Substructure, RunStream } from "./substructure";
export type { SubstructureConfig, SubmitRequest as SubstructureSubmitRequest, TurnResult } from "./substructure";

// ── Runtime ────────────────────────────────────────────────────────────────

export { InProcessRuntime } from "./runtime";
export type { NativeRuntime, InProcessRuntimeOptions } from "./runtime";

// ── Webhook ────────────────────────────────────────────────────────────────

export { verifyWebhookSignature, WebhookVerificationError } from "./webhook";

// ── Types ──────────────────────────────────────────────────────────────────

export { contentText } from "./types";
export type {
  ClientIdentity,
  RetryPolicy,
  Message,
  Content,
  ContentPart,
  Role,
  ToolCall,
  LlmTool,
  LlmRequest,
  LlmResponse,
  ToolResult,
  ClientAction,
  ClientPayload,
  DecisionTrigger,
  WorkerAction,
  WorkerDecisionRequestWire,
  Event,
  EventPayload,
  SessionStatus,
  SessionState,
} from "./types";
