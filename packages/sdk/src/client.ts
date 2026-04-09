// ── Default export ────────────────────────────────────────────────────────

export { Substructure as default, Substructure } from "./substructure";
export type { AgentOptions, AgentFactory, EmbeddedOptions, SubmitRequest as SubstructureSubmitRequest } from "./substructure";
export { EmbeddedInstance } from "./substructure";

// ── RunStream ─────────────────────────────────────────────────────────────

export { RunStream } from "./run-stream";
export type { TurnResult } from "./run-stream";

// ── Clients ───────────────────────────────────────────────────────────────

export { BackendClient } from "./backend-client";
export type { BackendClientOptions, BackendSubmitRequest, IssueClientTokenRequest, IssueClientTokenResponse } from "./backend-client";
export { FrontendClient } from "./frontend-client";
export type { FrontendClientOptions, FrontendSubmitRequest } from "./frontend-client";
export { WorkerClient } from "./worker-client";
export { AdminClient } from "./admin-client";
export { UserClient } from "./user-client";
export { BaseClient, type RequestOptions } from "./base";

// ── Worker & Agent ────────────────────────────────────────────────────────

export { Worker, defineAgent, HandlerBuilder } from "./worker";
export type { Handler, AgentRequest, AgentResponse, MiddlewareFn, Next, FetchHandlerOptions, StateContributor, DecisionHandler, DecisionResult } from "./worker";

// ── Middleware ─────────────────────────────────────────────────────────────

export {
    state,
    stateSlice,
    tool,
    logging,
    messageHistory,
    systemMessage,
    tools,
    llmLoop,
    subAgents,
} from "./worker";
export type { ToolFn, ToolDef, SubAgentTrack, LlmLoopSelection, ToolSelector, MessageSelector, SystemMessageSelector } from "./worker";

// ── Runtime ───────────────────────────────────────────────────────────────

export { InProcessRuntime } from "./runtime";
export type { NativeRuntime, InProcessRuntimeOptions } from "./runtime";

// ── Webhook ───────────────────────────────────────────────────────────────

export { verifyWebhookSignature, WebhookVerificationError } from "./webhook";

// ── Types ─────────────────────────────────────────────────────────────────

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
