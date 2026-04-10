// ── Default export ────────────────────────────────────────────────────────

export { Substructure as default, Substructure } from "./substructure";
export type { AgentOptions, AgentFactory, EmbeddedOptions } from "./substructure";

// ── RunStream ─────────────────────────────────────────────────────────────

export { RunStream } from "./run-stream";
export type { TurnResult } from "./run-stream";

// ── Worker & Agent ────────────────────────────────────────────────────────

export { HandlerBuilder } from "./worker";
export type {
    Handler,
    AgentRequest,
    AgentResponse,
    MiddlewareFn,
    Next,
    FetchHandlerOptions,
    StateContributor,
    DecisionHandler,
    DecisionResult,
} from "./worker";

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
} from "./middleware";
export type {
    ToolFn,
    ToolDef,
    SubAgentTrack,
    LlmLoopSelection,
    ToolSelector,
    MessageSelector,
    SystemMessageSelector,
} from "./middleware";

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
