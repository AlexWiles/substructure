// ── Default export ────────────────────────────────────────────────────────

export type { AgentFactory, AgentOptions, EmbeddedOptions } from "./substructure";
export { Substructure as default, Substructure } from "./substructure";

// ── Turns ────────────────────────────────────────────────────────────────

export type { SessionScope, TurnResult } from "./types";

// ── Worker & Agent ────────────────────────────────────────────────────────

export type {
    AgentRequest,
    AgentResponse,
    DecisionHandler,
    DecisionResult,
    FetchHandlerOptions,
    Handler,
    MiddlewareFn,
    Next,
    StateContributor,
} from "./worker";
export { HandlerBuilder } from "./worker";

// ── Middleware ─────────────────────────────────────────────────────────────

export type {
    ActionDef,
    ActionHandlerResult,
    LlmLoopSelection,
    Logger,
    LoggingOptions,
    LogLevel,
    MessageSelector,
    StateSliceMw,
    SubAgentTrack,
    SystemMessageSelector,
    ToolDef,
    ToolFn,
    ToolInput,
    ToolSelector,
} from "./middleware";
export {
    action,
    actions,
    DEFAULT_RETRY,
    jsonState,
    llmLoop,
    logging,
    messageHistory,
    middleware,
    prependHistoryToLlmCalls,
    stateSlice,
    subAgents,
    systemMessage,
    tool,
    tools,
    triggerToMessage,
} from "./middleware";

// ── Webhook ───────────────────────────────────────────────────────────────

export { verifyWebhookSignature, WebhookVerificationError } from "./webhook";

// ── Types ─────────────────────────────────────────────────────────────────

export type {
    ClientAction,
    ClientIdentity,
    ClientPayload,
    Content,
    ContentPart,
    DecisionTrigger,
    Event,
    EventPayload,
    LlmRequest,
    LlmResponse,
    LlmTool,
    Message,
    RetryPolicy,
    Role,
    SessionState,
    SessionStatus,
    SubmitToolCallResultRequest,
    SubmitToolCallResultResponse,
    ToolCall,
    ToolResult,
    WorkerAction,
    WorkerDecisionRequestWire,
    WorkerIdentity,
} from "./types";
export { contentText } from "./types";
