// ── Default export ────────────────────────────────────────────────────────

export type { AgentFactory, AgentOptions } from "./substructure";
export { Substructure as default, Substructure } from "./substructure";

// ── Turns ────────────────────────────────────────────────────────────────

export type { SessionScope, TurnResult } from "./types";

// ── Worker & Agent ────────────────────────────────────────────────────────

export type {
    AgentContext,
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
    Deferred,
    LlmToolLoopSelection,
    Logger,
    LoggingOptions,
    LogLevel,
    MessageHistoryOptions,
    MessageSelector,
    StateSliceMw,
    SystemMessageSelector,
    ToolDef,
    ToolExecutionContext,
    ToolFn,
    ToolInput,
    ToolSelector,
} from "./middleware";
export {
    action,
    actions,
    DEFAULT_RETRY,
    DEFERRED,
    jsonState,
    llmToolLoop,
    logging,
    messageHistory,
    middleware,
    prependHistoryToLlmCalls,
    stateSlice,
    subAgents,
    tool,
    tools,
    triggerToMessage,
    triggerToMessages,
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
    LlmHandler,
    LlmRequest,
    LlmResponse,
    LlmTokenDelta,
    LlmTool,
    Message,
    PersistedEvent,
    RetryPolicy,
    Role,
    SessionState,
    SessionStatus,
    StreamPart,
    SubmitToolCallFailure,
    SubmitToolCallResultArgs,
    SubmitToolCallResultOutcome,
    SubmitToolCallResultRequest,
    SubmitToolCallResultResponse,
    SubmitToolCallResultTarget,
    SubmitToolCallSuccess,
    ToolCall,
    ToolResult,
    WorkerAction,
    WorkerDecisionRequestWire,
    WorkerIdentity,
} from "./types";
export { contentText, isTokenDelta, persistedOnly } from "./types";
