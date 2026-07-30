export interface RuntimeOptions {
    /** SQLite database path */
    db: string;
    /** OpenRouter API base URL (default: "https://openrouter.ai/api") */
    openrouterBaseUrl?: string;
    /** OpenRouter API key */
    openrouterApiKey?: string;
    /** Number of concurrent LLM handler tasks (default: 4) */
    llmPoolSize?: number;
}

export interface SubmitPayloadResult {
    sessionId: string;
    turnId: string;
    /** The turn was taken but has not started: another turn holds the session. */
    queued: boolean;
}

export class EmbeddedRuntime {
    constructor(options: RuntimeOptions);

    registerWorker(
        tenantId: string,
        agentIds: string[],
        callback: (decision: string) => Promise<string>,
    ): Promise<void>;

    submitPayload(
        sessionId: string,
        agentId: string,
        payloadJson: string,
        identityJson: string,
        turnId?: string,
        queue?: boolean,
    ): Promise<SubmitPayloadResult>;

    settleEffect(
        sessionId: string,
        tenantId: string,
        kind: string,
        id: string,
        attempt: number | undefined,
        resultJson: string | undefined,
        responseJson: string | undefined,
        errorMessage: string | undefined,
        retryable: boolean | undefined,
    ): Promise<void>;

    streamSession(
        tenantId: string,
        sessionId: string,
        turnId?: string,
        afterSeq?: number,
    ): AsyncGenerator<string, void, unknown>;

    emitTokenDelta(deltaJson: string): Promise<void>;

    shutdown(): Promise<void>;
}
