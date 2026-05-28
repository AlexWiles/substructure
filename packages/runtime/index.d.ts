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
    ): Promise<SubmitPayloadResult>;

    submitToolCallResult(
        sessionId: string,
        tenantId: string,
        toolCallId: string,
        attempt: number,
        resultJson: string | undefined,
        errorMessage: string | undefined,
        retryable: boolean | undefined,
    ): Promise<void>;

    streamSession(sessionId: string, turnId?: string, sequenceAfter?: number): AsyncGenerator<string, void, unknown>;

    shutdown(): Promise<void>;
}
