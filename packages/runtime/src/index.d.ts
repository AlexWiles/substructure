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

export class JsRuntime {
  constructor(options: RuntimeOptions);

  registerWorker(
    tenantId: string,
    agentIds: string[],
    callback: (decision: string) => Promise<string>
  ): Promise<void>;

  submitPayload(
    sessionId: string,
    tenantId: string,
    agentId: string,
    payloadJson: string,
    turnId?: string,
  ): AsyncGenerator<string, void, unknown>;

  shutdown(): Promise<void>;
}
