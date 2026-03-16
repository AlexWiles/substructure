export interface RuntimeOptions {
  /** Number of concurrent LLM handler tasks (default: 4) */
  llmPoolSize?: number;
}

export class JsRuntime {
  /**
   * Register a JavaScript function as a worker for the given tenant and agent IDs.
   *
   * The callback receives a decision as a JSON string and must return a JSON string
   * with `{ actions: [...], state: "..." }`.
   *
   * @example
   * ```ts
   * rt.registerWorker('default', ['weather-agent'], async (decisionJson) => {
   *   const decision = JSON.parse(decisionJson)
   *   // handle decision...
   *   return JSON.stringify({ actions: [...], state: '' })
   * })
   * ```
   */
  registerWorker(
    tenantId: string,
    agentIds: string[],
    callback: (decision: string) => Promise<string>
  ): Promise<void>;

  /**
   * Send a message to an agent session and return events as JSON strings.
   */
  sendMessage(
    sessionId: string,
    tenantId: string,
    agentId: string,
    content: string
  ): Promise<string[]>;

  /**
   * Shut down the runtime and all worker loops.
   */
  shutdown(): Promise<void>;
}
