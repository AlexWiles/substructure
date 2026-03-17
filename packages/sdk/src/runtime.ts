import type {
  Event,
  SendMessageRequest,
  Uuid,
  WorkerDecisionRequestWire,
} from "./types";

// ── Runtime interface ────────────────────────────────────────────────────────

/**
 * Minimal interface that the NAPI JsRuntime must satisfy.
 * This avoids a hard dependency on @substructure.ai/runtime.
 */
export interface NativeRuntime {
  registerWorker(
    tenantId: string,
    agentIds: string[],
    callback: (decision: string) => Promise<string>,
  ): Promise<void>;
  sendMessage(
    sessionId: string,
    tenantId: string,
    agentId: string,
    content: string,
  ): AsyncGenerator<string, void, unknown>;
  shutdown(): Promise<void>;
}

// ── HasHandler (re-exported from worker) ─────────────────────────────────────

export interface DecisionResult {
  actions: unknown[];
  state: string;
}

export type DecisionHandler = (
  request: WorkerDecisionRequestWire,
) => Promise<DecisionResult>;

export interface HasHandler {
  readonly id: string;
  handler(): DecisionHandler;
}

// ── InProcessRuntime ─────────────────────────────────────────────────────────

export interface InProcessRuntimeOptions {
  /**
   * The native runtime instance from @substructure.ai/runtime.
   *
   * ```ts
   * import { JsRuntime } from '@substructure.ai/runtime'
   * const native = new JsRuntime()
   * const runtime = new InProcessRuntime({ native })
   * ```
   */
  native: NativeRuntime;
}

/**
 * Wraps the NAPI runtime into the same interface as the HTTP clients.
 *
 * ```ts
 * import { JsRuntime } from '@substructure.ai/runtime'
 * import { InProcessRuntime } from '@substructure.ai/sdk/runtime'
 * import { Worker, Agent } from '@substructure.ai/sdk/agent'
 *
 * const runtime = new InProcessRuntime({ native: new JsRuntime() })
 * const worker = Worker.from(myAgent)
 * await runtime.register(worker, 'default')
 *
 * for await (const event of runtime.sendMessage({ agent_id: 'my-agent', message: 'hello' })) {
 *   console.log(event.payload.type)
 * }
 * ```
 */
export class InProcessRuntime {
  private native: NativeRuntime;

  constructor(options: InProcessRuntimeOptions) {
    this.native = options.native;
  }

  /**
   * Register a Worker (or set of agents) with the in-process runtime.
   * This connects the worker's decision handlers directly — no HTTP.
   */
  async register(
    worker: { agentIds: string[]; handleDecision: (request: WorkerDecisionRequestWire) => Promise<unknown> },
    tenantId: string,
  ): Promise<void> {
    await this.native.registerWorker(
      tenantId,
      worker.agentIds,
      async (decisionJson: string) => {
        const request: WorkerDecisionRequestWire = JSON.parse(decisionJson);
        const result = await worker.handleDecision(request);
        return JSON.stringify(result);
      },
    );
  }

  /**
   * Send a message — same interface as UserClient.sendMessage().
   */
  async *sendMessage(request: SendMessageRequest): AsyncGenerator<Event> {
    const sessionId = request.session_id ?? crypto.randomUUID();
    const tenantId = request.tenant_id ?? "default";

    for await (const json of this.native.sendMessage(
      sessionId,
      tenantId,
      request.agent_id,
      request.message,
    )) {
      yield JSON.parse(json) as Event;
    }
  }

  /**
   * Shut down the runtime and all worker loops.
   */
  async shutdown(): Promise<void> {
    await this.native.shutdown();
  }
}
