import type { Event, Artifact, Decimal, TurnCompleted } from "./types";
import type { NativeRuntime } from "./runtime";
import type { Handler } from "./worker";
import { Worker } from "./worker";
import { UserClient } from "./user-client";
import { WorkerClient } from "./worker-client";

export { Agent } from "./agent";
export type { AgentOptions, LlmConfig } from "./agent";
export { retry, Retry } from "./types";
export { defineHandler, withJsonState, withState, withMessages, withAgentLoop, withLogging, subAgent, tool } from "./worker";
export type { HandlerContext, HandlerResult, Handler, MiddlewareFn, Next, ToolDef, ToolFn, Composable } from "./worker";

// ── RunStream ────────────────────────────────────────────────────────────────

export interface TurnResult {
  turnId: string;
  artifacts: Artifact[];
  cost: Decimal;
  tokenUsage: Record<string, number>;
}

export class RunStream {
  readonly result: Promise<TurnResult>;
  private events: Event[] = [];
  private resolveResult!: (r: TurnResult) => void;
  private rejectResult!: (e: Error) => void;
  private source: AsyncGenerator<Event>;

  constructor(source: AsyncGenerator<Event>) {
    this.source = source;
    this.result = new Promise<TurnResult>((resolve, reject) => {
      this.resolveResult = resolve;
      this.rejectResult = reject;
    });
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<Event> {
    try {
      for await (const event of this.source) {
        this.events.push(event);
        yield event;
      }
      const tc = this.events.findLast(
        (e): e is Event & { payload: TurnCompleted } =>
          e.payload.type === "turn.completed",
      );
      if (tc) {
        this.resolveResult({
          turnId: tc.payload.turn_id,
          artifacts: tc.payload.artifacts ?? [],
          cost: tc.payload.turn_cost ?? "0",
          tokenUsage: tc.payload.turn_token_usage ?? {},
        });
      } else {
        this.rejectResult(new Error("stream ended without turn.completed"));
      }
    } catch (err) {
      this.rejectResult(err instanceof Error ? err : new Error(String(err)));
      throw err;
    }
  }
}

// ── Config ──────────────────────────────────────────────────────────────────

export interface LocalConfig {
  /** SQLite database path */
  db: string;
  /** Handler returned by defineHandler().use(...).handle(...) */
  handler: Handler;
  /** OpenRouter API base URL (default: "https://openrouter.ai/api") */
  openrouterBaseUrl?: string;
  /** OpenRouter API key */
  openrouterApiKey?: string;
  /** Number of concurrent LLM handler tasks (default: 4) */
  llmPoolSize?: number;
}

export interface RemoteConfig {
  /** Substructure server URL */
  url: string;
  /** Handler returned by defineHandler().use(...).handle(...) */
  handler: Handler;
  /** Worker endpoint URL (required for HTTP push transport) */
  workerUrl?: string;
}

export type SubstructureConfig = LocalConfig | RemoteConfig;

function isRemote(config: SubstructureConfig): config is RemoteConfig {
  return "url" in config;
}

// ── Substructure ────────────────────────────────────────────────────────────

export class Substructure {
  private config: SubstructureConfig;
  private runtime: NativeRuntime | null = null;
  private userClient: UserClient | null = null;
  private workerClient: WorkerClient | null = null;
  private worker: Worker;
  private ready: Promise<void>;

  constructor(config: SubstructureConfig) {
    this.config = config;
    this.worker = new Worker(config.handler);

    if (isRemote(config)) {
      this.userClient = new UserClient({ baseUrl: config.url });
      this.workerClient = new WorkerClient({ baseUrl: config.url });
    }

    this.ready = this.register();
  }

  private async getRuntime(): Promise<NativeRuntime> {
    if (this.runtime) return this.runtime;
    if (isRemote(this.config)) {
      throw new Error("Cannot get native runtime in remote mode");
    }
    const { JsRuntime } = await import("@substructure.ai/runtime");
    this.runtime = new JsRuntime(this.config);
    return this.runtime;
  }

  private async register(): Promise<void> {
    const tenantId = "default";

    if (isRemote(this.config)) {
      if (!this.config.workerUrl) {
        throw new Error("workerUrl is required for remote mode registration");
      }
      await this.workerClient!.register({
        tenant_id: tenantId,
        agent_ids: this.worker.agentIds,
        transport_type: "http",
        config: { endpoint_url: this.config.workerUrl },
      });
    } else {
      const runtime = await this.getRuntime();
      await this.worker.register(runtime, tenantId);
    }
  }

  run(
    agentId: string,
    message: string,
    options?: { sessionId?: string; tenantId?: string; turnId?: string },
  ): RunStream {
    const sessionId = options?.sessionId ?? crypto.randomUUID();
    const tenantId = options?.tenantId ?? "default";
    const turnId = options?.turnId;

    const self = this;
    async function* generate(): AsyncGenerator<Event> {
      await self.ready;
      if (isRemote(self.config)) {
        yield* self.userClient!.sendMessage({
          agent_id: agentId,
          message,
          session_id: sessionId,
          tenant_id: tenantId,
          turn_id: turnId,
        });
      } else {
        const runtime = await self.getRuntime();
        for await (const json of runtime.sendMessage(sessionId, tenantId, agentId, message, turnId)) {
          yield JSON.parse(json) as Event;
        }
      }
    }

    return new RunStream(generate());
  }

  fetchHandler(): (req: Request) => Promise<Response> {
    return this.worker.fetchHandler();
  }

  async shutdown(): Promise<void> {
    if (this.runtime) {
      await this.runtime.shutdown();
    }
  }
}
