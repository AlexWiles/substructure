import type { Event } from "./types";
import type { NativeRuntime } from "./runtime";
import type { HasHandler } from "./worker";
import { Worker } from "./worker";
import { UserClient } from "./user-client";
import { WorkerClient } from "./worker-client";

export { Agent } from "./agent";
export type { AgentOptions, LlmConfig } from "./agent";

// ── Config ──────────────────────────────────────────────────────────────────

export interface LocalConfig {
  /** SQLite database path */
  db: string;
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
  private workerUrl: string | undefined;

  constructor(config: SubstructureConfig) {
    this.config = config;

    if (isRemote(config)) {
      this.workerUrl = config.workerUrl;
      this.userClient = new UserClient({ baseUrl: config.url });
      this.workerClient = new WorkerClient({ baseUrl: config.url });
    }
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

  async agents(...agents: HasHandler[]): Promise<Worker> {
    const worker = Worker.from(...agents);
    const tenantId = "default";

    if (isRemote(this.config)) {
      if (!this.workerUrl) {
        throw new Error("workerUrl is required for remote mode registration");
      }
      await this.workerClient!.register({
        tenant_id: tenantId,
        agent_ids: worker.agentIds,
        transport_type: "http",
        config: { endpoint_url: this.workerUrl },
      });
    } else {
      const runtime = await this.getRuntime();
      await worker.register({ runtime, tenantId });
    }

    return worker;
  }

  async *run(
    agentId: string,
    message: string,
    options?: { sessionId?: string; tenantId?: string },
  ): AsyncGenerator<Event> {
    const sessionId = options?.sessionId ?? crypto.randomUUID();
    const tenantId = options?.tenantId ?? "default";

    if (isRemote(this.config)) {
      yield* this.userClient!.sendMessage({
        agent_id: agentId,
        message,
        session_id: sessionId,
        tenant_id: tenantId,
      });
    } else {
      const runtime = await this.getRuntime();
      for await (const json of runtime.sendMessage(sessionId, tenantId, agentId, message)) {
        yield JSON.parse(json) as Event;
      }
    }
  }

  async shutdown(): Promise<void> {
    if (this.runtime) {
      await this.runtime.shutdown();
    }
  }
}
