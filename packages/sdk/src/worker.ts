import type {
  WorkerDecisionRequestWire,
  WorkerAction,
  SubmitRequest,
  SpanContext,
  RegisterResponse,
} from "./types";
import { WorkerClient } from "./worker-client";
import type { NativeRuntime } from "./runtime";

export interface DecisionResult {
  actions: WorkerAction[];
  state: string;
}

export type DecisionHandler = (
  request: WorkerDecisionRequestWire
) => Promise<DecisionResult>;

export interface HasHandler {
  readonly id: string;
  handler(): DecisionHandler;
}

export interface HttpRegisterOptions {
  runtimeUrl: string;
  tenantId: string;
  endpointUrl: string;
}

export interface InProcessRegisterOptions {
  runtime: NativeRuntime;
  tenantId: string;
}

export type RegisterOptions = HttpRegisterOptions | InProcessRegisterOptions;

function isInProcess(opts: RegisterOptions): opts is InProcessRegisterOptions {
  return "runtime" in opts;
}

export class Worker {
  readonly agentIds: string[];
  private handlers: Map<string, DecisionHandler>;

  constructor(handler: DecisionHandler);
  constructor(agents: HasHandler[]);
  constructor(arg: DecisionHandler | HasHandler[]) {
    if (typeof arg === "function") {
      this.agentIds = [];
      this.handlers = new Map([["*", arg]]);
    } else {
      this.agentIds = arg.map((a) => a.id);
      this.handlers = new Map(arg.map((a) => [a.id, a.handler()]));
    }
  }

  static from(...agents: HasHandler[]): Worker {
    return new Worker(agents);
  }

  /**
   * Returns a fetch-compatible handler: (Request) => Promise<Response>.
   * Works with Bun.serve, Deno.serve, Cloudflare Workers, or any Node adapter.
   */
  fetchHandler(): (req: Request) => Promise<Response> {
    return async (req: Request) => {
      const decision = (await req.json()) as WorkerDecisionRequestWire;
      const submit = await this.handleDecision(decision);
      return Response.json(submit);
    };
  }

  /**
   * Register this worker with a runtime.
   *
   * HTTP mode (with a remote server):
   * ```ts
   * await worker.register({
   *   runtimeUrl: 'http://localhost:8080',
   *   tenantId: 'default',
   *   endpointUrl: 'http://localhost:4444',
   * })
   * ```
   *
   * In-process mode (with NAPI runtime):
   * ```ts
   * import { JsRuntime } from '@substructure.ai/runtime'
   * await worker.register({
   *   runtime: new JsRuntime(),
   *   tenantId: 'default',
   * })
   * ```
   */
  async register(options: RegisterOptions): Promise<RegisterResponse> {
    if (isInProcess(options)) {
      const self = this;
      await options.runtime.registerWorker(
        options.tenantId,
        this.agentIds,
        async (decisionJson: string) => {
          const request: WorkerDecisionRequestWire = JSON.parse(decisionJson);
          const result = await self.handleDecision(request);
          return JSON.stringify(result);
        },
      );
      return { ok: true };
    }

    const client = new WorkerClient({ baseUrl: options.runtimeUrl });
    return client.register({
      tenant_id: options.tenantId,
      agent_ids: this.agentIds,
      transport_type: "http",
      config: { endpoint_url: options.endpointUrl },
    });
  }

  async handleDecision(request: WorkerDecisionRequestWire): Promise<SubmitRequest> {
    const handler = this.handlers.get(request.agent_id) ?? this.handlers.get("*");
    if (!handler) {
      throw new Error(`No handler for agent: ${request.agent_id}`);
    }

    const result = await handler(request);

    return {
      session_id: request.session_id,
      tenant_id: request.tenant_id,
      decision_id: request.decision_id,
      actions: result.actions,
      state: result.state,
      span: childSpan(request.span, "worker_submit"),
    };
  }
}

function randomHex(bytes: number): string {
  const buf = new Uint8Array(bytes);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}

function childSpan(parent: SpanContext, name: string): SpanContext {
  return {
    trace_id: parent.trace_id,
    span_id: randomHex(8),
    parent_span_id: parent.span_id,
    trace_flags: parent.trace_flags,
    trace_state: parent.trace_state,
    name,
  };
}
