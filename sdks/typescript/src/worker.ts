import type {
  WorkerDecisionRequestWire,
  WorkerAction,
  SubmitRequest,
  SpanContext,
  RegisterResponse,
} from "./types";
import { WorkerClient } from "./worker-client";

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
   * Registers this worker's agents with the runtime.
   */
  async register(options: {
    runtimeUrl: string;
    tenantId: string;
    endpointUrl: string;
  }): Promise<RegisterResponse> {
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
