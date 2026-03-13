import type {
  WorkerDecisionRequestWire,
  WorkerAction,
  SubmitRequest,
  SpanContext,
} from "./types";

export interface DecisionResult {
  actions: WorkerAction[];
  state: string;
}

export type DecisionHandler = (
  request: WorkerDecisionRequestWire
) => Promise<DecisionResult>;

export class Worker {
  private handler: DecisionHandler;

  constructor(handler: DecisionHandler) {
    this.handler = handler;
  }

  async handleDecision(request: WorkerDecisionRequestWire): Promise<SubmitRequest> {
    const result = await this.handler(request);

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
