/** Worker: HTTP poll/submit loop against a substructure2 runtime. */

import { Agent, type RunContext } from "./agent.js";
import { Client } from "./client.js";
import type {
  PollRequest,
  PollResponse,
  SubmitRequest,
  SubmitResponse,
} from "./types.js";

export class Worker {
  private url: string;
  private tenantId: string;
  private pollTimeoutMs: number;
  private agents: Map<string, Agent>;
  private running = true;

  constructor(
    public readonly client: Client,
    options: { agents: Agent[]; pollTimeoutMs?: number },
  ) {
    this.url = client.url;
    this.tenantId = client.tenantId;
    this.pollTimeoutMs = options.pollTimeoutMs ?? 30_000;
    this.agents = new Map(options.agents.map((a) => [a.name, a]));
  }

  run(): void {
    const agentIds = Array.from(this.agents.keys());

    const shutdown = () => {
      console.log("Shutdown signal received.");
      this.running = false;
    };
    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);

    this._loop(agentIds).then(
      () => console.log("Worker stopped."),
      (err) => console.error("Worker fatal error:", err),
    );
  }

  private async _loop(agentIds: string[]): Promise<void> {
    while (this.running) {
      try {
        await this._pollCycle(agentIds);
      } catch (err) {
        console.error("Poll error:", err);
        await sleep(1000);
      }
    }
  }

  private async _pollCycle(agentIds: string[]): Promise<void> {
    const pollReq: PollRequest = {
      tenant_id: this.tenantId,
      agent_ids: agentIds,
      timeout_ms: this.pollTimeoutMs,
    };

    const resp = await fetch(`${this.url}/workers/poll`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pollReq),
    });

    if (resp.status === 204) return;
    if (!resp.ok) throw new Error(`Poll failed: ${resp.status} ${resp.statusText}`);

    const poll = (await resp.json()) as PollResponse;

    const agent = this.agents.get(poll.agent_id);
    if (!agent) {
      console.error(`No agent registered for ${poll.agent_id}`);
      return;
    }

    const ctx: RunContext = {
      sessionId: poll.session_id,
      tenantId: poll.tenant_id,
      agentId: poll.agent_id,
      decisionId: poll.decision_id,
      attempts: poll.attempts ?? 0,
      span: poll.span,
    };

    const workerState = poll.worker_state
      ? base64ToBytes(poll.worker_state)
      : new Uint8Array(0);

    let actions, newState;
    try {
      [actions, newState] = await agent.decide(poll.trigger, workerState, ctx);
    } catch (err) {
      console.error(`Agent ${poll.agent_id} failed on decision ${poll.decision_id}:`, err);
      return;
    }

    const actionTypes = actions.map((a) => a.type);

    const submitReq: SubmitRequest = {
      session_id: poll.session_id,
      tenant_id: poll.tenant_id,
      decision_id: poll.decision_id,
      actions,
      state: bytesToBase64(newState),
      span: poll.span,
    };

    const submitResp = await fetch(`${this.url}/workers/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(submitReq),
    });
    if (!submitResp.ok) throw new Error(`Submit failed: ${submitResp.status}`);

    const result = (await submitResp.json()) as SubmitResponse;
    if (!result.ok) {
      console.error(`Submit rejected for decision ${poll.decision_id}: ${result.error}`);
    }
  }
}

// --- Helpers ---

function base64ToBytes(b64: string): Uint8Array {
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
