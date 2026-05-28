// ../../packages/sdk/dist/chunk-NJQM6NIJ.js
var WebhookVerificationError = class extends Error {
  constructor(message) {
    super(message);
    this.name = "WebhookVerificationError";
  }
};
async function verifyWebhookSignature(req, secret, options) {
  const timestamp = req.headers.get("x-substructure-timestamp");
  const signature = req.headers.get("x-substructure-signature");
  if (!timestamp || !signature) {
    throw new WebhookVerificationError("Missing signature headers");
  }
  const ts = parseInt(timestamp, 10);
  if (isNaN(ts)) {
    throw new WebhookVerificationError("Invalid timestamp");
  }
  const tolerance = options?.tolerance ?? 300;
  const now = Math.floor(Date.now() / 1e3);
  if (Math.abs(now - ts) > tolerance) {
    throw new WebhookVerificationError("Timestamp outside tolerance window");
  }
  const match = signature.match(/^v1=([0-9a-f]+)$/);
  if (!match) {
    throw new WebhookVerificationError("Invalid signature format");
  }
  const receivedSig = match[1];
  const body = await req.text();
  const signingPayload = `${timestamp}.${body}`;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(signingPayload));
  const expectedSig = Array.from(new Uint8Array(mac), (b) => b.toString(16).padStart(2, "0")).join("");
  if (!timingSafeEqual(receivedSig, expectedSig)) {
    throw new WebhookVerificationError("Signature mismatch");
  }
  return JSON.parse(body);
}
function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}

// ../../packages/sdk/dist/chunk-RGABIIPF.js
var DEFAULT_FALLBACK = (req) => ({ actions: [], state: req.state });
var HandlerBuilder = class {
  agentId;
  middlewares = [];
  constructor(agentId) {
    this.agentId = agentId;
  }
  use(mw) {
    this.middlewares.push(mw);
    return this;
  }
  toDecisionHandler() {
    const middlewares = this.middlewares;
    const chain = composeChain(middlewares, DEFAULT_FALLBACK);
    return async (request) => {
      const req = {
        agentId: request.agent_id,
        trigger: request.trigger,
        state: void 0,
        wire: request
      };
      const result = await chain(req);
      return {
        actions: result.actions,
        state: result.workerState ?? request.worker_state
      };
    };
  }
};
function composeChain(middlewares, handle) {
  let fn = handle;
  for (let i = middlewares.length - 1; i >= 0; i--) {
    const mw = middlewares[i];
    const next = fn;
    fn = (ctx) => mw(ctx, next);
  }
  return fn;
}
var Worker = class {
  agentIds;
  handlers;
  constructor(agents) {
    this.handlers = /* @__PURE__ */ new Map();
    for (const handler of agents) {
      this.handlers.set(handler.agentId, handler.toDecisionHandler());
    }
    this.agentIds = [...this.handlers.keys()];
  }
  async register(runtime, tenantId) {
    const self = this;
    await runtime.registerWorker(tenantId, this.agentIds, async (decisionJson) => {
      const request = JSON.parse(decisionJson);
      const submit = await self.handleDecision(request);
      return JSON.stringify(submit);
    });
  }
  /**
   * Returns a fetch-compatible handler: (Request) => Promise<Response>.
   * Works with Bun.serve, Deno.serve, Cloudflare Workers, or any Node adapter.
   *
   * When `options.signingSecret` is provided, incoming requests are verified
   * against the HMAC-SHA256 signature in the `X-Substructure-Signature` header.
   */
  fetchHandler(options) {
    return async (req) => {
      let decision;
      if (options?.signingSecret) {
        decision = await verifyWebhookSignature(req, options.signingSecret, {
          tolerance: options.tolerance
        });
      } else {
        decision = await req.json();
      }
      const submit = await this.handleDecision(decision);
      return Response.json(submit);
    };
  }
  async handleDecision(request) {
    const handler = this.handlers.get(request.agent_id);
    if (!handler) {
      throw new Error(`No handler registered for agent: ${request.agent_id}`);
    }
    const result = await handler(request);
    return {
      session_id: request.session_id,
      decision_id: request.decision_id,
      actions: result.actions,
      state: result.state,
      span: childSpan(request.span, "worker_submit")
    };
  }
};
function randomHex(bytes) {
  const buf = new Uint8Array(bytes);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}
function childSpan(parent, name) {
  return {
    trace_id: parent.trace_id,
    span_id: randomHex(8),
    parent_span_id: parent.span_id,
    trace_flags: parent.trace_flags,
    trace_state: parent.trace_state,
    name
  };
}

// ../../packages/sdk/dist/chunk-BNMIECBU.js
var SDK_VERSION = true ? "0.1.8" : "0.0.0-dev";

// ../../packages/sdk/dist/chunk-D527RYWY.js
var SDK_VERSION_HEADER = "X-Substructure-Sdk-Version";
var BaseClient = class {
  baseUrl;
  headers;
  constructor(options) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.headers = options.headers;
  }
  mergeHeaders(headers) {
    const merged = new Headers(this.headers);
    if (headers) {
      const extra = new Headers(headers);
      for (const [k, v] of extra.entries()) {
        merged.set(k, v);
      }
    }
    merged.set(SDK_VERSION_HEADER, SDK_VERSION);
    return merged;
  }
  buildUrl(path, query) {
    const url = `${this.baseUrl}${path}`;
    if (!query) return url;
    const search = new URLSearchParams(
      Object.entries(query).filter(([, v]) => v !== void 0)
    );
    return `${url}?${search}`;
  }
  async fetch(url, init) {
    const resp = await fetch(url, init);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${init?.method ?? "GET"} ${url} failed (${resp.status}): ${text}`);
    }
    return resp;
  }
  async get(path, params, opts) {
    const resp = await this.fetch(this.buildUrl(path, params), {
      signal: opts?.signal,
      headers: this.mergeHeaders(opts?.headers)
    });
    return await resp.json();
  }
  async post(path, body, opts) {
    const resp = await this.fetch(this.buildUrl(path), {
      method: "POST",
      headers: this.mergeHeaders({ "Content-Type": "application/json", ...opts?.headers }),
      body: JSON.stringify(body),
      signal: opts?.signal
    });
    return await resp.json();
  }
  async *streamSSEGet(path, params, opts) {
    const resp = await this.fetch(this.buildUrl(path, params), {
      signal: opts?.signal,
      headers: this.mergeHeaders({ Accept: "text/event-stream", ...opts?.headers })
    });
    yield* this.readSSE(resp);
  }
  async *streamSSE(path, body, opts) {
    const resp = await this.fetch(this.buildUrl(path), {
      method: "POST",
      headers: this.mergeHeaders({
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...opts?.headers
      }),
      body: JSON.stringify(body),
      signal: opts?.signal
    });
    yield* this.readSSE(resp);
  }
  async *readSSE(resp) {
    const body = resp.body;
    if (!body) throw new Error("Response body is null");
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const messages = buf.split("\n\n");
        buf = messages.pop();
        for (const msg of messages) {
          if (!msg.trim()) continue;
          let data = "";
          for (const line of msg.split("\n")) {
            if (line.startsWith("data: ")) {
              data += line.slice(6);
            } else if (line.startsWith("data:")) {
              data += line.slice(5);
            }
          }
          if (data) {
            yield JSON.parse(data);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
};

// ../../packages/sdk/dist/chunk-SZWAKTA5.js
var WorkerClient = class extends BaseClient {
  async submit(request, auth) {
    return this.post("/api/machine/workers/submit", request, { headers: buildWorkerAuthHeaders(auth) });
  }
  async mintClientToken(request, auth) {
    return this.post("/api/machine/client-tokens", request, { headers: buildWorkerAuthHeaders(auth) });
  }
  async submitClientPayload(request, auth) {
    return this.post("/api/machine/sessions/submit", request, {
      headers: buildWorkerAuthHeaders(auth)
    });
  }
  async submitToolCallResult(sessionId2, request, auth) {
    return this.post(`/api/machine/sessions/${sessionId2}/tool-call-results`, request, {
      headers: buildWorkerAuthHeaders(auth)
    });
  }
  async *streamSessionEvents(sessionId2, params, auth) {
    yield* this.streamSSEGet(
      `/api/machine/sessions/${sessionId2}/events/stream`,
      {
        turn_id: params?.turn_id,
        sequence_after: params?.sequence_after?.toString()
      },
      { headers: buildWorkerAuthHeaders(auth) }
    );
  }
};
function buildWorkerAuthHeaders(auth) {
  if (!auth) {
    return void 0;
  }
  return { Authorization: `Bearer ${auth.bearerToken}` };
}

// ../../packages/sdk/dist/chunk-LTQU2GDW.js
var AdminClient = class extends BaseClient {
  async listSessions(params) {
    return this.get("/api/admin/sessions", {
      tenant_id: params?.tenant_id,
      top_level: params?.top_level?.toString(),
      sort: params?.sort,
      limit: params?.limit?.toString(),
      cursor: params?.cursor
    });
  }
  async getSession(sessionId2) {
    return this.get(`/api/admin/sessions/${sessionId2}`);
  }
  async getSessionEvents(sessionId2, params) {
    return this.get(`/api/admin/sessions/${sessionId2}/events`, {
      sequence_after: params?.sequence_after?.toString(),
      limit: params?.limit?.toString()
    });
  }
  async *streamSessionEvents(sessionId2, params, opts) {
    yield* this.streamSSEGet(
      `/api/admin/sessions/${sessionId2}/events/stream`,
      {
        sequence_after: params?.sequence_after?.toString(),
        limit: params?.limit?.toString()
      },
      opts
    );
  }
};

// ../../packages/sdk/dist/chunk-5BTYOZ6C.js
function toSubmitToolCallResultRequest(args) {
  if (args.result !== void 0) {
    return {
      type: "return.tool.result",
      tool_call_id: args.toolCallId,
      result: args.result,
      attempt: args.attempt
    };
  }
  return {
    type: "return.tool.error",
    tool_call_id: args.toolCallId,
    error: args.error,
    retryable: args.retryable ?? false,
    attempt: args.attempt
  };
}
async function drainToTurnResult(stream) {
  let completed;
  for await (const event of stream) {
    if (event.payload.type === "turn.completed") {
      completed = event.payload;
    }
  }
  if (!completed) {
    throw new Error("stream ended without turn.completed");
  }
  return {
    turnId: completed.turn_id,
    data: completed.data,
    cost: completed.turn_cost ?? "0",
    tokenUsage: completed.turn_token_usage ?? {}
  };
}

// ../../packages/sdk/dist/chunk-DMIUTP74.js
var DEFAULT_URL = "https://api.substructure.ai";
var BackendClient = class {
  worker;
  admin;
  constructor(options) {
    const baseUrl = options.url ?? DEFAULT_URL;
    const headers = { Authorization: `Bearer ${options.apiKey}` };
    this.worker = new WorkerClient({ baseUrl, headers });
    this.admin = new AdminClient({ baseUrl, headers });
  }
  async mintClientToken(request) {
    const response = await this.worker.mintClientToken({
      identity: request.identity,
      ttl_seconds: request.ttlSeconds
    });
    return { token: response.token, expiresAt: response.expires_at };
  }
  async submitWorkerDecision(request) {
    return this.worker.submit(request);
  }
  async submitToolCallResult(args) {
    return this.worker.submitToolCallResult(args.sessionId, toSubmitToolCallResultRequest(args));
  }
  /** Fire-and-forget: enqueue a turn, return as soon as it's accepted. */
  async startTurn(request) {
    const response = await this.worker.submitClientPayload({
      agent_id: request.agentId,
      payload: request.payload,
      identity: request.identity,
      session_id: request.sessionId,
      turn_id: request.turnId
    });
    return { sessionId: response.session_id, turnId: response.turn_id };
  }
  /** Stream events for a session. If `scope.turnId` is set, the stream is
   *  filtered to that turn and auto-closes on completion. */
  stream(scope, options) {
    return this.worker.streamSessionEvents(scope.sessionId, {
      turn_id: scope.turnId,
      sequence_after: options?.sequenceAfter
    });
  }
  /** Stream a turn to completion and return its result. Requires `scope.turnId`. */
  turnResult(scope) {
    if (!scope.turnId) {
      throw new Error("turnResult requires scope.turnId");
    }
    return drainToTurnResult(this.stream(scope));
  }
  /** List sessions for the tenant scoped by this client's API key. */
  listSessions(params) {
    return this.admin.listSessions(params);
  }
  /** Fetch the current state snapshot for a session. */
  getSession(sessionId2) {
    return this.admin.getSession(sessionId2);
  }
  /** Fetch historical events for a session. */
  sessionEvents(sessionId2, params) {
    return this.admin.getSessionEvents(sessionId2, params);
  }
  /** Stream session events as they are appended (SSE). */
  streamSessionEvents(sessionId2, params, opts) {
    return this.admin.streamSessionEvents(sessionId2, params, opts);
  }
};

// ../../packages/sdk/dist/chunk-QX4IMKVI.js
var UserClient = class extends BaseClient {
  async submitPayload(request) {
    return this.post("/api/client/sessions/submit", request);
  }
  async submitToolCallResult(sessionId2, request) {
    return this.post(`/api/client/sessions/${sessionId2}/tool-call-results`, request);
  }
  /** Stream the session's persisted events and live LLM token deltas. Both
   *  arrive as `Event` envelopes; transient deltas have
   *  `payload.type === "llm.token.delta"` and lack a `sequence`. */
  async *streamSessionEvents(sessionId2, params) {
    yield* this.streamSSEGet(`/api/client/sessions/${sessionId2}/events/stream`, {
      turn_id: params?.turn_id,
      sequence_after: params?.sequence_after?.toString()
    });
  }
};

// ../../packages/sdk/dist/chunk-JIJERYHY.js
var DEFAULT_URL2 = "https://api.substructure.ai";
var FrontendClient = class {
  user;
  constructor(options) {
    this.user = new UserClient({
      baseUrl: options.url ?? DEFAULT_URL2,
      headers: { Authorization: `Bearer ${options.token}` }
    });
  }
  /** Fire-and-forget: enqueue a turn, return as soon as it's accepted. */
  async startTurn(request) {
    const response = await this.user.submitPayload({
      agent_id: request.agentId,
      payload: request.payload,
      session_id: request.sessionId,
      turn_id: request.turnId
    });
    return { sessionId: response.session_id, turnId: response.turn_id };
  }
  async submitToolCallResult(args) {
    return this.user.submitToolCallResult(args.sessionId, toSubmitToolCallResultRequest(args));
  }
  /** Stream events for a session. If `scope.turnId` is set, the stream is
   *  filtered to that turn and auto-closes on completion. */
  stream(scope, options) {
    return this.user.streamSessionEvents(scope.sessionId, {
      turn_id: scope.turnId,
      sequence_after: options?.sequenceAfter
    });
  }
  /** Stream a turn to completion and return its result. Requires `scope.turnId`. */
  turnResult(scope) {
    if (!scope.turnId) {
      throw new Error("turnResult requires scope.turnId");
    }
    return drainToTurnResult(this.stream(scope));
  }
};

// ../../packages/sdk/dist/chunk-YND5HMFG.js
var DEFAULT_RETRY = {
  timeout_secs: 120,
  max_retries: 0,
  backoff_base_secs: 1,
  backoff_max_secs: 10
};
function decodeWorkerState(raw) {
  if (!raw || raw === "") return {};
  return JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(raw), (c) => c.charCodeAt(0))));
}
function encodeWorkerState(value) {
  return btoa(String.fromCharCode(...new TextEncoder().encode(JSON.stringify(value))));
}
function initSlice(rawState, init) {
  const state = rawState && typeof rawState === "object" ? rawState : {};
  for (const key of Object.keys(init)) {
    state[key] ??= structuredClone(init[key]);
  }
  return state;
}
function middleware(config) {
  const handler = config.handler ?? ((req, next) => next(req));
  if (!config.state) {
    return handler;
  }
  const init = config.state;
  const mw = (req, next) => {
    const state = initSlice(req.state, init);
    const typedReq = { ...req, state };
    return handler(typedReq, next);
  };
  return Object.assign(mw, { _contributes: init, _init: init });
}
function stateSlice(init) {
  return middleware({ state: init });
}
function jsonState() {
  return middleware({
    handler: async (req, next) => {
      const enriched = {
        ...req,
        state: decodeWorkerState(req.wire.worker_state)
      };
      const res = await next(enriched);
      return {
        ...res,
        workerState: encodeWorkerState(res.state)
      };
    }
  });
}
var DEFERRED = /* @__PURE__ */ Symbol.for("substructure.tool.deferred");
function tool(config) {
  return {
    name: config.name,
    description: config.description,
    parameters: config.parameters,
    execute: async (args, state, ctx) => {
      if (config.state) {
        return config.execute(
          args,
          state,
          ctx
        );
      }
      return config.execute(
        args,
        ctx
      );
    },
    retry: config.retry,
    stateSlice: config.state,
    handler: config.handler
  };
}
function action(config) {
  return {
    name: config.name,
    parameters: config.parameters,
    stateSlice: config.state,
    handler: config.handler
  };
}
function actions(defs) {
  const byName = {};
  for (const def of defs) byName[def.name] = def;
  return middleware({
    handler: async (req, next) => {
      const { trigger } = req;
      if (trigger.type !== "client.action") return next(req);
      const def = byName[trigger.name];
      if (!def) return next(req);
      const state = def.stateSlice ? initSlice(req.state, def.stateSlice._init) : req.state;
      const result = await def.handler(trigger.args ?? {}, state);
      if (Array.isArray(result)) {
        return { state: req.state, actions: result };
      }
      return next(req);
    }
  });
}
var LOG_LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };
function defaultLogger(minLevel) {
  const min = LOG_LEVELS[minLevel];
  const noop = () => {
  };
  const emit = (level) => {
    if (LOG_LEVELS[level] < min) return noop;
    return (msg, data) => {
      console.log(JSON.stringify({ level, msg, ...data, ts: (/* @__PURE__ */ new Date()).toISOString() }));
    };
  };
  return { debug: emit("debug"), info: emit("info"), warn: emit("warn"), error: emit("error") };
}
function logging(options) {
  const opts = typeof options === "string" ? { label: options } : options ?? {};
  const level = opts.level ?? "info";
  const log = opts.logger ?? defaultLogger(level);
  const label = opts.label;
  return middleware({
    handler: async (req, next) => {
      const t = req.trigger;
      const tag = t.type === "tool.execute" ? `${t.type}:${t.name}` : t.type;
      const ctx = {
        agent: req.agentId,
        session: req.wire.session_id,
        decision: req.wire.decision_id,
        trigger: tag
      };
      if (label) ctx.label = label;
      log.info("decision.start", ctx);
      log.debug("decision.trigger", { ...ctx, payload: t });
      const start = performance.now();
      try {
        const result = await next(req);
        const durationMs = Number((performance.now() - start).toFixed(1));
        const actionTypes = result.actions.map((a) => a.type);
        log.info("decision.end", { ...ctx, actions: actionTypes, durationMs });
        log.debug("decision.actions", { ...ctx, actions: result.actions, durationMs });
        return result;
      } catch (err) {
        const durationMs = Number((performance.now() - start).toFixed(1));
        log.error("decision.error", {
          ...ctx,
          error: err instanceof Error ? err.message : String(err),
          durationMs
        });
        throw err;
      }
    }
  });
}
function triggerToMessage(trigger) {
  switch (trigger.type) {
    case "user.message":
    case "llm.response":
      return trigger.message;
    case "tool.result":
      return {
        role: "tool",
        content: trigger.result.content,
        tool_call_id: trigger.result.tool_call_id,
        name: trigger.result.name
      };
    default:
      return null;
  }
}
function prependHistoryToLlmCalls(history, actions2) {
  return actions2.map(
    (action2) => action2.type === "call.llm" ? {
      ...action2,
      request: {
        ...action2.request,
        messages: [...history, ...action2.request.messages]
      }
    } : action2
  );
}
function messageHistory() {
  return middleware({
    state: { messages: [] },
    handler: async (req, next) => {
      const msg = triggerToMessage(req.trigger);
      if (msg) req.state.messages.push(msg);
      const result = await next(req);
      return {
        ...result,
        actions: prependHistoryToLlmCalls(req.state.messages, result.actions)
      };
    }
  });
}
function messageHistoryCurrentTurn() {
  return middleware({
    state: { messages: [], lastTurnId: void 0 },
    handler: async (req, next) => {
      if (req.wire.turn_id !== req.state.lastTurnId) {
        req.state.messages = [];
        req.state.lastTurnId = req.wire.turn_id;
      }
      const msg = triggerToMessage(req.trigger);
      if (msg) req.state.messages.push(msg);
      const result = await next(req);
      return {
        ...result,
        actions: prependHistoryToLlmCalls(req.state.messages, result.actions)
      };
    }
  });
}
function systemMessage(selectorOrValue) {
  const selector = typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;
  return middleware({
    handler: async (req, next) => {
      const sysMsg = { role: "system", content: selector(req.state, req) };
      const result = await next(req);
      const actions2 = result.actions.map((action2) => {
        if (action2.type !== "call.llm") return action2;
        return {
          ...action2,
          request: {
            ...action2.request,
            messages: [sysMsg, ...action2.request.messages]
          }
        };
      });
      return { ...result, actions: actions2 };
    }
  });
}
function resolveTools(input2) {
  if (Array.isArray(input2)) {
    const record = {};
    for (const def of input2) {
      record[def.name] = def;
    }
    return record;
  }
  return input2;
}
function tools(selectorOrValue) {
  const selector = typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;
  return middleware({
    state: { pendingToolCalls: [] },
    handler: async (req, next) => {
      const toolMap = resolveTools(selector(req.state, req));
      const downstream = await next(req);
      if (req.trigger.type === "tool.execute") {
        const t = toolMap[req.trigger.name];
        if (!t) {
          return {
            ...downstream,
            actions: [
              {
                type: "return.tool.error",
                tool_call_id: req.trigger.tool_call_id,
                error: `Unknown tool: ${req.trigger.name}`,
                retryable: false,
                attempt: req.trigger.attempt
              },
              ...downstream.actions
            ]
          };
        }
        try {
          let toolState;
          if (t.stateSlice) {
            toolState = initSlice(req.state, t.stateSlice._init);
          }
          const ctx = {
            sessionId: req.wire.session_id,
            toolCallId: req.trigger.tool_call_id,
            attempt: req.trigger.attempt,
            defer: () => DEFERRED
          };
          const output = await t.execute(req.trigger.arguments, toolState, ctx);
          if (output === DEFERRED) {
            return downstream;
          }
          return {
            ...downstream,
            actions: [
              {
                type: "return.tool.result",
                tool_call_id: req.trigger.tool_call_id,
                result: typeof output === "string" ? output : JSON.stringify(output),
                attempt: req.trigger.attempt
              },
              ...downstream.actions
            ]
          };
        } catch (error) {
          return {
            ...downstream,
            actions: [
              {
                type: "return.tool.error",
                tool_call_id: req.trigger.tool_call_id,
                error: error instanceof Error ? error.message : String(error),
                retryable: false,
                attempt: req.trigger.attempt
              },
              ...downstream.actions
            ]
          };
        }
      }
      if (req.trigger.type === "llm.response") {
        const toolCalls = req.trigger.message.tool_calls;
        if (toolCalls && toolCalls.length > 0) {
          const known = toolCalls.filter((tc) => tc.function.name in toolMap);
          req.state.pendingToolCalls = known.map((tc) => tc.id);
          const callToolActions = known.map((tc) => {
            const def = toolMap[tc.function.name];
            return {
              type: "call.tool",
              tool_call_id: tc.id,
              name: tc.function.name,
              arguments: tc.function.arguments,
              handler: def?.handler ?? "worker",
              retry: def?.retry ?? DEFAULT_RETRY
            };
          });
          return {
            ...downstream,
            actions: [...callToolActions, ...downstream.actions]
          };
        }
      }
      const actions2 = downstream.actions.map((action2) => {
        if (action2.type === "call.llm") {
          return {
            ...action2,
            request: {
              ...action2.request,
              tools: mergeTools(
                action2.request.tools,
                Object.entries(toolMap).map(([name, def]) => ({
                  function: { name, description: def.description, parameters: def.parameters }
                }))
              )
            }
          };
        }
        if (action2.type === "call.tool") {
          const def = toolMap[action2.name];
          if (def?.retry) {
            return { ...action2, retry: def.retry };
          }
        }
        return action2;
      });
      if (req.trigger.type === "tool.result") {
        const resultId = req.trigger.result.tool_call_id;
        req.state.pendingToolCalls = req.state.pendingToolCalls.filter((id) => id !== resultId);
        if (req.state.pendingToolCalls.length > 0) {
          return {
            ...downstream,
            actions: actions2.filter((a) => a.type !== "call.llm")
          };
        }
      }
      return { ...downstream, actions: actions2 };
    }
  });
}
function llmLoop(selectorOrValue) {
  const selector = typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;
  return middleware({
    handler: async (req, next) => {
      const selection = selector(req.state, req);
      const downstream = await next(req);
      const { trigger } = req;
      switch (trigger.type) {
        case "user.message":
        case "client.action":
        case "tool.result": {
          return {
            ...downstream,
            actions: [
              {
                type: "call.llm",
                request: {
                  ...selection.request,
                  messages: []
                },
                retry: selection.retry ?? DEFAULT_RETRY,
                stream: selection.stream ?? false
              },
              ...downstream.actions
            ]
          };
        }
        case "llm.response": {
          if (!trigger.message.tool_calls || trigger.message.tool_calls.length === 0) {
            return {
              ...downstream,
              actions: [{ type: "done", data: trigger.message.content }, ...downstream.actions]
            };
          }
        }
        default:
          return downstream;
      }
    }
  });
}
function subAgents(config) {
  const subAgentMap = {};
  for (const handler of config.agents) {
    subAgentMap[handler.agentId] = { agentId: handler.agentId };
  }
  return middleware({
    state: { subAgentTracker: {} },
    handler: async (req, next) => {
      const tracker = req.state.subAgentTracker;
      switch (req.trigger.type) {
        case "llm.response": {
          const spawnActions = [];
          const toolCalls = req.trigger.message.tool_calls ?? [];
          for (const tc of toolCalls) {
            const sub2 = subAgentMap[tc.function.name];
            if (!sub2) continue;
            const childSessionId = crypto.randomUUID();
            let message = tc.function.arguments;
            try {
              const args = JSON.parse(tc.function.arguments);
              if (typeof args?.message === "string") {
                message = args.message;
              }
            } catch {
            }
            tracker[childSessionId] = {
              toolCallId: tc.id,
              name: tc.function.name
            };
            spawnActions.push(
              {
                type: "spawn.sub_agent",
                session_id: childSessionId,
                agent_id: sub2.agentId,
                retry: config.retry ?? DEFAULT_RETRY
              },
              {
                type: "send.message",
                session_id: childSessionId,
                message: {
                  role: "user",
                  content: message
                }
              }
            );
          }
          const downstream = await next(req);
          const actions2 = [];
          for (const action2 of downstream.actions) {
            if (action2.type === "call.tool" && subAgentMap[action2.name]) {
              continue;
            }
            if (action2.type === "call.llm") {
              actions2.push({
                ...action2,
                request: {
                  ...action2.request,
                  tools: mergeTools(action2.request.tools, handlersToLlmTools(config.agents))
                }
              });
              continue;
            }
            actions2.push(action2);
          }
          return { ...downstream, actions: [...spawnActions, ...actions2] };
        }
        case "sub_agent.turn.complete": {
          const tracked = tracker[req.trigger.session_id];
          if (!tracked) {
            return next(req);
          }
          delete tracker[req.trigger.session_id];
          const content = typeof req.trigger.data === "string" ? req.trigger.data : JSON.stringify(req.trigger.data);
          const result = {
            tool_call_id: tracked.toolCallId,
            name: tracked.name,
            content,
            is_error: false
          };
          return appendToolResultToLlmCalls(
            await next({ ...req, trigger: { type: "tool.result", result } }),
            result
          );
        }
        case "sub_agent.error": {
          const tracked = tracker[req.trigger.session_id];
          if (!tracked) {
            return next(req);
          }
          delete tracker[req.trigger.session_id];
          const result = {
            tool_call_id: tracked.toolCallId,
            name: tracked.name,
            content: `Sub-agent ${req.trigger.agent_id} failed: ${req.trigger.error}`,
            is_error: true
          };
          return appendToolResultToLlmCalls(
            await next({ ...req, trigger: { type: "tool.result", result } }),
            result
          );
        }
        default: {
          const downstream = await next(req);
          const actions2 = downstream.actions.map((action2) => {
            if (action2.type !== "call.llm") return action2;
            return {
              ...action2,
              request: {
                ...action2.request,
                tools: mergeTools(action2.request.tools, handlersToLlmTools(config.agents))
              }
            };
          });
          return { ...downstream, actions: actions2 };
        }
      }
    }
  });
}
function handlersToLlmTools(handlers) {
  return handlers.map((handler) => ({
    function: {
      name: handler.agentId,
      description: `Delegate to ${handler.agentId}`,
      parameters: {
        type: "object",
        properties: {
          message: {
            type: "string",
            description: "The message to send to the agent"
          }
        },
        required: ["message"]
      }
    }
  }));
}
function appendToolResultToLlmCalls(response, result) {
  const toolMsg = {
    role: "tool",
    content: result.content,
    tool_call_id: result.tool_call_id,
    name: result.name
  };
  const actions2 = response.actions.map((action2) => {
    if (action2.type !== "call.llm") return action2;
    return {
      ...action2,
      request: {
        ...action2.request,
        messages: [...action2.request.messages, toolMsg]
      }
    };
  });
  return { ...response, actions: actions2 };
}
function mergeTools(existing, added) {
  const byName = /* @__PURE__ */ new Map();
  for (const t of existing ?? []) {
    byName.set(t.function.name, t);
  }
  for (const t of added) {
    byName.set(t.function.name, t);
  }
  return Array.from(byName.values());
}

// ../../packages/sdk/dist/chunk-JPTDPZIA.js
function createAgentFactory() {
  const factory = (options) => {
    return new HandlerBuilder(options.id);
  };
  factory.jsonState = jsonState;
  factory.stateSlice = stateSlice;
  factory.tool = tool;
  factory.action = action;
  factory.actions = actions;
  factory.logging = logging;
  factory.messageHistory = messageHistory;
  factory.messageHistoryCurrentTurn = messageHistoryCurrentTurn;
  factory.systemMessage = systemMessage;
  factory.tools = tools;
  factory.llmLoop = llmLoop;
  factory.subAgents = subAgents;
  return factory;
}
var BackendNamespace = class {
  client(options) {
    return new BackendClient(options);
  }
};
var FrontendNamespace = class {
  client(options) {
    return new FrontendClient(options);
  }
};
var EmbeddedInstance = class {
  runtime;
  worker;
  registered;
  tenantId;
  constructor(runtime, agents, tenantId) {
    this.runtime = runtime;
    this.worker = new Worker(agents);
    this.tenantId = tenantId;
    this.registered = this.worker.register(runtime, tenantId);
  }
  /** Fire-and-forget: enqueue a turn, return as soon as it's accepted. */
  async startTurn(request) {
    await this.registered;
    const identity = request.identity;
    if (!identity?.id) {
      throw new Error("startTurn.identity.id is required for embedded runtime");
    }
    const sessionId2 = request.sessionId ?? crypto.randomUUID();
    return this.runtime.submitPayload(
      sessionId2,
      request.agentId,
      JSON.stringify(request.payload),
      JSON.stringify(identity),
      request.turnId
    );
  }
  /** Stream events for a session. If `scope.turnId` is set, the stream is
   *  filtered to that turn and auto-closes on completion. */
  async *stream(scope, options) {
    for await (const json of this.runtime.streamSession(scope.sessionId, scope.turnId, options?.sequenceAfter)) {
      yield JSON.parse(json);
    }
  }
  /** Stream a turn to completion and return its result. Requires `scope.turnId`. */
  turnResult(scope) {
    if (!scope.turnId) {
      throw new Error("turnResult requires scope.turnId");
    }
    return drainToTurnResult(this.stream(scope));
  }
  /** Complete (or fail) a tool call out-of-band. Use after a tool returns
   *  `DEFERRED`. */
  async submitToolCallResult(args) {
    if (args.result !== void 0) {
      await this.runtime.submitToolCallResult(
        args.sessionId,
        this.tenantId,
        args.toolCallId,
        args.attempt,
        args.result,
        void 0,
        void 0
      );
    } else {
      await this.runtime.submitToolCallResult(
        args.sessionId,
        this.tenantId,
        args.toolCallId,
        args.attempt,
        void 0,
        args.error,
        args.retryable
      );
    }
  }
  fetchHandler(options) {
    return this.worker.fetchHandler(options);
  }
  async shutdown() {
    await this.runtime.shutdown();
  }
};
var Substructure = class {
  backend;
  frontend;
  agent;
  constructor() {
    this.backend = new BackendNamespace();
    this.frontend = new FrontendNamespace();
    this.agent = createAgentFactory();
  }
  worker(options) {
    return new Worker(options.agents);
  }
  async embedded(options) {
    const { EmbeddedRuntime } = await import("@substructure.ai/runtime");
    const runtime = new EmbeddedRuntime({
      db: options.db ?? ":memory:",
      openrouterBaseUrl: options.openrouterBaseUrl,
      openrouterApiKey: options.openrouterApiKey,
      llmPoolSize: options.llmPoolSize
    });
    const instance = new EmbeddedInstance(runtime, options.agents, options.tenantId ?? "default");
    return instance;
  }
};

// public/main.ts
var chatLog = document.getElementById("chat");
var form = document.getElementById("form");
var input = document.getElementById("input");
var sendBtn = document.getElementById("send");
var tools2 = {
  get_user_location: () => new Promise((resolve) => {
    if (!navigator.geolocation) {
      resolve({ error: "Geolocation not supported in this browser." });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
        accuracy_m: pos.coords.accuracy
      }),
      (err) => resolve({ error: err.message }),
      { timeout: 1e4 }
    );
  }),
  set_theme: ({ background, accent }) => {
    const root = document.documentElement.style;
    if (background) root.setProperty("--bg", background);
    if (accent) root.setProperty("--accent", accent);
    return { ok: true, background, accent };
  }
};
function append(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}
function showTyping() {
  const div = document.createElement("div");
  div.className = "msg assistant typing";
  div.innerHTML = "<span></span><span></span><span></span>";
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}
async function fetchToken() {
  const r = await fetch("/token", { method: "POST" });
  if (!r.ok) throw new Error(`/token failed: ${r.status} ${await r.text()}`);
  return await r.json();
}
var sub = new Substructure();
var client;
try {
  const { token, substructureUrl } = await fetchToken();
  client = sub.frontend.client({ token, url: substructureUrl });
} catch (err) {
  append(
    "tool",
    `\u2717 couldn't mint a client token: ${err?.message ?? err}. is \`substructure start --dev --port 9000 --worker-url http://localhost:3333/agent\` running?`
  );
  throw err;
}
var sessionId;
async function sendMessage(content) {
  append("user", content);
  sendBtn.disabled = true;
  input.disabled = true;
  let typing = showTyping();
  const partials = /* @__PURE__ */ new Map();
  const flush = (p) => {
    let chunk = p.chunks.get(p.nextSeq);
    while (chunk !== void 0) {
      p.node.textContent = (p.node.textContent ?? "") + chunk;
      p.chunks.delete(p.nextSeq);
      p.nextSeq += 1;
      chunk = p.chunks.get(p.nextSeq);
    }
    chatLog.scrollTop = chatLog.scrollHeight;
  };
  try {
    const scope = await client.startTurn({
      agentId: "browser-assistant",
      payload: { type: "message", message: { role: "user", content } },
      sessionId
    });
    sessionId = scope.sessionId;
    for await (const event of client.stream(scope)) {
      const p = event.payload;
      if (p.type === "llm.token.delta") {
        let partial = partials.get(p.call_id);
        if (!partial) {
          typing?.remove();
          typing = null;
          partial = { node: append("assistant", ""), chunks: /* @__PURE__ */ new Map(), nextSeq: 0 };
          partials.set(p.call_id, partial);
        }
        if (typeof p.text === "string" && p.text.length > 0) {
          partial.chunks.set(p.seq, p.text);
          flush(partial);
        }
      } else if (p.type === "llm.call.completed") {
        const partial = partials.get(p.call_id);
        if (partial) {
          partial.node.remove();
          partials.delete(p.call_id);
        }
      } else if (p.type === "tool.call.requested" && tools2[p.name]) {
        const args = p.arguments ? JSON.parse(p.arguments) : {};
        typing?.remove();
        typing = null;
        append("tool", `\u2192 ${p.name}(${p.arguments || ""})`);
        try {
          const result = await tools2[p.name](args);
          append("tool", `\u2190 ${JSON.stringify(result)}`);
          await client.submitToolCallResult({
            sessionId: scope.sessionId,
            toolCallId: p.tool_call_id,
            attempt: p.attempt,
            result: JSON.stringify(result)
          });
        } catch (err) {
          const message = err?.message ?? String(err);
          append("tool", `\u2717 ${message}`);
          await client.submitToolCallResult({
            sessionId: scope.sessionId,
            toolCallId: p.tool_call_id,
            attempt: p.attempt,
            error: message,
            retryable: false
          });
        }
        typing = showTyping();
      } else if (p.type === "message.new" && p.message.role === "assistant") {
        const text = typeof p.message.content === "string" ? p.message.content : "";
        if (text.trim()) {
          typing?.remove();
          typing = null;
          append("assistant", text);
        }
      }
    }
  } catch (err) {
    append("tool", `\u2717 ${err?.message ?? String(err)}`);
  } finally {
    typing?.remove();
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  void sendMessage(text);
});
input.placeholder = "ask me anything\u2026";
input.disabled = false;
sendBtn.disabled = false;
input.focus();
