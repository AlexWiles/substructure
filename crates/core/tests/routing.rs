//! Where a decision goes, end to end.
//!
//! The routing rule is one line of code and the whole of the file's contract,
//! so it is checked against a real engine over a real database rather than
//! against a mock of the router: an agent with a `worker` is pushed there, an
//! agent without one is decided in-engine, and an agent nobody declared fails
//! immediately instead of climbing the retry ladder.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use substructure_core::event_store::Seq;
use substructure_core::llm::{
    CallContext, InMemoryTokenDeltaTransport, LlmBlock, LlmBlocks, LlmCallError, LlmCallable,
    LlmProviderRegistry, LlmProviderTrait, LlmTask, TokenDeltaTransport,
};
use substructure_core::protocol::{
    AgentConfig, ClientInput, Content, DecisionResponse, DecisionTrigger, DraftMessage, ErrorCode,
    LlmResponse, PromptContent, PromptRequest, Role, SessionOwner, SubAgent, ToolCall,
    ToolCallFunction,
};
use substructure_core::protocol::{Issuer, Requester, Subject};
use substructure_core::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use substructure_core::providers::sqlite::{
    SqliteCursorStore, SqliteDb, SqliteEventStore, SqliteSessionIndexStore, SqliteWakeStore,
    SqliteWorkerQueue,
};
use substructure_core::runtime::connector::ConnectorTask;
use substructure_core::session::events::{DecisionErrored, EventPayload};
use substructure_core::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use substructure_core::span::SpanContext;
use substructure_core::sub_agent::SubAgentTask;
use substructure_core::transport::ag_ui::events::AgUiEvent;
use substructure_core::transport::ag_ui::run as ag_ui_run;
use substructure_core::transport::ag_ui::translator::run_ag_ui_translation;
use substructure_core::transport::channel::ChannelContext;
use substructure_core::transport::http_push::http_transport;
use substructure_core::transport::push::PushAdapter;
use substructure_core::worker::push::TransportRegistry;
use substructure_core::worker::push::{PushError, PushTransport};
use substructure_core::worker::{
    AgentEntry, Hosting, StaticAgentDirectory, WorkerDecisionRequest, WorkerEndpoint,
};
use substructure_core::{Caller, HandleClientInput, Runtime, RuntimeConfig, RuntimeDeps};
use tokio_util::sync::CancellationToken;

const TENANT: &str = "default";

// ── A model that answers once ────────────────────────────────────────────

/// Replies with fixed text and records what it was asked, so a test can assert
/// on the prompt the engine composed without a network.
///
/// An offered tool is called once, before any tool result is in the view: a
/// config declaring a sub-agent gets one delegation, and the result that
/// follows ends the turn with the reply.
#[derive(Default)]
struct StubModel {
    reply: String,
    seen: Mutex<Vec<PromptRequest>>,
}

#[async_trait]
impl LlmCallable for StubModel {
    async fn call(
        &self,
        request: &PromptRequest,
        _ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError> {
        self.seen.lock().unwrap().push(request.clone());
        let answered = request.messages.iter().any(|m| m.role == Role::Tool);
        if let (false, Some(tool)) = (answered, request.tools.as_ref().and_then(|t| t.first())) {
            return Ok(LlmResponse {
                model: request.model.clone(),
                content: None,
                tool_calls: vec![ToolCall {
                    id: "tc-1".to_string(),
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: tool.name.clone(),
                        arguments: r#"{"message":"do it"}"#.to_string(),
                    },
                }],
                finish_reason: Some("tool_calls".to_string()),
                usage: None,
                cost: None,
                images: vec![],
                reasoning: None,
            });
        }
        Ok(LlmResponse {
            model: request.model.clone(),
            content: Some(self.reply.clone()),
            tool_calls: vec![],
            finish_reason: Some("stop".to_string()),
            usage: None,
            cost: None,
            images: vec![],
            reasoning: None,
        })
    }
}

struct StubProvider(Arc<StubModel>);

#[async_trait]
impl LlmProviderTrait for StubProvider {
    async fn resolve(&self, _owner: &SessionOwner) -> Result<Arc<dyn LlmCallable>, String> {
        Ok(self.0.clone())
    }
}

// ── Harness ──────────────────────────────────────────────────────────────

fn tmpdir() -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("subs-routing-test-{nanos}-{seq}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn config(llm: &str) -> AgentConfig {
    AgentConfig {
        llm: Some(llm.to_string()),
        model: "stub-model".to_string(),
        system: Some("Be brief.".to_string()),
        retry: None,
        tools: Vec::new(),
        sub_agents: Vec::new(),
        mcp: Vec::new(),
        defer_tools: None,
        announce_mcp: Default::default(),
        plugins: Vec::new(),
        effort: None,
    }
}

fn engine_hosted(llm: &str) -> AgentEntry {
    AgentEntry {
        config: Some(config(llm)),
        hosting: Hosting::Engine,
    }
}

fn worker_hosted(llm: &str, url: &str) -> AgentEntry {
    AgentEntry {
        config: Some(config(llm)),
        hosting: Hosting::Http(WorkerEndpoint {
            url: url.to_string(),
            signing_secret: None,
        }),
    }
}

struct InProcess {
    seen: Mutex<Vec<String>>,
}

#[async_trait]
impl PushTransport for InProcess {
    async fn push(
        &self,
        decision: &WorkerDecisionRequest,
        _deltas: Arc<dyn TokenDeltaTransport>,
    ) -> Result<DecisionResponse, PushError> {
        self.seen
            .lock()
            .unwrap()
            .push(decision.trigger.kind().to_string());
        Ok(match decision.trigger {
            DecisionTrigger::SessionStart => DecisionResponse {
                agent: Some(config("claude")),
                ..Default::default()
            },
            _ => decision.proposed.clone(),
        })
    }
}

/// An agent that delegates everything: the file names the URL and nothing else.
fn delegating(url: &str) -> AgentEntry {
    AgentEntry {
        config: None,
        hosting: Hosting::Http(WorkerEndpoint {
            url: url.to_string(),
            signing_secret: None,
        }),
    }
}

struct Harness {
    runtime: Arc<Runtime>,
    model: Arc<StubModel>,
    // Held for the test's life: dropping it aborts the decision loops.
    _adapter: Arc<PushAdapter>,
}

/// A real engine over a temp database, routing per `agents`.
async fn start(agents: BTreeMap<String, AgentEntry>) -> Harness {
    let db = SqliteDb::open(
        tmpdir().join("substructure.db").to_str().unwrap(),
        std::time::Duration::from_secs(5),
    )
    .unwrap();

    let model = Arc::new(StubModel {
        reply: "hello from the engine".to_string(),
        seen: Mutex::new(Vec::new()),
    });
    let providers = LlmProviderRegistry::new(BTreeMap::from([(
        "claude".to_string(),
        Arc::new(StubProvider(model.clone())) as Arc<dyn LlmProviderTrait>,
    )]));

    let directory = Arc::new(StaticAgentDirectory::new(
        TENANT.to_string(),
        agents,
        LlmBlocks::from_iter([
            ("claude".to_string(), LlmBlock::engine()),
            ("byo".to_string(), LlmBlock::worker(None)),
        ]),
    ));

    let config = RuntimeConfig::default();
    let runtime = substructure_core::start(
        RuntimeDeps {
            blobs: std::sync::Arc::new(substructure_core::runtime::blob::Nowhere),
            store: Arc::new(SqliteEventStore::new(db.clone()).unwrap()),
            agents: directory.clone(),
            llm: Arc::new(providers),
            llm_task_queue: Arc::new(ShardedInMemoryQueue::new(
                config.llm_executor_workers as u32,
            )) as Arc<dyn TaskQueue<LlmTask>>,
            sub_agent_task_queue: Arc::new(ShardedInMemoryQueue::new(
                config.sub_agent_executor_workers as u32,
            )) as Arc<dyn TaskQueue<SubAgentTask>>,
            connections: None,
            plugins: std::sync::Arc::new(substructure_core::plugins::StaticPlugins::default()),
            connector_task_queue: Arc::new(ShardedInMemoryQueue::new(
                config.connector_executor_workers as u32,
            )) as Arc<dyn TaskQueue<ConnectorTask>>,
            worker_queue: Arc::new(SqliteWorkerQueue::new(db.clone()).unwrap()),
            channel_proposers: Vec::new(),
            session_index_store: Arc::new(SqliteSessionIndexStore::new(db.clone()).unwrap()),
            cursor_store: Arc::new(SqliteCursorStore::new(db.clone()).unwrap()),
            wake_store: Arc::new(SqliteWakeStore::new(db).unwrap()),
            token_delta_transport: Arc::new(InMemoryTokenDeltaTransport::new()),
        },
        config,
    );

    let adapter = Arc::new(PushAdapter::new(
        runtime.clone(),
        directory,
        TransportRegistry::new(vec![http_transport()]),
        8,
    ));
    adapter.start();

    Harness {
        runtime,
        model,
        _adapter: adapter,
    }
}

/// Send one user message and drain the turn, returning its events.
///
/// Stops at the first terminal. A non-retryable decision failure counts as one:
/// a `session.start` that fails before any turn opens never emits
/// `TurnCompleted`, so waiting for one would just time out.
async fn turn(h: &Harness, agent_id: &str, text: &str) -> Vec<EventPayload> {
    drain(h, agent_id, text, |e| {
        matches!(
            e,
            EventPayload::TurnCompleted(_)
                | EventPayload::DecisionErrored(DecisionErrored {
                    retryable: false,
                    ..
                })
        )
    })
    .await
}

/// [`turn`], draining until the turn's own terminal. A failure emits its
/// `DecisionErrored` and its `TurnCompleted` in one commit, so a test that
/// wants the second must not stop at the first.
async fn turn_completed(h: &Harness, agent_id: &str, text: &str) -> Vec<EventPayload> {
    drain(h, agent_id, text, |e| {
        matches!(e, EventPayload::TurnCompleted(_))
    })
    .await
}

async fn drain(
    h: &Harness,
    agent_id: &str,
    text: &str,
    stop: impl Fn(&EventPayload) -> bool,
) -> Vec<EventPayload> {
    let session_id = uuid::Uuid::now_v7().to_string();
    let caller = Caller::System {
        tenant_id: TENANT.to_string(),
    };
    let turn_id = h
        .runtime
        .handle_client_input(HandleClientInput {
            session_id: session_id.clone(),
            caller: caller.clone(),
            owner: SessionOwner {
                tenant_id: TENANT.to_string(),
                requester: Requester::new(
                    Subject::new(Issuer::app(), "user-1".to_string()),
                    Default::default(),
                ),
                metadata: Default::default(),
            },
            input: ClientInput::Message {
                agent_id: agent_id.to_string(),
                turn_id: None,
                message: DraftMessage {
                    id: None,
                    role: Role::User,
                    content: Some(Content::Text(text.to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    reasoning: None,
                },
                stream: false,
                queue: false,
            },
            span: SpanContext::root(),
        })
        .await
        .expect("the input is accepted")
        .turn_id;

    let mut events = h
        .runtime
        .stream(
            SessionSubscriptionSpec {
                session_id,
                caller,
                scope: SubscriptionScope::Turn { turn_id },
            },
            Some(Seq(0)),
        )
        .await
        .expect("streams");

    let mut seen = Vec::new();
    let drained = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while let Some(event) = events.recv().await {
            let terminal = stop(&event.payload);
            seen.push(event.payload);
            if terminal {
                return;
            }
        }
    })
    .await;
    assert!(drained.is_ok(), "the turn did not settle: {seen:#?}");
    seen
}

// ── The route table ──────────────────────────────────────────────────────

/// The whole point: a declared agent with no worker runs a full turn with no
/// worker process anywhere.
#[tokio::test]
async fn a_declared_agent_runs_a_turn_in_engine() {
    let h = start(BTreeMap::from([(
        "assistant".to_string(),
        engine_hosted("claude"),
    )]))
    .await;

    let events = turn(&h, "assistant", "hi").await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the turn completes: {events:#?}"
    );

    let seen = h.model.seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "one model call");
    let roles: Vec<_> = seen[0].messages.iter().map(|m| &m.role).collect();
    assert!(
        matches!(roles[..], [Role::System, Role::User]),
        "the declared `system` is the prompt's first message; got {roles:?}"
    );
    assert_eq!(seen[0].model, "stub-model", "the declared model is used");
}

/// A typo in an agent id will not become correct by waiting, so the decision
/// fails at once rather than retrying for ten minutes.
#[tokio::test]
async fn an_undeclared_agent_fails_fast() {
    let h = start(BTreeMap::from([(
        "assistant".to_string(),
        engine_hosted("claude"),
    )]))
    .await;

    let events = turn(&h, "assistnat", "hi").await;
    let failed = events.iter().find_map(|e| match e {
        EventPayload::DecisionErrored(f) => Some(f),
        _ => None,
    });
    let failed = failed.expect("the decision fails: {events:#?}");
    assert!(
        failed.error.message.contains("no [agent.assistnat]"),
        "the error names the file and the agent; got {}",
        failed.error.message
    );
    assert!(
        failed.error.message.contains("assistant"),
        "and says what was declared; got {}",
        failed.error.message
    );
    assert!(h.model.seen.lock().unwrap().is_empty(), "no model call");
}

/// Hosting is per agent, so the two kinds coexist in one tenant: the same
/// engine decides for one and pushes the other.
#[tokio::test]
async fn engine_hosted_and_worker_hosted_agents_share_a_tenant() {
    let worker = echo_worker().await;
    let h = start(BTreeMap::from([
        ("assistant".to_string(), engine_hosted("claude")),
        ("triage".to_string(), worker_hosted("claude", &worker.url())),
    ]))
    .await;

    turn(&h, "assistant", "hi").await;
    assert_eq!(
        worker.calls(),
        0,
        "the engine-hosted agent never touches the worker"
    );

    // The worker-hosted agent's `session.start` reaches the worker instead.
    turn(&h, "triage", "hi").await;
    assert!(worker.calls() > 0, "the worker-hosted agent is pushed");
}

// ── A worker that echoes the proposal ────────────────────────────────────

struct StubWorker {
    addr: std::net::SocketAddr,
    seen: Arc<Mutex<Vec<serde_json::Value>>>,
    _handle: tokio::task::JoinHandle<()>,
}

impl StubWorker {
    fn calls(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    fn url(&self) -> String {
        format!("http://{}/agent", self.addr)
    }

    fn request(&self, trigger: &str) -> Option<serde_json::Value> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .find(|r| r["trigger"]["type"] == trigger)
            .cloned()
    }
}

/// The starter worker, in one line: `({proposed}) => proposed`. It records what
/// it was sent, so a test can assert on the request as well as the outcome.
async fn echo_worker() -> StubWorker {
    use axum::routing::post;
    use axum::{Json, Router};

    let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    let app = Router::new().route(
        "/agent",
        post(move |Json(req): Json<serde_json::Value>| {
            let recorder = recorder.clone();
            async move {
                recorder.lock().unwrap().push(req.clone());
                Json(req["proposed"].clone())
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    StubWorker {
        addr,
        seen,
        _handle: handle,
    }
}

// ── What a broken worker tells the operator ──────────────────────────────

/// A worker that answers every decision with `status` and `body`, whatever
/// those are — the shapes a handler produces when it breaks rather than
/// decides.
async fn broken_worker(status: u16, body: &'static str) -> StubWorker {
    use axum::http::{header, StatusCode};
    use axum::routing::post;
    use axum::Router;

    let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    let app = Router::new().route(
        "/agent",
        post(move |axum::Json(req): axum::Json<serde_json::Value>| {
            let recorder = recorder.clone();
            async move {
                recorder.lock().unwrap().push(req);
                (
                    StatusCode::from_u16(status).unwrap(),
                    [(header::CONTENT_TYPE, "application/json")],
                    body,
                )
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    StubWorker {
        addr,
        seen,
        _handle: handle,
    }
}

/// The events a broken worker produces, end to end.
async fn events_from(status: u16, body: &'static str) -> Vec<EventPayload> {
    let worker = broken_worker(status, body).await;
    let h = start(BTreeMap::from([(
        "triage".to_string(),
        worker_hosted("claude", &worker.url()),
    )]))
    .await;
    turn(&h, "triage", "hi").await
}

/// The decision error a broken worker produces, end to end.
async fn failure_from(status: u16, body: &'static str) -> DecisionErrored {
    let events = events_from(status, body).await;
    events
        .iter()
        .find_map(|e| match e {
            EventPayload::DecisionErrored(f) => Some(f.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the decision fails: {events:#?}"))
}

/// A worker whose own configuration is wrong says so in its body. Reporting
/// only the status leaves the operator with a number, and that number is what
/// reaches Slack.
#[tokio::test]
async fn a_worker_error_body_reaches_the_decision_error() {
    let failed = failure_from(400, r#"{"error":"ANTHROPIC_API_KEY is not set"}"#).await;
    assert!(
        failed
            .error
            .message
            .contains("ANTHROPIC_API_KEY is not set"),
        "the worker's own message comes through; got {}",
        failed.error.message
    );
    assert!(
        failed.error.message.contains("400"),
        "and the status it came with; got {}",
        failed.error.message
    );
    assert_eq!(failed.error.detail.as_ref().unwrap()["status"], 400);
}

/// A handler that catches its own exception and answers `{"error": …}` with a
/// 200 knows why it failed. That sentence is worth more than the parse failure
/// its non-protocol shape would otherwise produce.
#[tokio::test]
async fn an_error_shaped_body_is_reported_as_the_workers_error() {
    let failed = failure_from(200, r#"{"error":"llm block `claud` is not configured"}"#).await;
    assert!(
        failed
            .error
            .message
            .contains("llm block `claud` is not configured"),
        "got {}",
        failed.error.message
    );
    assert!(
        !failed.error.message.contains("invalid decision response"),
        "an error body is the worker's failure, not a parse failure; got {}",
        failed.error.message
    );
}

/// The reported bug: every malformed decision said "failed to parse response:
/// error decoding response body" — reqwest's constant, with serde's own message
/// one link down a source chain nobody printed.
#[tokio::test]
async fn a_malformed_decision_names_the_field_that_failed() {
    let failed = failure_from(
        200,
        r#"{"messages":[],"actions":[{"type":"llm.call","model":17}]}"#,
    )
    .await;
    assert!(
        failed.error.message.contains("invalid type") || failed.error.message.contains("expected"),
        "the error says what was wrong; got {}",
        failed.error.message
    );
    assert!(
        failed.error.message.contains("17"),
        "and quotes the value that was wrong, as Stripe quotes `cus_123`; got {}",
        failed.error.message
    );
    assert!(!failed.retryable, "the same bytes parse the same way");
    // An action is an internally-tagged enum, so serde buffers its body and the
    // param stops at the element rather than reaching `.model`.
    assert_eq!(failed.error.param.as_deref(), Some("actions[0]"));
    assert!(
        !failed.error.message.contains("{"),
        "the raw body is never quoted into the message; got {}",
        failed.error.message
    );
    assert!(
        failed.error.detail.is_none(),
        "nor into the event that is persisted and streamed; got {:?}",
        failed.error.detail
    );
}

/// `TurnCompleted` is the only terminal every consumer watches, so it is where
/// a renderer decides how a failure reads. Carrying only the sentence forced
/// one to match on prose to tell a budget failure from a malformed decision.
#[tokio::test]
async fn the_turn_terminal_carries_the_failure_code() {
    let worker = broken_worker(
        200,
        r#"{"messages":[],"actions":[{"type":"llm.call","model":17}]}"#,
    )
    .await;
    let h = start(BTreeMap::from([(
        "triage".to_string(),
        worker_hosted("claude", &worker.url()),
    )]))
    .await;
    let events = turn_completed(&h, "triage", "hi").await;
    let completed = events
        .iter()
        .find_map(|e| match e {
            EventPayload::TurnCompleted(t) => Some(t),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the turn ends: {events:#?}"));

    assert_eq!(
        completed.error.as_ref().unwrap().code,
        ErrorCode::InvalidResponse
    );
    assert_eq!(
        completed.error.as_ref().unwrap().param.as_deref(),
        Some("actions[0]")
    );
    assert!(
        completed
            .error
            .as_ref()
            .is_some_and(|e| e.message.contains("invalid decision response")),
        "the sentence still rides along; got {:?}",
        completed.error
    );
}

/// An `llm` block the file does not declare is the misconfiguration a worker
/// hits most, and the engine — not the worker — is the only party that knows
/// what *was* declared.
#[tokio::test]
async fn an_unknown_llm_block_names_the_blocks_that_exist() {
    let failed = failure_from(
        200,
        r#"{"messages":[],"actions":[{"type":"llm.call","llm":"claud","model":"m"}]}"#,
    )
    .await;
    assert!(
        failed.error.message.contains("claud") && failed.error.message.contains("claude"),
        "the error names what was asked for and what exists; got {}",
        failed.error.message
    );
    assert_eq!(failed.error.param.as_deref(), Some("llm"));
    let detail = failed.error.detail.as_ref().expect("a structured detail");
    assert_eq!(detail["reason"], "unknown_llm");
    assert_eq!(detail["name"], "claud");
}

/// A section that declares nothing but a `worker` URL seeds no config, so its
/// worker authors the whole identity — the shape of an agent that is entirely
/// your code, and the one the file must not force an `llm`/`model` onto.
#[tokio::test]
async fn an_agent_that_delegates_everything_gets_no_seeded_config() {
    use axum::routing::post;
    use axum::{Json, Router};

    let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    let app = Router::new().route(
        "/agent",
        post(move |Json(req): Json<serde_json::Value>| {
            let recorder = recorder.clone();
            async move {
                recorder.lock().unwrap().push(req.clone());
                // Nothing was seeded, so the worker declares the agent itself.
                if req["trigger"]["type"] == "session.start" {
                    return Json(serde_json::json!({
                        "agent": { "llm": "claude", "model": "stub-model", "system": "Be brief." }
                    }));
                }
                Json(req["proposed"].clone())
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let _server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let h = start(BTreeMap::from([(
        "reggu".to_string(),
        delegating(&format!("http://{addr}/agent")),
    )]))
    .await;
    let events = turn(&h, "reggu", "hi").await;

    let requests = seen.lock().unwrap();
    let start = requests
        .iter()
        .find(|r| r["trigger"]["type"] == "session.start")
        .expect("the worker sees session.start");
    assert!(
        start["proposed"]["agent"].is_null(),
        "nothing declared ⇒ nothing seeded; got {}",
        start["proposed"]
    );

    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the worker's own config runs the turn: {events:#?}"
    );
    assert_eq!(h.model.seen.lock().unwrap().len(), 1);
}

/// The seeded `session.start` proposal reaches the worker, so the starter
/// worker — `({proposed}) => proposed` — inherits the declared config for free.
#[tokio::test]
async fn a_worker_sees_the_declared_config_as_its_start_proposal() {
    let worker = echo_worker().await;
    let h = start(BTreeMap::from([(
        "triage".to_string(),
        worker_hosted("claude", &worker.url()),
    )]))
    .await;
    let events = turn(&h, "triage", "hi").await;

    let start = worker
        .request("session.start")
        .expect("the worker sees session.start");
    assert_eq!(
        start["proposed"]["agent"]["model"], "stub-model",
        "the declared config arrives as the start proposal"
    );
    assert_eq!(start["proposed"]["agent"]["llm"], "claude");

    // Echoing it configures the session, so the turn runs on the declared
    // identity without the worker authoring one.
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the echo worker completes the turn: {events:#?}"
    );
    assert_eq!(h.model.seen.lock().unwrap().len(), 1);
}

/// A delegation and its opening message travel together, so the message cannot
/// reach the child before the child's session exists. When it did, the child
/// started with an empty transcript, never answered, and the parent's turn
/// hung — which the drain in `turn` reports as a turn that never settled.
#[tokio::test]
async fn a_delegation_opens_its_child_with_the_message() {
    let mut boss = config("claude");
    boss.sub_agents = vec![SubAgent {
        id: "helper".to_string(),
        description: "Does the work.".to_string(),
    }];
    let h = start(BTreeMap::from([
        (
            "boss".to_string(),
            AgentEntry {
                config: Some(boss),
                hosting: Hosting::Engine,
            },
        ),
        ("helper".to_string(), engine_hosted("claude")),
    ]))
    .await;

    let events = turn(&h, "boss", "hi").await;

    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::SubAgentStarted(_))),
        "the child session starts: {events:#?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the delegation folds back and the turn completes: {events:#?}"
    );
    assert!(
        h.model
            .seen
            .lock()
            .unwrap()
            .iter()
            .any(|r| r.messages.iter().any(|m| m
                .content
                .as_ref()
                .map(PromptContent::text_owned)
                .as_deref()
                == Some("do it"))),
        "the child is prompted with the delegating message"
    );
}

// ── Starting a turn ──────────────────────────────────────────────────────

fn client_message(agent_id: &str, text: &str) -> ClientInput {
    ClientInput::Message {
        agent_id: agent_id.to_string(),
        turn_id: None,
        message: DraftMessage {
            id: None,
            role: Role::User,
            content: Some(Content::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: None,
        },
        stream: true,
        queue: false,
    }
}

async fn started(h: &Harness, translated: bool) -> (ChannelContext, String, ag_ui_run::Turn) {
    let ctx = ChannelContext::new(h.runtime.clone(), CancellationToken::new());
    let session_id = uuid::Uuid::now_v7().to_string();
    let turn = ag_ui_run::start(
        &ctx,
        &Caller::System {
            tenant_id: TENANT.to_string(),
        },
        &SessionOwner {
            tenant_id: TENANT.to_string(),
            requester: Requester::private(Subject::new(Issuer::cli(), "local".to_string())),
            metadata: Default::default(),
        },
        &session_id,
        client_message("assistant", "hi"),
        "test_start",
        translated,
    )
    .await
    .expect("the turn starts");
    (ctx, session_id, turn)
}

/// `subs run`, `subs chat`, and the `/api/v1/projects/{p}/run` route all open a
/// turn through this one function, so what it admits is what all three read.
///
/// The stream is scoped to the turn, which is what closes it: nothing here
/// asks it to stop.
#[tokio::test]
async fn a_started_turn_ends_its_own_stream() {
    let h = start(BTreeMap::from([(
        "assistant".to_string(),
        engine_hosted("claude"),
    )]))
    .await;
    let (_ctx, _session_id, turn) = started(&h, false).await;
    assert!(
        turn.deltas.is_none(),
        "a reader of engine events subscribes to no token deltas"
    );

    let mut events = turn.events;
    let mut seen = Vec::new();
    let drained = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while let Some(event) = events.recv().await {
            seen.push(event.payload);
        }
    })
    .await;

    assert!(drained.is_ok(), "the stream did not close: {seen:#?}");
    assert!(
        seen.iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the turn completed on the stream: {seen:#?}"
    );
}

/// The same start, read the translated way. A run that reaches its end emits
/// `RunFinished`, so a CLI that never sees one reports an unfinished run.
#[tokio::test]
async fn a_started_turn_translates_to_a_finished_run() {
    let h = start(BTreeMap::from([(
        "assistant".to_string(),
        engine_hosted("claude"),
    )]))
    .await;
    let (ctx, session_id, turn) = started(&h, true).await;
    let deltas = turn
        .deltas
        .expect("a translated reader subscribes to deltas");

    let mut out = run_ag_ui_translation(
        turn.events,
        deltas,
        session_id,
        turn.turn_id,
        ctx.shutdown.clone(),
    );
    let mut seen: Vec<AgUiEvent> = Vec::new();
    let drained = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while let Some(event) = out.recv().await {
            seen.push(event);
        }
    })
    .await;

    assert!(drained.is_ok(), "the stream did not close: {seen:#?}");
    assert!(
        matches!(seen.first(), Some(AgUiEvent::RunStarted { .. })),
        "a run opens with RunStarted: {seen:#?}"
    );
    assert!(
        matches!(seen.last(), Some(AgUiEvent::RunFinished { .. })),
        "and ends with RunFinished: {seen:#?}"
    );
}

#[tokio::test]
async fn an_in_process_decider_runs_a_turn() {
    let decider = Arc::new(InProcess {
        seen: Mutex::new(Vec::new()),
    });
    let h = start(BTreeMap::from([(
        "assistant".to_string(),
        AgentEntry {
            config: None,
            hosting: Hosting::InProcess(decider.clone()),
        },
    )]))
    .await;

    let events = turn(&h, "assistant", "hi").await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the turn completes: {events:#?}"
    );

    let seen = decider.seen.lock().unwrap();
    assert_eq!(
        seen.first().map(String::as_str),
        Some("session.start"),
        "it authors the identity first; got {seen:?}"
    );
    assert!(
        seen.len() > 1,
        "and keeps deciding for the rest of the turn; got {seen:?}"
    );
    assert_eq!(h.model.seen.lock().unwrap().len(), 1, "one model call");
}
