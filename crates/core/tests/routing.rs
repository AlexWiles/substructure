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
    LlmProviderRegistry, LlmProviderTrait, LlmTask,
};
use substructure_core::protocol::{
    AgentConfig, ClientInput, Content, DraftMessage, LlmRequest, LlmResponse, Role, SessionOwner,
};
use substructure_core::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use substructure_core::providers::sqlite::{
    SqliteCheckpointStore, SqliteDb, SqliteEventStore, SqliteSessionIndexStore, SqliteWakeStore,
    SqliteWorkerQueue,
};
use substructure_core::runtime::connector::ConnectorTask;
use substructure_core::session::events::{DecisionErrored, EventPayload};
use substructure_core::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use substructure_core::span::SpanContext;
use substructure_core::sub_agent::SubAgentTask;
use substructure_core::transport::http_push::http_transport;
use substructure_core::transport::push::PushAdapter;
use substructure_core::worker::push::TransportRegistry;
use substructure_core::worker::{AgentEntry, StaticAgentDirectory, WorkerEndpoint};
use substructure_core::{Caller, HandleClientInput, Runtime, RuntimeConfig, RuntimeDeps};

const TENANT: &str = "default";

// ── A model that answers once ────────────────────────────────────────────

/// Replies with fixed text and records what it was asked, so a test can assert
/// on the prompt the engine composed without a network.
#[derive(Default)]
struct StubModel {
    reply: String,
    seen: Mutex<Vec<LlmRequest>>,
}

#[async_trait]
impl LlmCallable for StubModel {
    async fn call(
        &self,
        request: &LlmRequest,
        _ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError> {
        self.seen.lock().unwrap().push(request.clone());
        Ok(LlmResponse {
            model: request.model.clone(),
            content: Some(self.reply.clone()),
            tool_calls: vec![],
            finish_reason: Some("stop".to_string()),
            usage: None,
            cost: None,
            images: vec![],
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
        stream: false,
        retry: None,
        tools: Vec::new(),
        sub_agents: Vec::new(),
        mcp: Vec::new(),
    }
}

fn engine_hosted(llm: &str) -> AgentEntry {
    AgentEntry {
        config: Some(config(llm)),
        worker: None,
    }
}

fn worker_hosted(llm: &str, url: &str) -> AgentEntry {
    AgentEntry {
        config: Some(config(llm)),
        worker: Some(WorkerEndpoint {
            url: url.to_string(),
            signing_secret: None,
        }),
    }
}

/// An agent that delegates everything: the file names the URL and nothing else.
fn delegating(url: &str) -> AgentEntry {
    AgentEntry {
        config: None,
        worker: Some(WorkerEndpoint {
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
            connector_task_queue: Arc::new(ShardedInMemoryQueue::new(
                config.connector_executor_workers as u32,
            )) as Arc<dyn TaskQueue<ConnectorTask>>,
            worker_queue: Arc::new(SqliteWorkerQueue::new(db.clone()).unwrap()),
            session_index_store: Arc::new(SqliteSessionIndexStore::new(db.clone()).unwrap()),
            checkpoint_store: Arc::new(SqliteCheckpointStore::new(db.clone()).unwrap()),
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
async fn turn(h: &Harness, agent_id: &str, text: &str) -> Vec<EventPayload> {
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
                id: Some("user-1".to_string()),
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
            // A non-retryable decision failure ends the turn as surely as a
            // completion; without it an unroutable agent would just time out.
            let terminal = matches!(
                event.payload,
                EventPayload::TurnCompleted(_)
                    | EventPayload::DecisionErrored(DecisionErrored {
                        retryable: false,
                        ..
                    })
            );
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
        failed.error.contains("no [agent.assistnat]"),
        "the error names the file and the agent; got {}",
        failed.error
    );
    assert!(
        failed.error.contains("assistant"),
        "and says what was declared; got {}",
        failed.error
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
