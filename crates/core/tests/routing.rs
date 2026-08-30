use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use substructure_core::event_store::Seq;
use substructure_core::llm::{
    CallContext, InMemoryTokenDeltaTransport, LlmBlock, LlmBlocks, LlmCallError, LlmCallable,
    LlmProviderRegistry, LlmProviderTrait, LlmTask,
};
use substructure_core::protocol::{
    AgentConfig, ChannelKind, ClientInput, Content, DecisionResponse, DraftMessage, ErrorCode,
    LlmResponse, PromptContent, PromptRequest, Role, SessionOwner, Subagent, SubagentTools,
    ToolCall, ToolCallFunction,
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
use substructure_core::subagent::SubagentTask;
use substructure_core::transport::ag_ui::events::AgUiEvent;
use substructure_core::transport::ag_ui::run as ag_ui_run;
use substructure_core::transport::ag_ui::translator::run_ag_ui_translation;
use substructure_core::transport::channel::ChannelContext;
use substructure_core::transport::http_push::http_transport;
use substructure_core::transport::push::PushAdapter;
use substructure_core::worker::push::TransportRegistry;
use substructure_core::worker::ChannelProposer;
use substructure_core::worker::{AgentEntry, Hosting, StaticAgentDirectory, WorkerEndpoint};
use substructure_core::{Caller, HandleClientInput, Runtime, RuntimeConfig, RuntimeDeps};
use tokio_util::sync::CancellationToken;

const TENANT: &str = "default";

#[derive(Default)]
struct StubModel {
    reply: String,
    seen: Mutex<Vec<PromptRequest>>,
    script: Mutex<Vec<ToolCall>>,
    follow_up: Mutex<Option<String>>,
}

fn answered_session(request: &PromptRequest) -> Option<String> {
    let answer = request.messages.iter().rfind(|m| m.role == Role::Tool)?;
    let text = answer.content.as_ref().map(PromptContent::text_owned)?;
    let answer: serde_json::Value = serde_json::from_str(&text).ok()?;
    Some(answer.get("session")?.as_str()?.to_string())
}

#[async_trait]
impl LlmCallable for StubModel {
    async fn call(
        &self,
        request: &PromptRequest,
        _ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError> {
        self.seen.lock().unwrap().push(request.clone());
        if let Some(scripted) = self.script.lock().unwrap().pop() {
            return Ok(LlmResponse {
                model: request.model.clone(),
                content: None,
                tool_calls: vec![scripted],
                finish_reason: Some("tool_calls".to_string()),
                usage: None,
                cost: None,
                images: vec![],
                reasoning: None,
            });
        }
        let answers = request
            .messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .count();
        let follow_up = self.follow_up.lock().unwrap().clone();
        if let (Some(name), 1, Some(session)) = (follow_up, answers, answered_session(request)) {
            let arguments = match name.as_str() {
                "subagent" => {
                    format!(r#"{{"agent":"helper","message":"again","session":"{session}"}}"#)
                }
                _ => format!(r#"{{"message":"again","session":"{session}"}}"#),
            };
            return Ok(LlmResponse {
                model: request.model.clone(),
                content: None,
                tool_calls: vec![ToolCall {
                    id: "tc-2".to_string(),
                    call_type: "function".to_string(),
                    function: ToolCallFunction { name, arguments },
                }],
                finish_reason: Some("tool_calls".to_string()),
                usage: None,
                cost: None,
                images: vec![],
                reasoning: None,
            });
        }
        let answered = answers > 0;
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
        ..Default::default()
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

fn team(boss: AgentConfig) -> BTreeMap<String, AgentEntry> {
    BTreeMap::from([
        (
            "boss".to_string(),
            AgentEntry {
                config: Some(boss),
                hosting: Hosting::Engine,
            },
        ),
        ("helper".to_string(), engine_hosted("claude")),
    ])
}

fn helper(defer: Option<bool>) -> Subagent {
    Subagent {
        id: "helper".to_string(),
        description: "Does the work.".to_string(),
        defer,
        prefix: None,
        mode: None,
    }
}

fn delegating(subagent_tools: Option<SubagentTools>) -> AgentConfig {
    AgentConfig {
        subagents: vec![helper(None)],
        subagent_tools,
        ..config("claude")
    }
}

fn unconfigured_worker(url: &str) -> AgentEntry {
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
    issuer: Issuer,
    _adapter: Arc<PushAdapter>,
}

async fn start(agents: BTreeMap<String, AgentEntry>) -> Harness {
    start_with(agents, Vec::new(), Issuer::app()).await
}

struct Recorder {
    channel: ChannelKind,
    rendered: Arc<Mutex<Vec<String>>>,
}

impl ChannelProposer for Recorder {
    fn channel(&self) -> ChannelKind {
        self.channel
    }

    fn render(
        &self,
        cx: &substructure_core::worker::Proposal<'_>,
        _proposed: &DecisionResponse,
    ) -> Option<serde_json::Value> {
        self.rendered
            .lock()
            .unwrap()
            .push(cx.trigger.kind().to_string());
        Some(serde_json::json!({ "seen": cx.trigger.kind() }))
    }
}

async fn start_with(
    agents: BTreeMap<String, AgentEntry>,
    proposers: Vec<Arc<dyn ChannelProposer>>,
    issuer: Issuer,
) -> Harness {
    let db = SqliteDb::open(
        tmpdir().join("subs.db").to_str().unwrap(),
        std::time::Duration::from_secs(5),
    )
    .unwrap();

    let model = Arc::new(StubModel {
        reply: "hello from the engine".to_string(),
        seen: Mutex::new(Vec::new()),
        script: Mutex::new(Vec::new()),
        follow_up: Mutex::new(None),
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
            subagent_task_queue: Arc::new(ShardedInMemoryQueue::new(
                config.subagent_executor_workers as u32,
            )) as Arc<dyn TaskQueue<SubagentTask>>,
            connections: None,
            plugins: std::sync::Arc::new(substructure_core::plugins::StaticPlugins::default()),
            connector_task_queue: Arc::new(ShardedInMemoryQueue::new(
                config.connector_executor_workers as u32,
            )) as Arc<dyn TaskQueue<ConnectorTask>>,
            worker_queue: Arc::new(SqliteWorkerQueue::new(db.clone()).unwrap()),
            channel_proposers: proposers,
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
        issuer,
        _adapter: adapter,
    }
}

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
                    Subject::new(h.issuer.clone(), "user-1".to_string()),
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

    turn(&h, "triage", "hi").await;
    assert!(worker.calls() > 0, "the worker-hosted agent is pushed");
}

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

async fn events_from(status: u16, body: &'static str) -> Vec<EventPayload> {
    let worker = broken_worker(status, body).await;
    let h = start(BTreeMap::from([(
        "triage".to_string(),
        worker_hosted("claude", &worker.url()),
    )]))
    .await;
    turn(&h, "triage", "hi").await
}

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
        unconfigured_worker(&format!("http://{addr}/agent")),
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

    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the echo worker completes the turn: {events:#?}"
    );
    assert_eq!(h.model.seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_subagent_call_opens_its_child_with_the_message() {
    let h = start(team(delegating(None))).await;

    let events = turn(&h, "boss", "hi").await;

    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::SubagentStarted(_))),
        "the child session starts: {events:#?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the subagent folds back and the turn completes: {events:#?}"
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

#[tokio::test]
async fn a_deferred_subagent_is_delegated_through_call_tool() {
    let boss = AgentConfig {
        subagents: vec![helper(Some(true))],
        ..delegating(None)
    };
    let h = start(team(boss)).await;
    h.model.script.lock().unwrap().push(ToolCall {
        id: "tc-1".to_string(),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: "call_tool".to_string(),
            arguments: r#"{"name":"helper","arguments":{"message":"do it"}}"#.to_string(),
        },
    });

    let events = turn(&h, "boss", "hi").await;

    {
        let seen = h.model.seen.lock().unwrap();
        let offered: Vec<&str> = seen[0]
            .tools
            .as_ref()
            .expect("tools offered")
            .iter()
            .filter(|t| !t.defer)
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(
            offered,
            ["subagent_wait", "tool_search", "call_tool"],
            "the deferred subagent tool stays out of the request"
        );
        assert!(
            seen.iter().any(|r| r.messages.iter().any(|m| m
                .content
                .as_ref()
                .map(PromptContent::text_owned)
                .as_deref()
                == Some("do it"))),
            "the child is prompted with the inner message"
        );
    }
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::SubagentStarted(_))),
        "the call_tool becomes a subagent call: {events:#?}"
    );
    let folded = events
        .iter()
        .find_map(|e| match e {
            EventPayload::NewMessage(n)
                if n.message.tool_call_id.as_deref() == Some("tc-1")
                    && n.message.role == Role::Tool =>
            {
                Some(&n.message)
            }
            _ => None,
        })
        .expect("the subagent folds back under the model's call id");
    assert_eq!(
        folded.name.as_deref(),
        Some("helper"),
        "the record reads as a direct subagent call's"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the turn completes: {events:#?}"
    );
}

fn spawns(events: &[EventPayload]) -> Vec<(String, String)> {
    events
        .iter()
        .filter_map(|e| match e {
            EventPayload::SubagentRequested(r) => Some((r.id.clone(), r.session_id.clone())),
            _ => None,
        })
        .collect()
}

fn subagent_answers(events: &[EventPayload]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            EventPayload::NewMessage(n) if n.message.role == Role::Tool => n
                .message
                .content
                .as_ref()
                .and_then(Content::text)
                .map(str::to_string),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_subagent_call_continues_the_session_its_answer_named() {
    let h = start(team(delegating(None))).await;
    *h.model.follow_up.lock().unwrap() = Some("helper".to_string());

    let events = turn(&h, "boss", "hi").await;

    let spawned = spawns(&events);
    assert_eq!(spawned.len(), 2, "two calls: {events:#?}");
    assert_ne!(spawned[0].0, spawned[1].0, "each call is its own effect");
    assert_eq!(
        spawned[0].1, spawned[1].1,
        "the second call reaches the session the first answer named"
    );

    let answers = subagent_answers(&events);
    assert_eq!(answers.len(), 2, "both calls answer: {answers:#?}");
    for answer in &answers {
        let value: serde_json::Value = serde_json::from_str(answer)
            .unwrap_or_else(|e| panic!("the answer is a JSON object: {answer} ({e})"));
        assert_eq!(value["session"], spawned[0].1);
        assert_eq!(value["result"], "hello from the engine");
    }

    let seen = h.model.seen.lock().unwrap();
    let both = seen.iter().any(|r| {
        let said: Vec<String> = r
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .filter_map(|m| m.content.as_ref().map(PromptContent::text_owned))
            .collect();
        said.iter().any(|t| t == "do it") && said.iter().any(|t| t == "again")
    });
    assert!(both, "the child's second turn sees the first exchange");

    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the turn completes: {events:#?}"
    );
}

#[tokio::test]
async fn a_session_this_agent_never_delegated_to_is_a_tool_error() {
    let h = start(team(delegating(None))).await;
    h.model.script.lock().unwrap().push(ToolCall {
        id: "tc-1".to_string(),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: "helper".to_string(),
            arguments: r#"{"message":"do it","session":"not-my-child"}"#.to_string(),
        },
    });

    let events = turn(&h, "boss", "hi").await;

    let answers = subagent_answers(&events);
    assert!(
        answers
            .iter()
            .any(|a| a.contains("names no session this agent delegated to")),
        "the resume is refused as the call's result: {answers:#?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the turn completes: {events:#?}"
    );
}

#[tokio::test]
async fn a_session_of_another_agent_is_a_tool_error() {
    let mut boss = delegating(None);
    boss.subagents.push(Subagent {
        id: "other".to_string(),
        description: "Does other work.".to_string(),
        defer: None,
        prefix: None,
        mode: None,
    });
    let mut agents = team(boss);
    agents.insert("other".to_string(), engine_hosted("claude"));
    let h = start(agents).await;
    *h.model.follow_up.lock().unwrap() = Some("other".to_string());

    let events = turn(&h, "boss", "hi").await;

    let answers = subagent_answers(&events);
    assert!(
        answers
            .iter()
            .any(|a| a.contains("is a session of `helper`, not of `other`")),
        "a session answers only as its own agent: {answers:#?}"
    );
}

fn single() -> Option<SubagentTools> {
    Some(SubagentTools {
        strategy: substructure_core::protocol::SubagentToolsStrategy::Single,
        wait: None,
    })
}

fn offered(h: &Harness) -> Vec<String> {
    h.model.seen.lock().unwrap()[0]
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .filter(|t| !t.defer)
                .map(|t| t.name.clone())
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn the_single_strategy_offers_one_subagent_tool_that_delegates() {
    let h = start(team(delegating(single()))).await;
    h.model.script.lock().unwrap().push(ToolCall {
        id: "tc-1".to_string(),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: "subagent".to_string(),
            arguments: r#"{"agent":"helper","message":"do it"}"#.to_string(),
        },
    });

    let events = turn(&h, "boss", "hi").await;

    assert_eq!(
        offered(&h),
        ["subagent", "subagent_wait"],
        "one tool stands for every subagent"
    );
    assert_eq!(spawns(&events).len(), 1, "the call delegates: {events:#?}");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::SubagentStarted(_))),
        "the child session starts: {events:#?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the turn completes: {events:#?}"
    );
}

#[tokio::test]
async fn the_single_tool_continues_the_session_it_named() {
    let h = start(team(delegating(single()))).await;
    *h.model.follow_up.lock().unwrap() = Some("subagent".to_string());
    h.model.script.lock().unwrap().push(ToolCall {
        id: "tc-1".to_string(),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: "subagent".to_string(),
            arguments: r#"{"agent":"helper","message":"do it"}"#.to_string(),
        },
    });

    let events = turn(&h, "boss", "hi").await;

    let spawned = spawns(&events);
    assert_eq!(spawned.len(), 2, "two calls: {events:#?}");
    assert_eq!(
        spawned[0].1, spawned[1].1,
        "the second call reaches the same child"
    );
}

#[tokio::test]
async fn an_unknown_agent_answers_with_the_tools_schema() {
    let h = start(team(delegating(single()))).await;
    h.model.script.lock().unwrap().push(ToolCall {
        id: "tc-1".to_string(),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: "subagent".to_string(),
            arguments: r#"{"agent":"nobody","message":"do it"}"#.to_string(),
        },
    });

    let events = turn(&h, "boss", "hi").await;

    assert!(spawns(&events).is_empty(), "nothing delegates: {events:#?}");
    let answers = subagent_answers(&events);
    assert!(
        answers.iter().any(|a| a.contains("helper")),
        "the fault lists the agents it offers: {answers:#?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the turn completes: {events:#?}"
    );
}

#[tokio::test]
async fn a_deferred_single_tool_is_delegated_through_call_tool() {
    let mut boss = delegating(single());
    boss.defer_tools = Some(Default::default());
    let h = start(team(boss)).await;
    h.model.script.lock().unwrap().push(ToolCall {
        id: "tc-1".to_string(),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: "call_tool".to_string(),
            arguments: r#"{"name":"subagent","arguments":{"agent":"helper","message":"do it"}}"#
                .to_string(),
        },
    });

    let events = turn(&h, "boss", "hi").await;

    assert_eq!(
        offered(&h),
        ["tool_search", "call_tool"],
        "the deferred subagent tool stays out of the request"
    );
    assert_eq!(
        spawns(&events).len(),
        1,
        "the call_tool becomes a subagent call: {events:#?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
        "the turn completes: {events:#?}"
    );
}

#[tokio::test]
async fn a_depth_limit_of_zero_keeps_the_subagent_tool_off_the_list() {
    for tools in [None, single()] {
        let boss = AgentConfig {
            max_subagent_depth: Some(0),
            ..delegating(tools)
        };
        let h = start(team(boss)).await;

        let events = turn(&h, "boss", "hi").await;

        assert!(
            h.model
                .seen
                .lock()
                .unwrap()
                .iter()
                .all(|r| r.tools.is_none()),
            "the model is never offered the subagent tool"
        );
        assert!(spawns(&events).is_empty(), "no child spawns: {events:#?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::SubagentStarted(_))),
            "no child starts: {events:#?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::TurnCompleted(_))),
            "the turn completes without delegating: {events:#?}"
        );
    }
}

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
async fn a_proposer_sees_only_the_sessions_its_own_channel_owns() {
    let mine = Arc::new(Mutex::new(Vec::new()));
    let theirs = Arc::new(Mutex::new(Vec::new()));
    let proposers: Vec<Arc<dyn ChannelProposer>> = vec![
        Arc::new(Recorder {
            channel: ChannelKind::CLI,
            rendered: mine.clone(),
        }),
        Arc::new(Recorder {
            channel: ChannelKind::SLACK,
            rendered: theirs.clone(),
        }),
    ];
    let h = start_with(
        BTreeMap::from([("assistant".to_string(), engine_hosted("claude"))]),
        proposers,
        Issuer::cli(),
    )
    .await;

    turn(&h, "assistant", "hi").await;

    assert!(
        !mine.lock().unwrap().is_empty(),
        "the owning channel renders the decision"
    );
    assert!(
        theirs.lock().unwrap().is_empty(),
        "another channel never sees it: {:?}",
        theirs.lock().unwrap()
    );
}
