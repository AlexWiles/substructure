use std::collections::BTreeMap;

use crate::protocol::{AgentConfig, RetryPolicy, SessionOwner, WorkerRef};
use crate::runtime::llm::LlmBlocks;
use crate::runtime::session::command::CommandPayload;
use crate::runtime::worker::WorkerDecisionRequest;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Hosting {
    Engine,
    Worker(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AgentEntry {
    pub config: Option<AgentConfig>,
    pub hosting: Hosting,
}

/// Where one decision goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    Worker { id: String, url: String },
    Engine,
}

/// Everything one tenant declares, read once.
///
/// The decision path asks a question per decision, not per field, so an
/// implementation backed by a database reads the tenant whole instead of
/// answering one accessor at a time.
#[derive(Debug, Clone, Default)]
pub struct TenantDirectory {
    pub agents: BTreeMap<String, AgentEntry>,
    pub workers: BTreeMap<String, WorkerBlock>,
    pub default_worker: Option<String>,
    pub llm: LlmBlocks,
}

impl TenantDirectory {
    pub fn agent(&self, agent_id: &str) -> Option<&AgentEntry> {
        self.agents.get(agent_id)
    }

    pub fn declares(&self, agent_id: &str) -> bool {
        self.agents.contains_key(agent_id)
    }

    pub fn worker(&self, id: &str) -> Option<&WorkerBlock> {
        self.workers.get(id)
    }

    pub fn agent_ids(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    /// The worker a new session is stamped with. A declared agent follows the
    /// file rather than a stamp; only one nobody declared takes the default.
    pub fn resolve_worker(&self, agent_id: &str, named: Option<WorkerRef>) -> Option<WorkerRef> {
        named.or_else(|| match self.declares(agent_id) {
            true => None,
            false => self
                .default_worker
                .clone()
                .map(|id| WorkerRef { id, url: None }),
        })
    }

    /// Where a dequeued decision goes: the worker the session pinned, else the
    /// one its agent names, else the engine. `Err` is unroutable and says why.
    pub fn route(&self, decision: &WorkerDecisionRequest) -> Result<Route, String> {
        let agent_id = &decision.agent_id;
        let hosting = match &decision.worker {
            Some(w) => Some(Hosting::Worker(w.id.clone())),
            None => self.agent(agent_id).map(|e| e.hosting.clone()),
        };
        let worker_id = match hosting {
            Some(Hosting::Engine) => return Ok(Route::Engine),
            Some(Hosting::Worker(id)) => id,
            None if decision.agent.is_some() => return Ok(Route::Engine),
            None => {
                return Err(format!(
                    "no [agent.{agent_id}], no worker on the session, and no config on the \
                     session. Declared agents: {}",
                    crate::copy::declared(self.agent_ids())
                ))
            }
        };
        let Some(block) = self.worker(&worker_id) else {
            return Err(format!("no [worker.{worker_id}] in subs.toml"));
        };
        let Some(url) = decision
            .worker
            .as_ref()
            .and_then(|w| w.url.clone())
            .or_else(|| block.url.clone())
        else {
            return Err(format!(
                "[worker.{worker_id}] has no `url` and the session brought none"
            ));
        };
        Ok(Route::Worker { id: worker_id, url })
    }
}

pub trait AgentDirectory: Send + Sync {
    fn tenant(&self, tenant_id: &str) -> TenantDirectory;

    fn tenants(&self) -> Vec<String>;
}

pub fn create_session_command(
    agents: &dyn AgentDirectory,
    agent_id: String,
    owner: SessionOwner,
    ancestry: Vec<String>,
    worker_retry: RetryPolicy,
    agent: Option<AgentConfig>,
    worker: Option<WorkerRef>,
) -> CommandPayload {
    CommandPayload::CreateSession {
        worker: agents
            .tenant(&owner.tenant_id)
            .resolve_worker(&agent_id, worker),
        agent_id,
        owner,
        ancestry,
        worker_retry,
        agent,
    }
}

pub struct StaticAgentDirectory {
    tenant_id: String,
    directory: TenantDirectory,
}

impl StaticAgentDirectory {
    pub fn new(tenant_id: String, agents: BTreeMap<String, AgentEntry>, llm: LlmBlocks) -> Self {
        Self {
            tenant_id,
            directory: TenantDirectory {
                agents,
                llm,
                ..Default::default()
            },
        }
    }

    pub fn with_workers(
        mut self,
        workers: BTreeMap<String, WorkerBlock>,
        default_worker: Option<String>,
    ) -> Self {
        self.directory.workers = workers;
        self.directory.default_worker = default_worker;
        self
    }
}

impl AgentDirectory for StaticAgentDirectory {
    fn tenant(&self, tenant_id: &str) -> TenantDirectory {
        match tenant_id == self.tenant_id {
            true => self.directory.clone(),
            false => TenantDirectory::default(),
        }
    }

    fn tenants(&self) -> Vec<String> {
        vec![self.tenant_id.clone()]
    }
}

pub struct EmptyAgentDirectory;

impl AgentDirectory for EmptyAgentDirectory {
    fn tenant(&self, _tenant_id: &str) -> TenantDirectory {
        TenantDirectory::default()
    }

    fn tenants(&self) -> Vec<String> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{DecisionResponse, DecisionTrigger, MessageTree, WorkerState};
    use crate::runtime::llm::LlmBlock;
    use crate::runtime::span::SpanContext;

    fn config(llm: &str) -> AgentConfig {
        AgentConfig {
            llm: Some(llm.to_string()),
            model: "m".to_string(),
            ..Default::default()
        }
    }

    fn directory() -> StaticAgentDirectory {
        StaticAgentDirectory::new(
            "default".to_string(),
            BTreeMap::from([
                (
                    "assistant".to_string(),
                    AgentEntry {
                        config: Some(config("claude")),
                        hosting: Hosting::Engine,
                    },
                ),
                (
                    "triage".to_string(),
                    AgentEntry {
                        config: None,
                        hosting: Hosting::Worker("triage".to_string()),
                    },
                ),
            ]),
            LlmBlocks::from_iter([("claude".to_string(), LlmBlock::engine())]),
        )
        .with_workers(
            BTreeMap::from([
                (
                    "triage".to_string(),
                    WorkerBlock {
                        url: Some("https://triage.internal/agent".to_string()),
                    },
                ),
                ("customers".to_string(), WorkerBlock { url: None }),
            ]),
            Some("customers".to_string()),
        )
    }

    fn decision(agent_id: &str) -> WorkerDecisionRequest {
        WorkerDecisionRequest {
            session_id: "s-1".to_string(),
            decision_id: "d-1".to_string(),
            agent_id: agent_id.to_string(),
            identity: SessionOwner::default(),
            trigger: DecisionTrigger::SessionStart,
            proposed: DecisionResponse::default(),
            state: WorkerState::default(),
            agent: None,
            worker: None,
            calls: Vec::new(),
            pending_calls: 0,
            transcript: Vec::new(),
            message_tree: MessageTree::default(),
            ancestry: Vec::new(),
            span: SpanContext::root(),
            attempts: 0,
            deadline: None,
            turn_id: None,
        }
    }

    #[test]
    fn an_agent_carries_its_config_and_its_hosting() {
        let d = directory().tenant("default");
        let engine_hosted = d.agent("assistant").expect("declared");
        assert_eq!(
            engine_hosted.config.as_ref().and_then(|c| c.llm.as_deref()),
            Some("claude")
        );
        assert_eq!(engine_hosted.hosting, Hosting::Engine);

        let pushed = d.agent("triage").expect("declared");
        assert_eq!(pushed.hosting, Hosting::Worker("triage".to_string()));
        let block = d.worker("triage").expect("declared");
        assert_eq!(block.url.as_deref(), Some("https://triage.internal/agent"));
    }

    #[test]
    fn the_default_worker_is_the_one_the_file_names() {
        let d = directory();
        assert_eq!(
            d.tenant("default").default_worker.as_deref(),
            Some("customers")
        );
        assert!(d.tenant("other").default_worker.is_none());
        assert!(d.tenant("other").worker("triage").is_none());
    }

    #[test]
    fn an_undeclared_agent_and_another_tenant_are_both_absent() {
        let d = directory();
        assert!(d.tenant("default").agent("typo").is_none());
        assert!(d.tenant("other").agent("assistant").is_none());
        assert!(d.tenant("other").llm.is_empty());
        assert_eq!(d.tenants(), vec!["default".to_string()]);
        assert_eq!(d.tenant("default").agent_ids(), ["assistant", "triage"]);
    }

    #[test]
    fn creation_resolves_the_worker_once() {
        let d = directory().tenant("default");
        let named = WorkerRef {
            id: "triage".to_string(),
            url: None,
        };
        assert_eq!(
            d.resolve_worker("invented", Some(named.clone())),
            Some(named),
            "a named worker wins"
        );
        assert_eq!(
            d.resolve_worker("assistant", None),
            None,
            "a declared agent follows the file, not a stamp"
        );
        assert_eq!(
            d.resolve_worker("invented", None),
            Some(WorkerRef {
                id: "customers".to_string(),
                url: None,
            }),
            "an undeclared agent is stamped with the default at creation"
        );
        assert_eq!(
            EmptyAgentDirectory
                .tenant("default")
                .resolve_worker("invented", None),
            None,
            "no default, no stamp"
        );
    }

    #[test]
    fn a_decision_follows_its_agents_hosting() {
        let d = directory().tenant("default");
        assert_eq!(d.route(&decision("assistant")), Ok(Route::Engine));
        assert_eq!(
            d.route(&decision("triage")),
            Ok(Route::Worker {
                id: "triage".to_string(),
                url: "https://triage.internal/agent".to_string(),
            })
        );
    }

    /// A session pins a worker and may bring the address with it, which is how
    /// a block with no `url` is reached at all.
    #[test]
    fn a_session_can_pin_a_worker_and_bring_its_address() {
        let d = directory().tenant("default");
        let pinned = |id: &str, url: Option<&str>| WorkerDecisionRequest {
            worker: Some(WorkerRef {
                id: id.to_string(),
                url: url.map(str::to_string),
            }),
            ..decision("assistant")
        };
        assert_eq!(
            d.route(&pinned("customers", Some("https://brought.test"))),
            Ok(Route::Worker {
                id: "customers".to_string(),
                url: "https://brought.test".to_string(),
            }),
            "the session's address wins over the agent's hosting"
        );
        assert!(
            d.route(&pinned("customers", None))
                .is_err_and(|e| e.contains("has no `url`")),
            "a block with no address and a session that brought none is unroutable"
        );
        assert!(d
            .route(&pinned("invented", None))
            .is_err_and(|e| e.contains("no [worker.invented]")));
    }

    /// An agent nothing declares is still decidable when the session carries
    /// its config: the engine answers with its own proposal.
    #[test]
    fn an_undeclared_agent_needs_a_config_on_the_session() {
        let d = directory().tenant("default");
        assert!(d
            .route(&decision("typo"))
            .is_err_and(|e| e.contains("no [agent.typo]")));
        assert_eq!(
            d.route(&WorkerDecisionRequest {
                agent: Some(config("claude")),
                ..decision("typo")
            }),
            Ok(Route::Engine)
        );
    }

    #[test]
    fn an_empty_directory_declares_nothing() {
        let d = EmptyAgentDirectory;
        assert!(d.tenant("default").agent("assistant").is_none());
        assert!(d.tenant("default").worker("main").is_none());
        assert!(d.tenant("default").default_worker.is_none());
        assert!(d.tenants().is_empty());
        assert_eq!(
            crate::copy::declared(d.tenant("default").agent_ids()),
            "none"
        );
    }
}
