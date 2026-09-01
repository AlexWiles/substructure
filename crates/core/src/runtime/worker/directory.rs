use std::collections::BTreeMap;

use crate::protocol::{AgentConfig, RetryPolicy, SessionOwner, WorkerRef};
use crate::runtime::llm::LlmBlocks;
use crate::runtime::session::command::CommandPayload;

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

pub trait AgentDirectory: Send + Sync {
    fn agent(&self, tenant_id: &str, agent_id: &str) -> Option<AgentEntry>;

    fn declares(&self, tenant_id: &str, agent_id: &str) -> bool {
        self.agent(tenant_id, agent_id).is_some()
    }

    fn worker(&self, _tenant_id: &str, _id: &str) -> Option<WorkerBlock> {
        None
    }

    fn default_worker(&self, _tenant_id: &str) -> Option<String> {
        None
    }

    fn llm(&self, tenant_id: &str) -> LlmBlocks;

    fn agent_ids(&self, tenant_id: &str) -> Vec<String>;

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
        worker: resolve_worker(agents, &owner.tenant_id, &agent_id, worker),
        agent_id,
        owner,
        ancestry,
        worker_retry,
        agent,
    }
}

pub fn resolve_worker(
    agents: &dyn AgentDirectory,
    tenant_id: &str,
    agent_id: &str,
    named: Option<WorkerRef>,
) -> Option<WorkerRef> {
    named.or_else(|| match agents.declares(tenant_id, agent_id) {
        true => None,
        false => agents
            .default_worker(tenant_id)
            .map(|id| WorkerRef { id, url: None }),
    })
}

pub struct StaticAgentDirectory {
    tenant_id: String,
    agents: BTreeMap<String, AgentEntry>,
    llm: LlmBlocks,
    workers: BTreeMap<String, WorkerBlock>,
    default_worker: Option<String>,
}

impl StaticAgentDirectory {
    pub fn new(tenant_id: String, agents: BTreeMap<String, AgentEntry>, llm: LlmBlocks) -> Self {
        Self {
            tenant_id,
            agents,
            llm,
            workers: BTreeMap::new(),
            default_worker: None,
        }
    }

    pub fn with_workers(
        mut self,
        workers: BTreeMap<String, WorkerBlock>,
        default_worker: Option<String>,
    ) -> Self {
        self.workers = workers;
        self.default_worker = default_worker;
        self
    }
}

impl AgentDirectory for StaticAgentDirectory {
    fn agent(&self, tenant_id: &str, agent_id: &str) -> Option<AgentEntry> {
        (tenant_id == self.tenant_id)
            .then(|| self.agents.get(agent_id).cloned())
            .flatten()
    }

    fn declares(&self, tenant_id: &str, agent_id: &str) -> bool {
        tenant_id == self.tenant_id && self.agents.contains_key(agent_id)
    }

    fn worker(&self, tenant_id: &str, id: &str) -> Option<WorkerBlock> {
        (tenant_id == self.tenant_id)
            .then(|| self.workers.get(id).cloned())
            .flatten()
    }

    fn default_worker(&self, tenant_id: &str) -> Option<String> {
        (tenant_id == self.tenant_id)
            .then(|| self.default_worker.clone())
            .flatten()
    }

    fn llm(&self, tenant_id: &str) -> LlmBlocks {
        match tenant_id == self.tenant_id {
            true => self.llm.clone(),
            false => LlmBlocks::default(),
        }
    }

    fn agent_ids(&self, tenant_id: &str) -> Vec<String> {
        match tenant_id == self.tenant_id {
            true => self.agents.keys().cloned().collect(),
            false => Vec::new(),
        }
    }

    fn tenants(&self) -> Vec<String> {
        vec![self.tenant_id.clone()]
    }
}

pub struct EmptyAgentDirectory;

impl AgentDirectory for EmptyAgentDirectory {
    fn agent(&self, _tenant_id: &str, _agent_id: &str) -> Option<AgentEntry> {
        None
    }

    fn llm(&self, _tenant_id: &str) -> LlmBlocks {
        LlmBlocks::default()
    }

    fn agent_ids(&self, _tenant_id: &str) -> Vec<String> {
        Vec::new()
    }

    fn tenants(&self) -> Vec<String> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::llm::LlmBlock;

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

    #[test]
    fn an_agent_carries_its_config_and_its_hosting() {
        let d = directory();
        let engine_hosted = d.agent("default", "assistant").expect("declared");
        assert_eq!(
            engine_hosted.config.and_then(|c| c.llm).as_deref(),
            Some("claude")
        );
        assert_eq!(engine_hosted.hosting, Hosting::Engine);

        let pushed = d.agent("default", "triage").expect("declared");
        assert_eq!(pushed.hosting, Hosting::Worker("triage".to_string()));
        let block = d.worker("default", "triage").expect("declared");
        assert_eq!(block.url.as_deref(), Some("https://triage.internal/agent"));
    }

    #[test]
    fn the_default_worker_is_the_one_the_file_names() {
        let d = directory();
        assert_eq!(d.default_worker("default").as_deref(), Some("customers"));
        assert!(d.default_worker("other").is_none());
        assert!(d.worker("other", "triage").is_none());
    }

    #[test]
    fn an_undeclared_agent_and_another_tenant_are_both_absent() {
        let d = directory();
        assert!(d.agent("default", "typo").is_none());
        assert!(d.agent("other", "assistant").is_none());
        assert!(d.llm("other").is_empty());
        assert_eq!(d.tenants(), vec!["default".to_string()]);
        assert_eq!(d.agent_ids("default"), ["assistant", "triage"]);
    }

    #[test]
    fn creation_resolves_the_worker_once() {
        let d = directory();
        let named = WorkerRef {
            id: "triage".to_string(),
            url: None,
        };
        assert_eq!(
            resolve_worker(&d, "default", "invented", Some(named.clone())),
            Some(named),
            "a named worker wins"
        );
        assert_eq!(
            resolve_worker(&d, "default", "assistant", None),
            None,
            "a declared agent follows the file, not a stamp"
        );
        assert_eq!(
            resolve_worker(&d, "default", "invented", None),
            Some(WorkerRef {
                id: "customers".to_string(),
                url: None,
            }),
            "an undeclared agent is stamped with the default at creation"
        );
        assert_eq!(
            resolve_worker(&EmptyAgentDirectory, "default", "invented", None),
            None,
            "no default, no stamp"
        );
    }

    #[test]
    fn an_empty_directory_declares_nothing() {
        let d = EmptyAgentDirectory;
        assert!(d.agent("default", "assistant").is_none());
        assert!(d.worker("default", "main").is_none());
        assert!(d.default_worker("default").is_none());
        assert!(d.tenants().is_empty());
        assert_eq!(crate::copy::declared(d.agent_ids("default")), "none");
    }
}
