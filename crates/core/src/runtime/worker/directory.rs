use std::collections::BTreeMap;

use crate::protocol::AgentConfig;
use crate::runtime::llm::LlmBlocks;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Hosting {
    Engine,
    Http(WorkerEndpoint),
}

#[derive(Debug, Clone)]
pub struct AgentEntry {
    pub config: Option<AgentConfig>,
    pub hosting: Hosting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerEndpoint {
    pub url: String,
    pub signing_secret: Option<String>,
}

pub trait AgentDirectory: Send + Sync {
    fn agent(&self, tenant_id: &str, agent_id: &str) -> Option<AgentEntry>;

    fn llm(&self, tenant_id: &str) -> LlmBlocks;

    fn agent_ids(&self, tenant_id: &str) -> Vec<String>;

    fn tenants(&self) -> Vec<String>;
}

pub struct StaticAgentDirectory {
    tenant_id: String,
    agents: BTreeMap<String, AgentEntry>,
    llm: LlmBlocks,
}

impl StaticAgentDirectory {
    pub fn new(tenant_id: String, agents: BTreeMap<String, AgentEntry>, llm: LlmBlocks) -> Self {
        Self {
            tenant_id,
            agents,
            llm,
        }
    }
}

impl AgentDirectory for StaticAgentDirectory {
    fn agent(&self, tenant_id: &str, agent_id: &str) -> Option<AgentEntry> {
        (tenant_id == self.tenant_id)
            .then(|| self.agents.get(agent_id).cloned())
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
        match self.agents.is_empty() {
            true => Vec::new(),
            false => vec![self.tenant_id.clone()],
        }
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

pub fn declared(ids: &[String]) -> String {
    match ids.is_empty() {
        true => "none".to_string(),
        false => ids.join(", "),
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
                        hosting: Hosting::Http(WorkerEndpoint {
                            url: "https://triage.internal/agent".to_string(),
                            signing_secret: None,
                        }),
                    },
                ),
            ]),
            LlmBlocks::from_iter([("claude".to_string(), LlmBlock::engine())]),
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
        assert!(pushed.config.is_none());
        let Hosting::Http(endpoint) = pushed.hosting else {
            panic!("a declared worker is hosted over http");
        };
        assert_eq!(endpoint.url, "https://triage.internal/agent");
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
    fn an_empty_directory_declares_nothing() {
        let d = EmptyAgentDirectory;
        assert!(d.agent("default", "assistant").is_none());
        assert!(d.tenants().is_empty());
        assert_eq!(declared(&d.agent_ids("default")), "none");
    }
}
