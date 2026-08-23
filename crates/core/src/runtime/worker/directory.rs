use std::collections::BTreeMap;
use std::sync::Arc;

use crate::protocol::AgentConfig;
use crate::runtime::llm::LlmBlocks;

use super::push::PushTransport;

#[derive(Clone)]
pub enum Hosting {
    Engine,
    Http(WorkerEndpoint),
    InProcess(Arc<dyn PushTransport>),
}

impl std::fmt::Debug for Hosting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Engine => f.write_str("Engine"),
            Self::Http(endpoint) => f.debug_tuple("Http").field(endpoint).finish(),
            Self::InProcess(_) => f.write_str("InProcess"),
        }
    }
}

impl PartialEq for Hosting {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Engine, Self::Engine) => true,
            (Self::Http(a), Self::Http(b)) => a == b,
            (Self::InProcess(a), Self::InProcess(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// One agent as declared: the config the app seeds, plus the hosting that never
/// crosses the wire.
///
/// `config` is `None` for an agent that delegates everything to its worker —
/// the declaration says the agent exists and where its decisions go, and the
/// worker authors the identity on `session.start`. An engine-hosted agent
/// always has one; nothing else could supply it.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentEntry {
    pub config: Option<AgentConfig>,
    pub hosting: Hosting,
}

/// A worker attached to one agent. `signing_secret` is what the named
/// environment variable held; unset means the requests go unsigned rather than
/// signed with a secret nobody can check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerEndpoint {
    pub url: String,
    pub signing_secret: Option<String>,
}

/// Every agent and llm block one deployment serves. Read-only: the declaration
/// is applied at startup (locally, from `substructure.toml`), never mutated by
/// a running engine.
pub trait AgentDirectory: Send + Sync {
    /// The agent as declared, or `None` when this app declares no such agent.
    fn agent(&self, tenant_id: &str, agent_id: &str) -> Option<AgentEntry>;

    /// The `[llm.*]` blocks this app declares.
    fn llm(&self, tenant_id: &str) -> LlmBlocks;

    /// The declared agent ids, for the error that says what could have been named.
    fn agent_ids(&self, tenant_id: &str) -> Vec<String>;

    /// Tenants with anything declared — the decision loops to start.
    fn tenants(&self) -> Vec<String>;
}

/// One tenant's declaration, read from the file at startup.
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

/// An engine nothing is declared to — an embedded runtime whose workers are
/// registered in process. Every agent is undeclared, so every decision it is
/// asked to route fails saying so.
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

/// The declared ids as an error names them.
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
            system: None,
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

        // A worker-hosted agent may seed nothing: its worker authors the config.
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
