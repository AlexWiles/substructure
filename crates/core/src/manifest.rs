use std::collections::BTreeMap;

use anyhow::{bail, Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::connectors::registry::{
    AuthKind, ConnectionDecl, ConnectionPath, ConnectionSpec, CredentialScope,
};
use crate::plugins::{BundleServer, PluginBundle};
use crate::protocol::ReasoningEffort;
use crate::protocol::{
    AgentConfig, AgentPlugin, AgentTool, Approve, ConnectorProtocol, DeferTools, Handler,
    LlmFormat, McpAnnounce, McpAuthFailure, McpServer, McpToolSyncFailure, McpTools, RetryConfig,
    SpawnMode, Subagent, SubagentMode, SubagentTools, SubagentToolsStrategy, SUBAGENT,
    SUBAGENT_WAIT,
};
use crate::runtime::llm::{LlmBlock, LlmBlocks};
use crate::runtime::worker::{AgentEntry, Hosting, WorkerEndpoint};

pub use crate::cli::env::ProviderKind;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Manifest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_subagent_depth: Option<u32>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub llm: BTreeMap<String, ProviderSpec>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub agent: BTreeMap<String, AgentSection>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, ConnectionDecl>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub plugin: BTreeMap<String, PluginSpec>,
}

impl Manifest {
    pub fn validate(&self) -> Result<()> {
        for (id, spec) in &self.mcp {
            check_id(id).map_err(|e| anyhow::anyhow!("[mcp.{id}]: {e}"))?;
            check_url(&spec.url).map_err(|e| anyhow::anyhow!("[mcp.{id}]: {e}"))?;
            check_connection(spec).map_err(|e| anyhow::anyhow!("[mcp.{id}]: {e}"))?;
        }
        for (id, spec) in &self.plugin {
            check_plugin(id, spec).map_err(|e| anyhow::anyhow!("[plugin.{id}]: {e}"))?;
        }
        check_tool_prefixes(self)?;
        for (id, spec) in &self.llm {
            check_llm(id, spec).map_err(|e| anyhow::anyhow!("[llm.{id}]: {e}"))?;
        }
        for (id, section) in &self.agent {
            check_agent(id, section, self).map_err(|e| anyhow::anyhow!("[agent.{id}]: {e}"))?;
        }
        Ok(())
    }

    pub fn local_bindings(&self) -> Vec<String> {
        let llm = self
            .llm
            .iter()
            .filter(|(_, s)| s.api_key_env.is_some())
            .map(|(id, _)| format!("[llm.{id}].api_key_env"));
        let agent = self
            .agent
            .iter()
            .filter(|(_, s)| s.signing_secret_env.is_some())
            .map(|(id, _)| format!("[agent.{id}].signing_secret_env"));
        let mcp = self.mcp.iter().flat_map(|(id, s)| {
            s.client_id_env
                .iter()
                .map(move |_| format!("[mcp.{id}].client_id_env"))
                .chain(
                    s.client_secret_env
                        .iter()
                        .map(move |_| format!("[mcp.{id}].client_secret_env")),
                )
        });
        let plugin = self.plugin.iter().flat_map(|(pid, spec)| {
            spec.mcp.iter().flat_map(move |(name, s)| {
                s.client_id_env
                    .iter()
                    .map(move |_| format!("[plugin.{pid}.mcp.{name}].client_id_env"))
                    .chain(
                        s.client_secret_env
                            .iter()
                            .map(move |_| format!("[plugin.{pid}.mcp.{name}].client_secret_env")),
                    )
            })
        });
        llm.chain(agent).chain(mcp).chain(plugin).collect()
    }

    pub fn for_wire(&self) -> Self {
        let mut wire = self.clone();
        for spec in wire.llm.values_mut() {
            spec.api_key_env = None;
        }
        for section in wire.agent.values_mut() {
            section.signing_secret_env = None;
        }
        for spec in wire.mcp.values_mut() {
            spec.client_id_env = None;
            spec.client_secret_env = None;
        }
        for spec in wire.plugin.values_mut() {
            spec.path = None;
            spec.bundle = None;
            for server in spec.mcp.values_mut() {
                server.client_id_env = None;
                server.client_secret_env = None;
            }
        }
        wire
    }

    pub fn connections(&self) -> BTreeMap<ConnectionPath, ConnectionSpec> {
        let declared = self.mcp.iter().map(|(id, decl)| {
            let path = ConnectionPath::Mcp(id.clone());
            (path.clone(), decl.clone().at(path, ConnectorProtocol::Mcp))
        });
        let from_plugins = self.plugin.iter().flat_map(|(pid, spec)| {
            spec.bundle.iter().flat_map(move |b| {
                b.servers.iter().map(move |(name, server)| {
                    let path = ConnectionPath::PluginServer {
                        plugin: pid.clone(),
                        server: name.clone(),
                    };
                    let decl = spec.mcp.get(name).cloned().unwrap_or_default().over(server);
                    (path.clone(), decl.at(path, ConnectorProtocol::Mcp))
                })
            })
        });
        declared.chain(from_plugins).collect()
    }

    pub fn connection_at(&self, path: &ConnectionPath) -> Option<ConnectionSpec> {
        self.connections().get(path).cloned()
    }

    pub fn connection_paths(&self) -> Vec<ConnectionPath> {
        self.connections().into_keys().collect()
    }

    pub fn resolve_plugins(&mut self, base: &std::path::Path) -> Result<ResolvedPlugins> {
        let mut resolved = ResolvedPlugins::default();
        for (id, spec) in &mut self.plugin {
            if spec.bundle.is_some() {
                continue;
            }
            let Some(path) = &spec.path else {
                continue;
            };
            let loaded = plugin_dir(base, path)
                .and_then(|dir| crate::plugins::load_dir(&dir))
                .map_err(|e| anyhow::anyhow!("[plugin.{id}]: {e}"))?;
            spec.hash = Some(loaded.hash());
            resolved.notices.extend(
                loaded
                    .notices
                    .into_iter()
                    .map(|n| format!("[plugin.{id}]: {n}")),
            );
            spec.bundle = Some(loaded.bundle);
            if !loaded.pending.is_empty() {
                resolved.pending.insert(id.clone(), loaded.pending);
            }
        }
        self.validate()?;
        Ok(resolved)
    }

    pub fn llm_blocks(&self) -> LlmBlocks {
        self.llm
            .iter()
            .map(|(name, spec)| (name.clone(), spec.block()))
            .collect()
    }

    pub fn agents(&self) -> BTreeMap<String, AgentEntry> {
        self.agent
            .iter()
            .map(|(id, section)| (id.clone(), section.to_entry(self)))
            .collect()
    }

    pub fn agent_ids(&self) -> Vec<String> {
        self.agent.keys().cloned().collect()
    }

    pub fn slack_apps(&self) -> Vec<(&str, &AgentSlackConfig)> {
        self.agent
            .iter()
            .filter_map(|(id, section)| Some((id.as_str(), section.slack.as_ref()?)))
            .collect()
    }

    pub fn scope_notices(&self) -> Vec<String> {
        let connections = self.connections();
        let personal: Vec<&ConnectionPath> = connections
            .values()
            .filter(|spec| spec.effective_scope() == CredentialScope::User)
            .map(|spec| &spec.path)
            .collect();
        if personal.is_empty() {
            return Vec::new();
        }

        let mut notices = Vec::new();
        for (agent_id, section) in &self.agent {
            if !section.slack.as_ref().is_some_and(|s| s.answers.channels()) {
                continue;
            }
            let mut reached: Vec<ConnectionPath> = section
                .mcp
                .iter()
                .filter_map(|r| ConnectionPath::parse(r.id()))
                .filter(|path| personal.contains(&path))
                .collect();
            for plugin in &section.plugins {
                let servers = self
                    .plugin
                    .get(plugin.id())
                    .and_then(|s| s.bundle.as_ref())
                    .map(|b| b.servers.keys())
                    .into_iter()
                    .flatten();
                for server in servers {
                    let path = ConnectionPath::PluginServer {
                        plugin: plugin.id().to_string(),
                        server: server.clone(),
                    };
                    if personal.contains(&&path) {
                        reached.push(path);
                    }
                }
            }
            for connection in &reached {
                notices.push(format!(
                    "\u{258e} [agent.{agent_id}] reaches [{connection}], which is \
                     credential = \"user\".\n\
                     \u{258e} Those tools do not work when it is mentioned in a channel.\n\
                     \u{258e} They work in direct messages."
                ));
            }
        }
        notices
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSpec {
    #[serde(rename = "type")]
    pub kind: ProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<LlmFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_ttl: Option<String>,
}

impl ProviderSpec {
    pub fn block(&self) -> LlmBlock {
        match self.kind {
            ProviderKind::Worker => LlmBlock::worker(self.format),
            _ => LlmBlock::engine(),
        }
    }

    pub fn api_key_env(&self) -> Option<String> {
        self.api_key_env
            .clone()
            .or_else(|| self.kind.default_api_key_env().map(str::to_string))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AgentTool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub subagents: Vec<SubagentRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subagent_tools: Option<SubagentTools>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subagent_mode: Option<SubagentMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_subagent_depth: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mcp: Vec<McpRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<PluginRef>,
    #[serde(
        default,
        deserialize_with = "crate::protocol::de_defer_tools",
        skip_serializing_if = "Option::is_none"
    )]
    pub defer_tools: Option<DeferTools>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_announce: Option<McpAnnounce>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_auth_failure: Option<McpAuthFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_tool_sync_failure: Option<McpToolSyncFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signing_secret_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slack: Option<AgentSlackConfig>,
}

impl AgentSection {
    fn declares_config(&self) -> bool {
        self.llm.is_some()
            || self.model.is_some()
            || self.system.is_some()
            || self.retry.is_some()
            || !self.tools.is_empty()
            || !self.subagents.is_empty()
            || !self.mcp.is_empty()
            || !self.plugins.is_empty()
    }

    pub fn to_agent_config(&self, manifest: &Manifest) -> Option<AgentConfig> {
        Some(AgentConfig {
            llm: self.llm.clone(),
            model: self.model.clone()?,
            system: self.system.clone(),
            effort: self.effort,
            retry: self.retry.clone().map(Box::new),
            tools: self.tools.clone(),
            subagents: self.to_subagents(manifest),
            subagent_tools: self.subagent_tools,
            max_subagent_depth: self.max_subagent_depth.or(manifest.max_subagent_depth),
            mcp: self
                .mcp
                .iter()
                .filter_map(|m| m.to_server(self.mcp_defaults()))
                .collect(),
            defer_tools: self.defer_tools,
            mcp_announce: self.mcp_announce.unwrap_or_default(),
            plugins: self
                .plugins
                .iter()
                .map(|p| p.to_wire(manifest, self.mcp_defaults()))
                .collect(),
        })
    }

    fn mcp_defaults(&self) -> McpDefaults {
        McpDefaults {
            auth_failure: self.mcp_auth_failure.unwrap_or_default(),
            tool_sync_failure: self.mcp_tool_sync_failure.unwrap_or_default(),
        }
    }

    fn to_subagents(&self, manifest: &Manifest) -> Vec<Subagent> {
        self.subagents
            .iter()
            .map(|sub| {
                let path = sub.id();
                let id = match ConnectionPath::parse(path) {
                    Some(ConnectionPath::Agent(id)) => id,
                    _ => path.to_string(),
                };
                Subagent {
                    description: manifest
                        .agent
                        .get(&id)
                        .and_then(|s| s.description.clone())
                        .unwrap_or_default(),
                    id,
                    defer: sub.defer(),
                    prefix: sub.prefix(),
                    mode: sub.mode().or(self.subagent_mode),
                }
            })
            .collect()
    }

    pub fn to_entry(&self, manifest: &Manifest) -> AgentEntry {
        AgentEntry {
            config: self.to_agent_config(manifest),
            hosting: match self.worker.clone() {
                None => Hosting::Engine,
                Some(url) => Hosting::Http(WorkerEndpoint {
                    url,
                    signing_secret: self
                        .signing_secret_env
                        .as_deref()
                        .and_then(crate::cli::env_value),
                }),
            },
        }
    }
}

/// A plugin's directory: `~` and `$VAR` expand, an absolute path stands alone,
/// and the rest is read against the file that names it.
fn plugin_dir(base: &std::path::Path, path: &str) -> Result<std::path::PathBuf> {
    let expanded = shellexpand::full(path).with_context(|| format!("`path = {path:?}`"))?;
    Ok(base.join(&*expanded))
}

#[derive(Debug, Default)]
pub struct ResolvedPlugins {
    pub notices: Vec<String>,
    pub pending: BTreeMap<String, Vec<crate::plugins::Pending>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PluginSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle: Option<PluginBundle>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, PluginServerSpec>,
    #[serde(default, deserialize_with = "saw_auth_map", skip_serializing)]
    pub auth: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

fn saw_auth_map<'de, D: serde::Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    <serde::de::IgnoredAny as Deserialize>::deserialize(d)?;
    Ok(true)
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PluginServerSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential: Option<CredentialScope>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_tools: Option<bool>,
}

impl PluginServerSpec {
    fn over(&self, base: &BundleServer) -> ConnectionDecl {
        ConnectionDecl {
            url: self.url.clone().unwrap_or_else(|| base.url.clone()),
            auth: self.auth,
            header: self.header.clone(),
            credential: self.credential,
            scopes: self.scopes.clone(),
            client_id_env: self.client_id_env.clone(),
            client_secret_env: self.client_secret_env.clone(),
            prefix_tools: self.prefix_tools.unwrap_or(true),
        }
    }
}

fn check_plugin(id: &str, spec: &PluginSpec) -> Result<()> {
    check_id(id)?;
    if spec.path.is_none() && spec.bundle.is_none() && spec.hash.is_none() {
        bail!("declares nothing. Set `path` to the plugin's directory.");
    }
    if spec.auth {
        bail!(
            "`auth` is a section now: write `[plugin.{id}.mcp.<server>]` with `auth = \"token\"`"
        );
    }
    let Some(bundle) = &spec.bundle else {
        return Ok(());
    };
    for name in spec.mcp.keys() {
        if !bundle.servers.contains_key(name) {
            bail!(
                "`[plugin.{id}.mcp.{name}]` names no server `{name}`. The plugin's servers: {}",
                declared(bundle.servers.keys())
            );
        }
    }
    for (name, server) in &bundle.servers {
        check_id(name).map_err(|e| anyhow::anyhow!("server `{name}`: {e}"))?;
        let resolved = spec.mcp.get(name).cloned().unwrap_or_default().over(server);
        check_url(&resolved.url).map_err(|e| anyhow::anyhow!("server `{name}`: {e}"))?;
        check_connection(&resolved).map_err(|e| anyhow::anyhow!("server `{name}`: {e}"))?;
    }
    Ok(())
}

fn unresolved(path: &ConnectionPath, manifest: &Manifest) -> bool {
    let ConnectionPath::PluginServer { plugin, .. } = path else {
        return false;
    };
    manifest
        .plugin
        .get(plugin)
        .is_some_and(|spec| spec.bundle.is_none())
}

fn check_tool_prefixes(manifest: &Manifest) -> Result<()> {
    let paths = manifest.connection_paths();
    for (i, path) in paths.iter().enumerate() {
        for other in &paths[i + 1..] {
            if path.tool_prefix() == other.tool_prefix() {
                bail!(
                    "`{path}` and `{other}` both prefix their tools with `{}`. Rename one.",
                    path.tool_prefix()
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum McpRef {
    All(String),
    Filtered(McpEntry),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SubagentRef {
    Id(String),
    Configured(SubagentEntry),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubagentEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defer: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<SubagentMode>,
}

impl SubagentRef {
    pub fn id(&self) -> &str {
        match self {
            Self::Id(id) => id,
            Self::Configured(entry) => &entry.id,
        }
    }

    fn defer(&self) -> Option<bool> {
        match self {
            Self::Id(_) => None,
            Self::Configured(entry) => entry.defer,
        }
    }

    fn prefix(&self) -> Option<bool> {
        match self {
            Self::Id(_) => None,
            Self::Configured(entry) => entry.prefix,
        }
    }

    fn mode(&self) -> Option<SubagentMode> {
        match self {
            Self::Id(_) => None,
            Self::Configured(entry) => entry.mode,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpToolsEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_failure: Option<McpAuthFailure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_sync_failure: Option<McpToolSyncFailure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approve: Option<Approve>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct McpDefaults {
    pub auth_failure: McpAuthFailure,
    pub tool_sync_failure: McpToolSyncFailure,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpToolsEntry {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_destructive: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotent: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defer: Option<bool>,
}

impl McpToolsEntry {
    fn to_wire(&self) -> McpTools {
        McpTools {
            include: self.include.clone(),
            exclude: self.exclude.clone(),
            read_only: self.read_only,
            non_destructive: self.non_destructive,
            idempotent: self.idempotent,
            defer: self.defer,
        }
    }
}

impl McpRef {
    pub fn id(&self) -> &str {
        match self {
            Self::All(id) => id,
            Self::Filtered(entry) => &entry.id,
        }
    }

    fn to_server(&self, defaults: McpDefaults) -> Option<McpServer> {
        Some(match self {
            Self::All(written) => McpServer {
                path: ConnectionPath::parse(written)?,
                tools: None,
                auth_failure: defaults.auth_failure,
                tool_sync_failure: defaults.tool_sync_failure,
                approve: Approve::default(),
            },
            Self::Filtered(entry) => McpServer {
                path: ConnectionPath::parse(&entry.id)?,
                tools: entry.tools.as_ref().map(McpToolsEntry::to_wire),
                auth_failure: entry.auth_failure.unwrap_or(defaults.auth_failure),
                tool_sync_failure: entry
                    .tool_sync_failure
                    .unwrap_or(defaults.tool_sync_failure),
                approve: entry.approve.unwrap_or_default(),
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PluginRef {
    All(String),
    Configured(PluginEntry),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpToolsEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_failure: Option<McpAuthFailure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_sync_failure: Option<McpToolSyncFailure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approve: Option<Approve>,
}

impl PluginRef {
    pub fn id(&self) -> &str {
        match self {
            Self::All(id) => id,
            Self::Configured(entry) => &entry.id,
        }
    }

    fn to_wire(&self, manifest: &Manifest, defaults: McpDefaults) -> AgentPlugin {
        let id = self.id();
        let bundle = manifest.plugin.get(id).and_then(|s| s.bundle.as_ref());
        let (description, skills, servers) = match bundle {
            Some(b) => (
                b.description.clone(),
                b.skill_metas(),
                b.servers
                    .keys()
                    .map(|name| ConnectionPath::PluginServer {
                        plugin: id.to_string(),
                        server: name.clone(),
                    })
                    .collect(),
            ),
            None => (String::new(), Vec::new(), Vec::new()),
        };
        let entry = match self {
            Self::All(_) => None,
            Self::Configured(entry) => Some(entry),
        };
        AgentPlugin {
            id: id.to_string(),
            description,
            skills,
            servers,
            tools: entry
                .and_then(|e| e.tools.as_ref())
                .map(McpToolsEntry::to_wire),
            auth_failure: entry
                .and_then(|e| e.auth_failure)
                .unwrap_or(defaults.auth_failure),
            tool_sync_failure: entry
                .and_then(|e| e.tool_sync_failure)
                .unwrap_or(defaults.tool_sync_failure),
            approve: entry.and_then(|e| e.approve).unwrap_or_default(),
        }
    }
}

impl Serialize for PluginRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::All(id) => s.serialize_str(id),
            Self::Configured(entry) => entry.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for PluginRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = PluginRef;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a plugin id, or a table with `id`")
            }

            fn visit_str<E: serde::de::Error>(self, id: &str) -> Result<PluginRef, E> {
                Ok(PluginRef::All(id.to_string()))
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                map: M,
            ) -> Result<PluginRef, M::Error> {
                PluginEntry::deserialize(serde::de::value::MapAccessDeserializer::new(map))
                    .map(PluginRef::Configured)
            }
        }

        d.deserialize_any(V)
    }
}

impl Serialize for McpRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::All(id) => s.serialize_str(id),
            Self::Filtered(entry) => entry.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for McpRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = McpRef;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a connection id, or a table with `id` and `tools`")
            }

            fn visit_str<E: serde::de::Error>(self, id: &str) -> Result<McpRef, E> {
                Ok(McpRef::All(id.to_string()))
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(self, map: M) -> Result<McpRef, M::Error> {
                McpEntry::deserialize(serde::de::value::MapAccessDeserializer::new(map))
                    .map(McpRef::Filtered)
            }
        }

        d.deserialize_any(V)
    }
}

impl Serialize for SubagentRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Id(id) => s.serialize_str(id),
            Self::Configured(entry) => entry.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for SubagentRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = SubagentRef;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str(
                    "an `agent.<id>` path, or a table with `id`, `defer`, `prefix`, and `mode`",
                )
            }

            fn visit_str<E: serde::de::Error>(self, id: &str) -> Result<SubagentRef, E> {
                Ok(SubagentRef::Id(id.to_string()))
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                map: M,
            ) -> Result<SubagentRef, M::Error> {
                SubagentEntry::deserialize(serde::de::value::MapAccessDeserializer::new(map))
                    .map(SubagentRef::Configured)
            }
        }

        d.deserialize_any(V)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentSlackConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "SlackAudience::is_default")]
    pub answers: SlackAudience,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SlackAudience {
    #[default]
    Both,
    Dm,
    Channels,
}

impl SlackAudience {
    fn is_default(&self) -> bool {
        *self == Self::Both
    }

    pub const ALL: [Self; 3] = [Self::Both, Self::Dm, Self::Channels];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::Dm => "dm",
            Self::Channels => "channels",
        }
    }

    pub fn parse(v: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.as_str() == v)
    }

    pub fn dms(self) -> bool {
        matches!(self, Self::Both | Self::Dm)
    }

    pub fn channels(self) -> bool {
        matches!(self, Self::Both | Self::Channels)
    }
}

impl AgentSlackConfig {
    pub fn name(&self, agent_id: &str) -> String {
        match self
            .name
            .as_deref()
            .map(str::trim)
            .filter(|n| !n.is_empty())
        {
            Some(name) => name.to_string(),
            None => agent_id.to_string(),
        }
    }
}

pub fn check_llm(id: &str, spec: &ProviderSpec) -> Result<()> {
    check_id(id)?;
    match spec.kind {
        ProviderKind::Worker => {
            if spec.api_key_env.is_some() || spec.base_url.is_some() {
                bail!(
                    "a `worker` block needs no `api_key_env` or `base_url`: the call never \
                     leaves your worker"
                );
            }
            if spec.cache_ttl.is_some() {
                bail!("a `worker` block needs no `cache_ttl`: the call never leaves your worker");
            }
        }
        _ => {
            if spec.format.is_some() {
                bail!(
                    "`format` is the wire shape of an `llm.execute`, so it only applies to \
                     `type = \"worker\"`"
                );
            }
            if let Some(ttl) = &spec.cache_ttl {
                let allowed = cache_ttls(spec.kind);
                if !allowed.contains(&ttl.as_str()) {
                    bail!(
                        "`cache_ttl = \"{ttl}\"` is not a life {} holds a prompt for. Use {}.",
                        spec.kind.as_str(),
                        allowed
                            .iter()
                            .map(|t| format!("`{t}`"))
                            .collect::<Vec<_>>()
                            .join(" or ")
                    );
                }
            }
        }
    }
    Ok(())
}

pub fn cache_ttls(kind: ProviderKind) -> &'static [&'static str] {
    match kind {
        ProviderKind::Anthropic | ProviderKind::Openrouter => &["5m", "1h"],
        ProviderKind::Openai => &["in_memory", "24h"],
        ProviderKind::Worker => &[],
    }
}

pub fn check_agent(id: &str, section: &AgentSection, manifest: &Manifest) -> Result<()> {
    check_id(id)?;

    if let Some(url) = &section.worker {
        check_url(url)?;
    } else if section.signing_secret_env.is_some() {
        bail!(
            "`signing_secret_env` signs decision requests, and there is no `worker` to send any to"
        );
    }

    if let Some(slack) = &section.slack {
        check_agent_slack(id, slack).map_err(|e| anyhow::anyhow!("`slack`: {e}"))?;
    }

    if !section.declares_config() {
        if section.worker.is_none() {
            bail!(
                "declares nothing. An agent the engine decides for needs an `llm` and a \
                 `model` to propose from; an agent whose worker authors its config needs a \
                 `worker` URL."
            );
        }
        return Ok(());
    }

    let Some(llm) = section.llm.as_deref() else {
        bail!(
            "no `llm`. Name one of the declared blocks: {}",
            declared(manifest.llm.keys())
        );
    };
    let Some(block) = manifest.llm.get(llm) else {
        bail!(
            "`llm = \"{llm}\"` names no block. Declared: {}",
            declared(manifest.llm.keys())
        );
    };
    if section.model.is_none() {
        bail!("no `model`. An agent that declares an `llm` has to say which model on it.");
    }

    let connections = manifest.connections();
    let via_plugins: BTreeMap<ConnectionPath, &str> = section
        .plugins
        .iter()
        .filter_map(|p| Some((p.id(), manifest.plugin.get(p.id())?.bundle.as_ref()?)))
        .flat_map(|(plugin, bundle)| {
            bundle.servers.keys().map(move |server| {
                let path = ConnectionPath::PluginServer {
                    plugin: plugin.to_string(),
                    server: server.clone(),
                };
                (path, plugin)
            })
        })
        .collect();

    for server in &section.mcp {
        let written = server.id();
        let Some(path) = ConnectionPath::parse(written) else {
            if let Some(known) = connections.keys().find(|p| p.tool_prefix() == *written) {
                bail!(
                    "`mcp` names `{written}`, and a connection is named by where it is \
                     declared. Write `{known}`."
                );
            }
            bail!(
                "`mcp` names no connection `{written}`. Declared: {}",
                list(&connections)
            );
        };
        if !connections.contains_key(&path) && !unresolved(&path, manifest) {
            bail!(
                "`mcp` names no connection `{path}`. Declared: {}",
                list(&connections)
            );
        }
        if let Some(plugin) = via_plugins.get(&path) {
            bail!(
                "`mcp` names `{path}`, which `plugins = [\"{plugin}\"]` already grants. \
                 Name it once."
            );
        }
    }

    for plugin in &section.plugins {
        if !manifest.plugin.contains_key(plugin.id()) {
            bail!(
                "`plugins` names no plugin `{}`. Declared: {}",
                plugin.id(),
                declared(manifest.plugin.keys())
            );
        }
    }

    for sub in &section.subagents {
        check_subagent(sub, section, manifest).map_err(|e| anyhow::anyhow!("`subagents`: {e}"))?;
    }

    if !section.subagents.is_empty()
        && SubagentTools::strategy_of(section.subagent_tools) == SubagentToolsStrategy::Single
        && section.tools.iter().any(|t| t.name == SUBAGENT)
    {
        return Err(reserved_by_single(SUBAGENT));
    }

    if offers_wait_tool(section) && section.tools.iter().any(|t| t.name == SUBAGENT_WAIT) {
        bail!(
            "`{SUBAGENT_WAIT}` is the tool the engine offers for detached subagents, and the \
             model sees one namespace for both. Rename the tool, or set \
             `subagent_tools = {{ wait = false }}`."
        );
    }

    for tool in &section.tools {
        if tool.handler != Some(Handler::Client) {
            bail!(
                "tool `{}` needs `handler = \"client\"`: a file can only declare tools the \
                 browser runs, and worker-run tools come from worker code",
                tool.name
            );
        }
    }

    if section.worker.is_none() && block.kind == ProviderKind::Worker {
        bail!("`llm = \"{llm}\"` is a `worker` block, so this agent needs a `worker` to run its calls");
    }
    Ok(())
}

fn offers_wait_tool(section: &AgentSection) -> bool {
    SubagentTools::wait_of(section.subagent_tools)
        && section.subagents.iter().any(|sub| {
            sub.mode()
                .or(section.subagent_mode)
                .unwrap_or_default()
                .offered()
                .contains(&SpawnMode::Detached)
        })
}

fn check_subagent(sub: &SubagentRef, section: &AgentSection, manifest: &Manifest) -> Result<()> {
    let written = sub.id();
    let id = match ConnectionPath::parse(written) {
        Some(ConnectionPath::Agent(id)) => id,
        _ => {
            if manifest.agent.contains_key(written) {
                bail!(
                    "`{written}` names an agent, and an agent is named by where it is declared. \
                     Write `agent.{written}`."
                );
            }
            bail!(
                "`{written}` names no agent. Declared: {}",
                agent_paths(manifest)
            );
        }
    };
    let Some(child) = manifest.agent.get(&id) else {
        bail!(
            "`{written}` names no agent. Declared: {}",
            agent_paths(manifest)
        );
    };
    let single =
        SubagentTools::strategy_of(section.subagent_tools) == SubagentToolsStrategy::Single;
    if single {
        if sub.prefix().is_some() {
            bail!("`prefix` shapes a per-agent tool, and the strategy is `single`");
        }
        if sub.defer().is_some() {
            bail!(
                "`defer` hides a per-agent tool, and the strategy is `single`. Set \
                 `defer_tools` on the agent to hide `{SUBAGENT}`."
            );
        }
    }
    let wire = Subagent {
        id: id.clone(),
        description: String::new(),
        defer: sub.defer(),
        prefix: sub.prefix(),
        mode: None,
    };
    let offered = wire.offered_name();
    if section.tools.iter().any(|t| t.name == offered) {
        bail!("`{offered}` is also a tool name, and the model sees one namespace for both");
    }
    if single && offered == SUBAGENT {
        return Err(reserved_by_single(&offered));
    }
    if offers_wait_tool(section) && offered == SUBAGENT_WAIT {
        bail!(
            "`{offered}` is the tool the engine offers for detached subagents. Set `prefix`, \
             or `subagent_tools = {{ wait = false }}`."
        );
    }
    let defers = wire.defers(section.defer_tools.is_some());
    if !single && !defers && offered.len() > crate::connectors::filter::MAX_NAME {
        bail!(
            "`{offered}` is longer than the {} characters a provider accepts; shorten the id, \
             or set `defer = true`",
            crate::connectors::filter::MAX_NAME
        );
    }
    if child.description.is_none() && section.worker.is_none() {
        bail!(
            "`{id}` has no description, and there is no `worker` to supply one. Set \
             `description` on [agent.{id}]."
        );
    }
    Ok(())
}

fn reserved_by_single(offered: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "`{offered}` is the tool the `single` strategy offers, and the model sees one \
         namespace for both"
    )
}

fn agent_paths(manifest: &Manifest) -> String {
    let paths: Vec<String> = manifest
        .agent
        .keys()
        .map(|id| format!("agent.{id}"))
        .collect();
    declared(paths.iter())
}

const SLACK_APP_NAME_MAX: usize = 35;
const SLACK_APP_DESCRIPTION_MAX: usize = 140;

pub fn check_agent_slack(id: &str, slack: &AgentSlackConfig) -> Result<()> {
    let name = slack.name(id);
    if name.chars().count() > SLACK_APP_NAME_MAX {
        bail!("`name` is longer than the {SLACK_APP_NAME_MAX} characters Slack allows");
    }
    if let Some(description) = &slack.description {
        if description.chars().count() > SLACK_APP_DESCRIPTION_MAX {
            bail!(
                "`description` is longer than the {SLACK_APP_DESCRIPTION_MAX} characters Slack \
                 allows"
            );
        }
    }
    Ok(())
}

fn list(connections: &BTreeMap<ConnectionPath, ConnectionSpec>) -> String {
    let paths: Vec<String> = connections.keys().map(ConnectionPath::to_string).collect();
    match paths.is_empty() {
        true => "none".to_string(),
        false => paths.join(", "),
    }
}

pub fn declared<'a>(mut ids: impl Iterator<Item = &'a String>) -> String {
    let joined = ids.by_ref().cloned().collect::<Vec<_>>().join(", ");
    match joined.is_empty() {
        true => "none".to_string(),
        false => joined,
    }
}

pub fn check_id(id: &str) -> Result<()> {
    if id.is_empty() {
        bail!("the id is empty");
    }
    if let Some(c) = id
        .chars()
        .find(|c| !c.is_ascii_alphanumeric() && *c != '_' && *c != '-')
    {
        bail!("`{id}` cannot prefix a tool name: `{c}` is not a letter, digit, `_`, or `-`");
    }
    Ok(())
}

fn check_connection(spec: &ConnectionDecl) -> Result<()> {
    if spec.header.is_some() && spec.auth != Some(AuthKind::Token) {
        bail!("`header` carries a static token, so it needs `auth = \"token\"`");
    }
    if spec.client_secret_env.is_some() && spec.client_id_env.is_none() {
        bail!("`client_secret_env` needs `client_id_env`");
    }
    if spec.auth == Some(AuthKind::Token) && !spec.scopes.is_empty() {
        bail!("`scopes` is asked for at consent, which `auth = \"token\"` does not do");
    }
    Ok(())
}

pub fn check_url(url: &str) -> Result<()> {
    reqwest::Url::parse(url).with_context(|| format!("`{url}` is not a URL"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::DeferToolsStrategy;

    fn manifest(toml: &str) -> Manifest {
        toml::from_str(toml).unwrap()
    }

    fn server(url: &str) -> BundleServer {
        BundleServer { url: url.into() }
    }

    fn decl(url: &str) -> ConnectionDecl {
        ConnectionDecl {
            url: url.into(),
            auth: None,
            header: None,
            credential: None,
            scopes: Vec::new(),
            client_id_env: None,
            client_secret_env: None,
            prefix_tools: true,
        }
    }

    fn path(written: &str) -> ConnectionPath {
        ConnectionPath::parse(written).expect("a path")
    }

    #[test]
    fn the_wire_copy_carries_no_local_binding() {
        let m = manifest(
            r#"
            [llm.claude]
            type = "anthropic"
            api_key_env = "MY_KEY"

            [agent.support]
            llm = "claude"
            model = "claude-sonnet-4-5"
            worker = "https://bot.example.com/agent"
            signing_secret_env = "MY_SECRET"

            [mcp.gmail]
            url = "https://gmailmcp.googleapis.com/mcp/v1"
            scopes = ["https://www.googleapis.com/auth/gmail.modify"]
            client_id_env = "MY_CLIENT"
            client_secret_env = "MY_CLIENT_SECRET"
            "#,
        );
        assert_eq!(
            m.local_bindings(),
            [
                "[llm.claude].api_key_env",
                "[agent.support].signing_secret_env",
                "[mcp.gmail].client_id_env",
                "[mcp.gmail].client_secret_env",
            ]
        );

        let wire = m.for_wire();
        assert!(wire.local_bindings().is_empty());
        assert_eq!(wire.agent["support"].worker, m.agent["support"].worker);
        assert_eq!(wire.llm["claude"].kind, ProviderKind::Anthropic);
        assert_eq!(wire.mcp["gmail"].scopes, m.mcp["gmail"].scopes);
    }

    #[test]
    fn a_client_needs_both_halves_and_a_token_asks_for_no_scopes() {
        let named = manifest(
            r#"
            [mcp.gmail]
            url = "https://gmailmcp.googleapis.com/mcp/v1"
            client_secret_env = "MY_CLIENT_SECRET"
            "#,
        );
        let err = named.validate().unwrap_err().to_string();
        assert!(err.contains("needs `client_id_env`"), "{err}");

        let static_token = manifest(
            r#"
            [mcp.thing]
            url = "https://thing.example.test/mcp"
            auth = "token"
            scopes = ["read"]
            "#,
        );
        let err = static_token.validate().unwrap_err().to_string();
        assert!(err.contains("does not do"), "{err}");
    }

    #[test]
    fn a_cache_life_is_read_against_the_block_it_sits_on() {
        let ok = manifest(
            r#"
            [llm.claude]
            type = "anthropic"
            cache_ttl = "1h"

            [llm.gpt]
            type = "openai"
            cache_ttl = "24h"

            [llm.router]
            type = "openrouter"
            cache_ttl = "5m"
            "#,
        );
        ok.validate().unwrap();

        let wrong_vendor = manifest(
            r#"
            [llm.gpt]
            type = "openai"
            cache_ttl = "1h"
            "#,
        );
        let err = wrong_vendor.validate().unwrap_err().to_string();
        assert!(err.contains("[llm.gpt]"), "{err}");
        assert!(err.contains("`in_memory` or `24h`"), "{err}");

        let on_a_worker = manifest(
            r#"
            [llm.mine]
            type = "worker"
            cache_ttl = "1h"
            "#,
        );
        let err = on_a_worker.validate().unwrap_err().to_string();
        assert!(err.contains("never leaves your worker"), "{err}");
    }

    #[test]
    fn validation_is_the_same_on_both_ends() {
        let bad = manifest(
            r#"
            [agent.support]
            llm = "nope"
            model = "m"
            "#,
        );
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("[agent.support]"), "{err}");
        assert!(err.contains("names no block"), "{err}");
    }

    fn team(assistant_extra: &str, poet_extra: &str) -> Manifest {
        manifest(&format!(
            r#"
            [llm.claude]
            type = "anthropic"

            [agent.assistant]
            llm = "claude"
            model = "m"
            {assistant_extra}

            [agent.poet]
            llm = "claude"
            model = "m"
            {poet_extra}
            "#
        ))
    }

    #[test]
    fn a_subagent_takes_its_description_from_the_agent_it_names() {
        let m = team(
            r#"subagents = ["agent.poet"]"#,
            r#"description = "Writes a haiku.""#,
        );
        m.validate().unwrap();
        let agents = m.agents();
        let subs = &agents["assistant"]
            .config
            .as_ref()
            .expect("seeded")
            .subagents;
        assert_eq!(
            subs,
            &[Subagent {
                defer: None,
                prefix: None,
                mode: None,
                id: "poet".to_string(),
                description: "Writes a haiku.".to_string(),
            }],
            "declared once, on the agent it describes"
        );
    }

    #[test]
    fn a_subagent_mode_comes_from_the_entry_or_the_agent_default() {
        let m = team(
            r#"
            subagent_mode = "detached"
            subagents = ["agent.poet", { id = "agent.poet2", mode = "any" }]
            "#
            .trim(),
            r#"description = "Writes a haiku.""#,
        );
        let m = {
            let mut m = m;
            let poet = m.agent["poet"].clone();
            m.agent.insert("poet2".to_string(), poet);
            m
        };
        m.validate().unwrap();
        let agents = m.agents();
        let subs = &agents["assistant"]
            .config
            .as_ref()
            .expect("seeded")
            .subagents;
        assert_eq!(
            subs.iter().map(|s| s.mode).collect::<Vec<_>>(),
            vec![Some(SubagentMode::Detached), Some(SubagentMode::Any)],
            "the entry's own mode wins over the agent's default"
        );
    }

    #[test]
    fn a_wait_mode_does_not_parse_as_configuration() {
        for section in [
            r#"subagents = [{ id = "agent.poet", mode = "wait" }]"#,
            r#"subagent_mode = "wait""#,
        ] {
            let err = toml::from_str::<Manifest>(&format!(
                r#"
                [agent.assistant]
                {section}
                "#
            ))
            .unwrap_err()
            .to_string();
            assert!(err.contains("unknown variant `wait`"), "{err}");
        }
    }

    #[test]
    fn max_subagent_depth_folds_into_the_config_and_the_agent_s_own_wins() {
        let m = manifest(
            r#"
            max_subagent_depth = 1

            [llm.claude]
            type = "anthropic"

            [agent.assistant]
            llm = "claude"
            model = "m"

            [agent.poet]
            llm = "claude"
            model = "m"
            max_subagent_depth = 5
            "#,
        );
        let agents = m.agents();
        let depth = |id: &str| {
            agents[id]
                .config
                .as_ref()
                .expect("seeded")
                .max_subagent_depth
        };
        assert_eq!(depth("assistant"), Some(1), "the top level fills in");
        assert_eq!(depth("poet"), Some(5), "the agent's own value wins");
    }

    #[test]
    fn an_unset_max_subagent_depth_stays_unset() {
        let m = team("", "");
        assert_eq!(
            m.agents()["assistant"]
                .config
                .as_ref()
                .expect("seeded")
                .max_subagent_depth,
            None,
            "the engine default applies at run time, not in the file"
        );
    }

    #[test]
    fn a_subagent_names_a_declared_agent() {
        let bad = team(r#"subagents = ["agent.potet"]"#, r#"description = "d""#);
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("[agent.assistant]"), "{err}");
        assert!(err.contains("names no agent"), "{err}");
        assert!(err.contains("agent.poet"), "{err}");
    }

    #[test]
    fn a_bare_id_points_at_the_agent_path() {
        let bad = team(r#"subagents = ["poet"]"#, r#"description = "d""#);
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("Write `agent.poet`"), "{err}");
    }

    #[test]
    fn a_connection_path_is_not_an_agent() {
        let bad = team(r#"subagents = ["mcp.poet"]"#, r#"description = "d""#);
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("`mcp.poet` names no agent"), "{err}");
    }

    #[test]
    fn a_subagent_cannot_take_a_tool_s_name() {
        let bad = team(
            r#"subagents = ["agent.poet"]
               tools = [{ name = "poet", description = "d", handler = "client" }]"#,
            r#"description = "d""#,
        );
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("one namespace"), "{err}");
    }

    #[test]
    fn a_subagent_entry_takes_a_map_with_defer_and_prefix() {
        let m = team(
            r#"subagents = [{ id = "agent.poet", defer = true, prefix = true }]"#,
            r#"description = "Writes a haiku.""#,
        );
        m.validate().unwrap();
        let agents = m.agents();
        let subs = &agents["assistant"]
            .config
            .as_ref()
            .expect("seeded")
            .subagents;
        assert_eq!(
            subs,
            &[Subagent {
                id: "poet".to_string(),
                description: "Writes a haiku.".to_string(),
                defer: Some(true),
                prefix: Some(true),
                mode: None,
            }]
        );
    }

    #[test]
    fn subagent_tools_reads_the_strategy_from_the_agent() {
        let m = team(
            r#"subagents = ["agent.poet"]
               subagent_tools = { strategy = "single" }"#,
            r#"description = "Writes a haiku.""#,
        );
        m.validate().unwrap();
        let agents = m.agents();
        let config = agents["assistant"].config.as_ref().expect("seeded");
        assert_eq!(
            config.subagent_tools,
            Some(SubagentTools {
                strategy: SubagentToolsStrategy::Single,
                wait: None,
            })
        );
        assert_eq!(config.subagent_strategy(), SubagentToolsStrategy::Single);
    }

    #[test]
    fn an_absent_subagent_tools_offers_one_tool_per_agent() {
        let m = team(
            r#"subagents = ["agent.poet"]"#,
            r#"description = "Writes a haiku.""#,
        );
        let agents = m.agents();
        let config = agents["assistant"].config.as_ref().expect("seeded");
        assert_eq!(config.subagent_tools, None);
        assert_eq!(config.subagent_strategy(), SubagentToolsStrategy::PerAgent);
    }

    #[test]
    fn an_unknown_subagent_tools_key_is_rejected() {
        let raw = r#"
            [agent.a]
            subagent_tools = { strategy = "single", extra = true }
            "#;
        let err = toml::from_str::<Manifest>(raw).unwrap_err().to_string();
        assert!(err.contains("extra"), "{err}");
    }

    #[test]
    fn the_single_strategy_rejects_a_per_agent_shape() {
        for (sub, said) in [
            (
                r#"{ id = "agent.poet", prefix = true }"#,
                "`prefix` shapes a per-agent tool",
            ),
            (
                r#"{ id = "agent.poet", defer = true }"#,
                "`defer` hides a per-agent tool",
            ),
        ] {
            let bad = team(
                &format!(
                    r#"subagents = [{sub}]
                       subagent_tools = {{ strategy = "single" }}"#
                ),
                r#"description = "d""#,
            );
            let err = bad.validate().unwrap_err().to_string();
            assert!(err.contains(said), "{sub}: {err}");
            assert!(err.contains("the strategy is `single`"), "{sub}: {err}");
        }
    }

    #[test]
    fn the_single_strategy_keeps_the_subagent_name_for_itself() {
        let bad = team(
            r#"subagents = ["agent.poet"]
               subagent_tools = { strategy = "single" }
               tools = [{ name = "subagent", description = "d", handler = "client" }]"#,
            r#"description = "d""#,
        );
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("one namespace"), "{err}");
        assert!(err.contains("subagent"), "{err}");

        team(
            r#"subagents = ["agent.poet"]
               tools = [{ name = "subagent", description = "d", handler = "client" }]"#,
            r#"description = "d""#,
        )
        .validate()
        .expect("per_agent offers no `subagent` tool, so the name is free");
    }

    #[test]
    fn a_subagent_named_subagent_collides_with_the_single_tool() {
        let raw = r#"
            [llm.claude]
            type = "anthropic"

            [agent.assistant]
            llm = "claude"
            model = "m"
            subagents = ["agent.subagent"]
            subagent_tools = { strategy = "single" }

            [agent.subagent]
            llm = "claude"
            model = "m"
            description = "d"
            "#;
        let err = manifest(raw).validate().unwrap_err().to_string();
        assert!(err.contains("one namespace"), "{err}");
    }

    #[test]
    fn the_single_strategy_ignores_the_offered_name_length() {
        let long = "p".repeat(70);
        let raw = format!(
            r#"
            [llm.claude]
            type = "anthropic"

            [agent.assistant]
            llm = "claude"
            model = "m"
            subagents = ["agent.{long}"]
            subagent_tools = {{ strategy = "single" }}

            [agent.{long}]
            llm = "claude"
            model = "m"
            description = "d"
            "#
        );
        manifest(&raw)
            .validate()
            .expect("no tool is named after the agent, so its id may be long");
    }

    #[test]
    fn an_unknown_subagent_key_is_rejected() {
        let raw = r#"
            [agent.a]
            subagents = [{ id = "agent.b", approve = true }]
            "#;
        let err = toml::from_str::<Manifest>(raw).unwrap_err().to_string();
        assert!(err.contains("approve"), "{err}");
    }

    #[test]
    fn a_prefixed_subagent_frees_the_bare_name_and_claims_the_prefixed_one() {
        team(
            r#"subagents = [{ id = "agent.poet", prefix = true }]
               tools = [{ name = "poet", description = "d", handler = "client" }]"#,
            r#"description = "d""#,
        )
        .validate()
        .expect("the offered name is `agent__poet`, so `poet` is free");

        let bad = team(
            r#"subagents = [{ id = "agent.poet", prefix = true }]
               tools = [{ name = "agent__poet", description = "d", handler = "client" }]"#,
            r#"description = "d""#,
        );
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("agent__poet"), "{err}");
        assert!(err.contains("one namespace"), "{err}");
    }

    #[test]
    fn an_over_long_offered_name_is_an_error_unless_it_defers() {
        let long = "p".repeat(70);
        let manifest_with = |sub: &str| {
            manifest(&format!(
                r#"
                [llm.claude]
                type = "anthropic"

                [agent.assistant]
                llm = "claude"
                model = "m"
                subagents = [{sub}]

                [agent.{long}]
                llm = "claude"
                model = "m"
                description = "d"
                "#
            ))
        };
        let err = manifest_with(&format!(r#""agent.{long}""#))
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("64 characters"), "{err}");
        manifest_with(&format!(r#"{{ id = "agent.{long}", defer = true }}"#))
            .validate()
            .expect("a name that never reaches the request has nothing to fit");
    }

    #[test]
    fn an_engine_hosted_parent_needs_its_teammate_described() {
        let bad = team(r#"subagents = ["agent.poet"]"#, "");
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("no description"), "{err}");

        team(
            r#"subagents = ["agent.poet"]
               worker = "https://bot.example.com/agent""#,
            "",
        )
        .validate()
        .expect("the worker can supply it");
    }

    #[test]
    fn a_description_alone_does_not_seed_a_config() {
        let m = manifest(
            r#"
            [agent.poet]
            description = "Writes a haiku."
            worker = "https://bot.example.com/agent"
            "#,
        );
        m.validate().unwrap();
        assert!(
            m.agents()["poet"].config.is_none(),
            "the worker still authors the config"
        );
    }

    fn connected(mcp: &str) -> Result<Manifest, toml::de::Error> {
        toml::from_str(&format!(
            r#"
            [llm.claude]
            type = "anthropic"

            [agent.support]
            llm = "claude"
            model = "m"
            mcp = {mcp}

            [mcp.sentry]
            url = "https://mcp.sentry.dev/mcp"
            "#
        ))
    }

    #[test]
    fn a_connection_is_named_bare_or_narrowed() {
        for spelling in [r#"["mcp.sentry"]"#, r#"[{ id = "mcp.sentry" }]"#] {
            let m = connected(spelling).unwrap();
            m.validate().unwrap();
            let agents = m.agents();
            let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
            assert_eq!(mcp.len(), 1, "{spelling}");
            assert_eq!(mcp[0].path.to_string(), "mcp.sentry", "{spelling}");
            assert!(mcp[0].tools.is_none(), "no filter ⇒ everything: {spelling}");
        }

        let m = connected(r#"[{ id = "mcp.sentry", tools = { read_only = true } }]"#).unwrap();
        m.validate().unwrap();
        let agents = m.agents();
        let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
        assert_eq!(
            mcp[0].tools.as_ref().expect("a filter").read_only,
            Some(true)
        );
    }

    #[test]
    fn approve_is_read_from_the_entry_that_names_the_connection() {
        let m = connected(r#"[{ id = "mcp.sentry", approve = "destructive" }]"#).unwrap();
        m.validate().unwrap();
        let agents = m.agents();
        let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
        assert_eq!(mcp[0].approve, Approve::Destructive);

        let m = connected(r#"["mcp.sentry"]"#).unwrap();
        let agents = m.agents();
        let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
        assert_eq!(mcp[0].approve, Approve::Never, "nothing asks by default");
    }

    #[test]
    fn an_unknown_approve_value_is_an_error() {
        let err = connected(r#"[{ id = "mcp.sentry", approve = "sometimes" }]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("sometimes"), "names the value: {err}");
    }

    #[test]
    fn a_misspelled_filter_is_an_error_rather_than_no_filter() {
        let err = connected(r#"[{ id = "mcp.sentry", tool = { read_only = true } }]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("tool"), "{err}");
        assert!(err.contains("unknown field"), "names the field: {err}");
    }

    #[test]
    fn a_connection_takes_the_agents_mcp_policy_and_may_override_it() {
        let m: Manifest = toml::from_str(
            r#"
name = "p"
[llm.claude]
type = "anthropic"
[mcp.sentry]
url = "https://sentry.example/mcp"
[mcp.linear]
url = "https://linear.example/mcp"
[agent.support]
llm = "claude"
model = "m"
mcp_auth_failure = "degrade"
mcp_tool_sync_failure = "silent"
mcp = ["mcp.sentry", { id = "mcp.linear", tool_sync_failure = "warn" }]
"#,
        )
        .unwrap();
        m.validate().unwrap();
        let config = m.agents()["support"].config.clone().expect("seeded");
        let sentry = config
            .mcp
            .iter()
            .find(|s| s.path.to_string() == "mcp.sentry")
            .expect("sentry");
        assert_eq!(sentry.auth_failure, McpAuthFailure::Degrade);
        assert_eq!(sentry.tool_sync_failure, McpToolSyncFailure::Silent);
        let linear = config
            .mcp
            .iter()
            .find(|s| s.path.to_string() == "mcp.linear")
            .expect("linear");
        assert_eq!(linear.auth_failure, McpAuthFailure::Degrade);
        assert_eq!(
            linear.tool_sync_failure,
            McpToolSyncFailure::Warn,
            "the connection overrides the agent"
        );
    }

    #[test]
    fn a_plugin_takes_the_agents_mcp_policy_and_may_override_it() {
        let m: Manifest = toml::from_str(
            r#"
name = "p"
[llm.claude]
type = "anthropic"
[plugin.pdf]
path = "./pdf"
[plugin.ocr]
path = "./ocr"
[agent.support]
llm = "claude"
model = "m"
mcp_tool_sync_failure = "silent"
plugins = ["pdf", { id = "ocr", tool_sync_failure = "warn" }]
"#,
        )
        .unwrap();
        let config = m.agents()["support"].config.clone().expect("seeded");
        let pdf = config.plugins.iter().find(|p| p.id == "pdf").expect("pdf");
        assert_eq!(pdf.tool_sync_failure, McpToolSyncFailure::Silent);
        let ocr = config.plugins.iter().find(|p| p.id == "ocr").expect("ocr");
        assert_eq!(
            ocr.tool_sync_failure,
            McpToolSyncFailure::Warn,
            "the plugin overrides the agent"
        );
        assert_eq!(
            pdf.server(&ConnectionPath::PluginServer {
                plugin: "pdf".into(),
                server: "renderer".into(),
            })
            .tool_sync_failure,
            McpToolSyncFailure::Silent,
            "and each of its servers is stamped with it"
        );
    }

    #[test]
    fn an_agent_that_names_no_mcp_policy_warns() {
        let m: Manifest = toml::from_str(
            r#"
name = "p"
[llm.claude]
type = "anthropic"
[mcp.sentry]
url = "https://sentry.example/mcp"
[agent.support]
llm = "claude"
model = "m"
mcp = ["mcp.sentry"]
"#,
        )
        .unwrap();
        m.validate().unwrap();
        let config = m.agents()["support"].config.clone().expect("seeded");
        assert_eq!(config.mcp[0].tool_sync_failure, McpToolSyncFailure::Warn);
        assert_eq!(config.mcp[0].auth_failure, McpAuthFailure::Interrupt);
    }

    #[test]
    fn effort_is_read_from_the_agent_and_becomes_its_reasoning() {
        let m: Manifest = toml::from_str(
            r#"
name = "p"
[llm.claude]
type = "anthropic"
[agent.support]
llm = "claude"
model = "m"
effort = "high"
"#,
        )
        .unwrap();
        m.validate().unwrap();
        let config = m.agents()["support"].config.clone().expect("seeded");
        assert_eq!(config.effort, Some(ReasoningEffort::High));
        assert_eq!(
            config.reasoning().and_then(|r| r.effort),
            Some(ReasoningEffort::High)
        );
    }

    #[test]
    fn an_agent_that_names_no_effort_sends_no_reasoning() {
        let m: Manifest = toml::from_str(
            r#"
name = "p"
[llm.claude]
type = "anthropic"
[agent.support]
llm = "claude"
model = "m"
"#,
        )
        .unwrap();
        let config = m.agents()["support"].config.clone().expect("seeded");
        assert!(config.reasoning().is_none());
    }

    #[test]
    fn defer_tools_is_read_from_the_agent() {
        let m: Manifest = toml::from_str(
            r#"
name = "p"
[llm.claude]
type = "anthropic"
[agent.support]
llm = "claude"
model = "m"
defer_tools = { strategy = "search" }
"#,
        )
        .unwrap();
        m.validate().unwrap();
        let config = m.agents()["support"].config.clone().expect("seeded");
        assert!(config.defers_tools());
        assert_eq!(config.defer_strategy(), DeferToolsStrategy::Search);
    }

    #[test]
    fn the_bool_shorthand_takes_the_defaults() {
        let m: Manifest = toml::from_str(
            r#"
name = "p"
[llm.claude]
type = "anthropic"
[agent.support]
llm = "claude"
model = "m"
defer_tools = true
"#,
        )
        .unwrap();
        let config = m.agents()["support"].config.clone().expect("seeded");
        assert_eq!(config.defer_tools, Some(DeferTools::default()));
    }

    #[test]
    fn a_false_flag_reads_as_no_opinion() {
        let m: Manifest = toml::from_str(
            r#"
name = "p"
[llm.claude]
type = "anthropic"
[agent.support]
llm = "claude"
model = "m"
defer_tools = false
"#,
        )
        .unwrap();
        let config = m.agents()["support"].config.clone().expect("seeded");
        assert!(
            !config.defers_tools(),
            "a config can turn off what it inherits"
        );
    }

    #[test]
    fn an_unknown_strategy_value_is_an_error() {
        let err = toml::from_str::<Manifest>(
            r#"
name = "p"
[llm.claude]
type = "anthropic"
[agent.support]
llm = "claude"
model = "m"
defer_tools = { strategy = "sometimes" }
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("sometimes"), "names the value: {err}");
    }

    #[test]
    fn defer_is_read_from_the_tools_table() {
        let m = connected(r#"[{ id = "mcp.sentry", tools = { defer = true } }]"#).unwrap();
        m.validate().unwrap();
        let agents = m.agents();
        let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
        assert_eq!(mcp[0].tools.as_ref().expect("a table").defer, Some(true));

        let m = connected(r#"[{ id = "mcp.sentry", tools = { read_only = true } }]"#).unwrap();
        let agents = m.agents();
        let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
        assert!(
            mcp[0].tools.as_ref().expect("a table").defer.is_none(),
            "unset ⇒ the agent decides"
        );
    }

    #[test]
    fn a_misspelled_key_inside_the_tools_table_is_an_error() {
        let err = connected(r#"[{ id = "mcp.sentry", tools = { defr = true } }]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("defr"), "names the field: {err}");
        assert!(err.contains("unknown field"), "{err}");
    }

    #[test]
    fn a_defer_that_is_not_a_boolean_is_an_error() {
        let err = connected(r#"[{ id = "mcp.sentry", tools = { defer = "search" } }]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("boolean"), "says what it wanted: {err}");
    }

    #[test]
    fn a_bare_id_is_written_back_bare() {
        let m = connected(r#"["mcp.sentry"]"#).unwrap();
        let written = toml::to_string(&m).unwrap();
        assert!(
            written.contains(r#"mcp = ["mcp.sentry"]"#),
            "a rewrite does not wrap it in a table: {written}"
        );
    }

    #[test]
    fn a_connection_an_agent_names_is_declared() {
        let m = connected(r#"["mcp.sentyr"]"#).unwrap();
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("names no connection"), "{err}");
    }

    #[test]
    fn a_header_belongs_to_a_token_connection() {
        let with = |auth: &str| {
            manifest(&format!(
                "[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\n\
                 auth = \"{auth}\"\nheader = \"sentry-bearer\"\n"
            ))
            .validate()
        };
        with("token").unwrap();
        for auth in ["oauth", "none"] {
            let err = with(auth).unwrap_err().to_string();
            assert!(err.contains("needs `auth = \"token\"`"), "{err}");
        }
    }

    #[test]
    fn a_personal_connection_an_agents_own_app_reaches_is_reported() {
        let m = manifest(
            r#"
            [llm.claude]
            type = "anthropic"

            [agent.assistant]
            llm = "claude"
            model = "m"
            mcp = ["mcp.gmail", "mcp.sentry"]

            [agent.assistant.slack]

            [mcp.gmail]
            url = "https://mcp.example.test/mcp"
            credential = "user"

            [mcp.sentry]
            url = "https://mcp.sentry.dev/mcp"
            "#,
        );
        m.validate().unwrap();
        let notices = m.scope_notices();
        assert_eq!(notices.len(), 1, "one bad route, one notice: {notices:?}");
        assert!(notices[0].contains("[agent.assistant]"), "{}", notices[0]);
        assert!(notices[0].contains("[mcp.gmail]"), "{}", notices[0]);
        assert!(
            !notices[0].contains("sentry"),
            "a shared connection works anywhere: {}",
            notices[0]
        );

        let mut no_app = m.clone();
        no_app.agent.get_mut("assistant").unwrap().slack = None;
        assert!(no_app.scope_notices().is_empty());
    }

    #[test]
    fn an_agent_declares_its_own_app_by_the_blocks_presence() {
        let m = manifest(
            r#"
            [llm.claude]
            type = "anthropic"

            [agent.support]
            llm = "claude"
            model = "m"

            [agent.support.slack]

            [agent.triage]
            llm = "claude"
            model = "m"
            "#,
        );
        m.validate().unwrap();
        let apps = m.slack_apps();
        assert_eq!(apps.len(), 1, "only the agent that declared one");
        assert_eq!(apps[0].0, "support");
        assert_eq!(apps[0].1.name("support"), "support");
    }

    #[test]
    fn an_audience_round_trips_through_its_name() {
        for answers in SlackAudience::ALL {
            assert_eq!(SlackAudience::parse(answers.as_str()), Some(answers));
        }
        assert_eq!(SlackAudience::parse("channel"), None);
        assert_eq!(SlackAudience::parse(""), None);
    }

    #[test]
    fn an_agent_can_say_it_only_answers_in_one_place() {
        let m = manifest(
            r#"
            [llm.claude]
            type = "anthropic"

            [agent.assistant]
            llm = "claude"
            model = "m"
            mcp = ["mcp.gmail"]

            [agent.assistant.slack]
            answers = "dm"

            [mcp.gmail]
            url = "https://mcp.example.test/mcp"
            credential = "user"
            "#,
        );
        m.validate().unwrap();
        let answers = m.slack_apps()[0].1.answers;
        assert!(answers.dms());
        assert!(!answers.channels());
        assert!(
            m.scope_notices().is_empty(),
            "a personal credential in a DM is the case it is for"
        );
    }

    #[test]
    fn a_slack_app_name_slack_would_refuse_is_refused_here() {
        let long = manifest(&format!(
            r#"
            [llm.claude]
            type = "anthropic"

            [agent.support]
            llm = "claude"
            model = "m"

            [agent.support.slack]
            name = "{}"
            "#,
            "n".repeat(SLACK_APP_NAME_MAX + 1)
        ));
        let err = long.validate().unwrap_err().to_string();
        assert!(err.contains("characters Slack allows"), "{err}");
    }

    #[test]
    fn a_path_into_an_unresolved_plugin_parses() {
        let m = manifest(
            r#"
            [llm.claude]
            type = "anthropic"

            [plugin.reggu]
            path = "./plugin"

            [agent.support]
            llm = "claude"
            model = "m"
            mcp = ["plugin.reggu.mcp.code"]
            "#,
        );
        m.validate()
            .expect("no bundle yet, so the server cannot be denied");

        let mut resolved = m.clone();
        resolved.plugin.get_mut("reggu").unwrap().bundle = Some(crate::plugins::PluginBundle {
            name: "reggu".into(),
            servers: [("admin".to_string(), server("https://reggu.test/mcp"))].into(),
            ..Default::default()
        });
        let err = resolved.validate().unwrap_err().to_string();
        assert!(
            err.contains("names no connection `plugin.reggu.mcp.code`"),
            "{err}"
        );
    }

    fn plugin_manifest(agent_plugins: &str) -> Manifest {
        manifest(&format!(
            r#"
            [llm.claude]
            type = "anthropic"

            [plugin.pdf]
            path = "./plugins/pdf-tools"

            [agent.support]
            llm = "claude"
            model = "m"
            plugins = {agent_plugins}
            "#
        ))
    }

    #[test]
    fn a_plugin_is_declared_and_referenced_like_a_connection() {
        let m = plugin_manifest(r#"["pdf", { id = "pdf", approve = "always" }]"#);
        m.validate().unwrap();
        let config = m.agent["support"].to_agent_config(&m).unwrap();
        assert_eq!(config.plugins.len(), 2);
        assert_eq!(config.plugins[0].approve, Approve::default());
        assert_eq!(config.plugins[1].approve, Approve::Always);
    }

    #[test]
    fn an_agent_cannot_name_an_undeclared_plugin() {
        let m = plugin_manifest(r#"["typo"]"#);
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("names no plugin `typo`"), "{err}");
    }

    #[test]
    fn a_misspelled_plugin_entry_key_is_a_parse_error() {
        let err = toml::from_str::<Manifest>(
            r#"
            [agent.support]
            plugins = [{ id = "pdf", enable = true }]
            "#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("enable"), "{err}");
    }

    #[test]
    fn a_resolved_bundle_stamps_skills_and_servers_onto_the_wire_config() {
        let mut m = plugin_manifest(r#"["pdf"]"#);
        m.plugin.get_mut("pdf").unwrap().bundle = Some(crate::plugins::PluginBundle {
            name: "pdf-tools".into(),
            description: "PDF work.".into(),
            skills: vec![crate::plugins::Skill {
                name: "form-filling".into(),
                description: "Fill forms.".into(),
                body: "…".into(),
                ..Default::default()
            }],
            servers: [(
                "renderer".to_string(),
                BundleServer {
                    url: "https://pdf.example.com/mcp".into(),
                },
            )]
            .into(),
            version: None,
        });
        m.validate().unwrap();

        let config = m.agent["support"].to_agent_config(&m).unwrap();
        let p = &config.plugins[0];
        assert_eq!(p.description, "PDF work.");
        assert_eq!(p.skills[0].name, "form-filling");
        assert_eq!(p.servers, [path("plugin.pdf.mcp.renderer")]);
        assert!(
            m.connections()
                .contains_key(&path("plugin.pdf.mcp.renderer")),
            "the plugin's server joins the registry"
        );

        let wire = m.for_wire();
        assert!(wire.plugin["pdf"].path.is_none());
        assert!(
            wire.plugin["pdf"].bundle.is_none(),
            "the content travels as a plugin, not inside the config"
        );
        assert!(
            !serde_json::to_string(&wire)
                .unwrap()
                .contains("form-filling"),
            "and nothing of it is left behind"
        );
    }

    #[test]
    fn two_plugins_cannot_derive_one_connection_id() {
        let mut m = plugin_manifest(r#"["pdf"]"#);
        m.plugin.get_mut("pdf").unwrap().bundle = Some(crate::plugins::PluginBundle {
            name: "pdf_tools".into(),
            servers: [(
                "tools_render".to_string(),
                server("https://a.example.com/mcp"),
            )]
            .into(),
            ..Default::default()
        });
        m.plugin.insert(
            "pdf_tools".to_string(),
            PluginSpec {
                bundle: Some(crate::plugins::PluginBundle {
                    name: "other".into(),
                    servers: [("render".to_string(), server("https://b.example.com/mcp"))].into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        let e = m
            .validate()
            .expect_err("one id cannot answer for two servers");
        assert!(e.to_string().contains("pdf_tools_render"), "{e}");
    }

    #[test]
    fn a_wire_plugin_declares_itself_with_its_hash_alone() {
        let mut m = plugin_manifest(r#"["pdf"]"#);
        let spec = m.plugin.get_mut("pdf").unwrap();
        spec.path = None;
        spec.hash = Some("d0d0".into());
        m.validate()
            .expect("the deployment reads the plugin the hash names");

        m.plugin.get_mut("pdf").unwrap().hash = None;
        assert!(m.validate().is_err(), "and nothing at all is nothing");
    }

    #[test]
    fn a_plugin_auth_override_reaches_the_derived_connection() {
        let mut m = plugin_manifest(r#"["pdf"]"#);
        let spec = m.plugin.get_mut("pdf").unwrap();
        spec.bundle = Some(crate::plugins::PluginBundle {
            name: "pdf-tools".into(),
            servers: [(
                "renderer".to_string(),
                BundleServer {
                    url: "https://pdf.example.com/mcp".into(),
                },
            )]
            .into(),
            ..Default::default()
        });
        spec.mcp = [(
            "renderer".to_string(),
            PluginServerSpec {
                auth: Some(AuthKind::None),
                ..Default::default()
            },
        )]
        .into();
        m.validate().unwrap();
        assert_eq!(
            m.connections()[&path("plugin.pdf.mcp.renderer")].decl.auth,
            Some(AuthKind::None),
            "the override is how a credential-less server clears the authorize notice"
        );

        m.plugin.get_mut("pdf").unwrap().mcp =
            [("typo".to_string(), PluginServerSpec::default())].into();
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("names no server `typo`"), "{err}");
    }

    #[test]
    fn a_deployment_can_point_a_plugins_server_somewhere_else() {
        let mut m = plugin_manifest(r#"["pdf"]"#);
        let spec = m.plugin.get_mut("pdf").unwrap();
        spec.bundle = Some(crate::plugins::PluginBundle {
            name: "pdf-tools".into(),
            servers: [(
                "renderer".to_string(),
                server("https://pdf.example.com/mcp"),
            )]
            .into(),
            ..Default::default()
        });
        spec.mcp = [(
            "renderer".to_string(),
            PluginServerSpec {
                url: Some("https://staging.example.com/mcp".into()),
                auth: Some(AuthKind::Token),
                ..Default::default()
            },
        )]
        .into();
        m.validate().unwrap();

        let resolved = &m.connections()[&path("plugin.pdf.mcp.renderer")];
        assert_eq!(resolved.decl.url, "https://staging.example.com/mcp");
        assert_eq!(resolved.decl.auth, Some(AuthKind::Token));
        assert_eq!(
            resolved.path,
            ConnectionPath::PluginServer {
                plugin: "pdf".into(),
                server: "renderer".into()
            }
        );

        m.plugin.get_mut("pdf").unwrap().mcp.clear();
        assert_eq!(
            m.connections()[&path("plugin.pdf.mcp.renderer")].decl.url,
            "https://pdf.example.com/mcp"
        );
    }

    #[test]
    fn a_plugin_server_cannot_shadow_a_declared_connection() {
        let mut m = plugin_manifest(r#"["pdf"]"#);
        m.mcp.insert(
            "pdf_renderer".to_string(),
            decl("https://other.example.com/mcp"),
        );
        m.plugin.get_mut("pdf").unwrap().bundle = Some(crate::plugins::PluginBundle {
            name: "pdf-tools".into(),
            servers: [(
                "renderer".to_string(),
                BundleServer {
                    url: "https://pdf.example.com/mcp".into(),
                },
            )]
            .into(),
            ..Default::default()
        });
        let err = m.validate().unwrap_err().to_string();
        assert!(
            err.contains("`mcp.pdf_renderer` and `plugin.pdf.mcp.renderer` both prefix"),
            "{err}"
        );
    }

    #[test]
    fn two_ids_that_flatten_to_one_prefix_are_refused() {
        let mut m = manifest(
            r#"
            [llm.claude]
            type = "anthropic"

            [agent.support]
            llm = "claude"
            model = "m"
            "#,
        );
        m.mcp
            .insert("a-b".to_string(), decl("https://a.example.com/mcp"));
        m.mcp
            .insert("a_b".to_string(), decl("https://b.example.com/mcp"));
        let err = m.validate().unwrap_err().to_string();
        assert!(
            err.contains("`mcp.a-b` and `mcp.a_b` both prefix their tools with `a_b`"),
            "{err}"
        );
    }

    #[test]
    fn a_plugins_servers_travel_as_a_table_under_the_plugin() {
        let m: Manifest = toml::from_str(
            r#"
            [mcp.sentry]
            url = "https://mcp.sentry.dev/mcp"

            [plugin.reggu]
            path = "./plugin"

            [plugin.reggu.mcp.code]
            auth = "token"
            url = "https://sprite.example.com/code-search/mcp"
            "#,
        )
        .unwrap();

        let wire = serde_json::to_value(m.for_wire()).unwrap();
        assert_eq!(
            wire["plugin"]["reggu"]["mcp"]["code"],
            serde_json::json!({
                "url": "https://sprite.example.com/code-search/mcp",
                "auth": "token",
            })
        );
        assert!(
            !serde_json::to_string(&wire).unwrap().contains("\"path\""),
            "the path is the nesting, not a field: {wire}"
        );
    }

    #[test]
    fn a_path_names_the_connection_it_was_declared_at() {
        let mut m = plugin_manifest(r#"["pdf"]"#);
        m.plugin.get_mut("pdf").unwrap().bundle = Some(crate::plugins::PluginBundle {
            name: "pdf-tools".into(),
            servers: [(
                "renderer".to_string(),
                server("https://pdf.example.com/mcp"),
            )]
            .into(),
            ..Default::default()
        });
        m.mcp
            .insert("sentry".to_string(), decl("https://mcp.sentry.dev/mcp"));

        assert_eq!(
            m.connection_at(&path("plugin.pdf.mcp.renderer"))
                .unwrap()
                .path,
            path("plugin.pdf.mcp.renderer")
        );
        assert_eq!(
            m.connection_at(&path("mcp.sentry")).unwrap().path,
            path("mcp.sentry")
        );
        assert!(ConnectionPath::parse("pdf_renderer").is_none());
        assert_eq!(
            m.connection_paths()
                .iter()
                .map(ConnectionPath::to_string)
                .collect::<Vec<_>>(),
            ["mcp.sentry", "plugin.pdf.mcp.renderer"]
        );
    }

    #[test]
    fn an_agent_can_take_one_of_a_plugins_servers() {
        let mut m = manifest(
            r#"
            [llm.claude]
            type = "anthropic"

            [plugin.reggu]
            path = "./plugin"

            [agent.searcher]
            llm = "claude"
            model = "m"
            mcp = [{ id = "plugin.reggu.mcp.code", tools = { read_only = true } }]
            "#,
        );
        m.plugin.get_mut("reggu").unwrap().bundle = Some(crate::plugins::PluginBundle {
            name: "reggu".into(),
            servers: [
                ("admin".to_string(), server("https://admin.example.com/mcp")),
                ("code".to_string(), server("https://code.example.com/mcp")),
            ]
            .into(),
            ..Default::default()
        });
        m.validate().unwrap();

        let config = m.agent["searcher"].to_agent_config(&m).unwrap();
        assert!(config.plugins.is_empty(), "no bundle, just the one server");
        assert_eq!(config.mcp.len(), 1);
        assert_eq!(
            config.mcp[0].path,
            path("plugin.reggu.mcp.code"),
            "the file names the path and the wire carries it"
        );
        assert_eq!(config.mcp[0].tools.as_ref().unwrap().read_only, Some(true));
    }

    #[test]
    fn a_server_granted_twice_is_named_once() {
        let mut m = manifest(
            r#"
            [llm.claude]
            type = "anthropic"

            [plugin.reggu]
            path = "./plugin"

            [agent.support]
            llm = "claude"
            model = "m"
            plugins = ["reggu"]
            mcp = [{ id = "plugin.reggu.mcp.code", approve = "never" }]
            "#,
        );
        m.plugin.get_mut("reggu").unwrap().bundle = Some(crate::plugins::PluginBundle {
            name: "reggu".into(),
            servers: [("code".to_string(), server("https://code.example.com/mcp"))].into(),
            ..Default::default()
        });
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("already grants"), "{err}");
    }

    #[test]
    fn a_bare_id_is_told_which_path_it_meant() {
        let err = connected(r#"["sentry"]"#).unwrap().validate().unwrap_err();
        assert!(err.to_string().contains("Write `mcp.sentry`"), "{err}");
    }

    #[test]
    fn a_plugin_path_reads_against_the_file() {
        let base = std::path::Path::new("/proj");
        let dir = |p| plugin_dir(base, p).unwrap();
        assert_eq!(dir("./pdf"), std::path::Path::new("/proj/./pdf"));
        assert_eq!(dir("/opt/pdf"), std::path::Path::new("/opt/pdf"));
    }

    #[test]
    fn a_plugin_path_expands_a_home_directory_and_a_variable() {
        let home = dirs::home_dir().expect("a home directory");
        let base = std::path::Path::new("/proj");
        let dir = |p| plugin_dir(base, p).unwrap();
        assert_eq!(dir("~/plugins/pdf"), home.join("plugins/pdf"));
        assert_eq!(dir("~"), home);
        assert_eq!(dir("$HOME/plugins/pdf"), home.join("plugins/pdf"));
        let err = plugin_dir(base, "$SUBS_NO_SUCH_VAR/pdf").unwrap_err();
        assert!(err.to_string().contains("path ="), "{err}");
    }
}
