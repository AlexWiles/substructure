//! What a project *is*, as `substructure.toml` declares it — the half of the
//! file a deployment holds rather than the half that describes running an
//! engine here.
//!
//! One type, both ends: `subs apply` sends this, the deployment stores it, and
//! both run the same `check_*` functions, so the server enforces exactly what
//! the file already promised. `deny_unknown_fields` throughout makes version
//! skew a loud error rather than a silently dropped section.
//!
//! Nothing here binds a credential. `api_key_env` and `signing_secret_env` name
//! variables on the machine an engine runs on, which is why [`Manifest`] strips
//! them before the document crosses the wire.

use std::collections::BTreeMap;

use anyhow::{bail, Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::connectors::registry::{AuthKind, ConnectionSpec};
use crate::plugins::PluginBundle;
use crate::protocol::ReasoningEffort;
use crate::protocol::{
    AgentConfig, AgentPlugin, AgentTool, Approve, AuthFailure, ConnectorProtocol, DeferTools,
    Handler, LlmFormat, McpServer, McpTools, RetryConfig, SubAgent,
};
use crate::runtime::llm::{LlmBlock, LlmBlocks};
use crate::runtime::worker::{AgentEntry, WorkerEndpoint};

pub use crate::cli::env::ProviderKind;

/// The declaration a deployment holds: the agents, the blocks they name, the
/// connections they may reach, and where Slack routes.
///
/// Replace, not merge — the document is the whole declaration, so an agent,
/// block, or channel absent from it is one that was removed.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Manifest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub llm: BTreeMap<String, ProviderSpec>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub agent: BTreeMap<String, AgentSection>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, ConnectionSpec>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub plugin: BTreeMap<String, PluginSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slack: Option<SlackConfig>,
}

impl Manifest {
    /// Every check the file's own parse runs, in the order that reports the
    /// most specific cause: a block before the agent that names it, an agent
    /// before the Slack table that routes to it.
    pub fn validate(&self) -> Result<()> {
        for (id, spec) in &self.mcp {
            check_id(id).map_err(|e| anyhow::anyhow!("[mcp.{id}]: {e}"))?;
            check_url(&spec.url).map_err(|e| anyhow::anyhow!("[mcp.{id}]: {e}"))?;
            check_connection(spec).map_err(|e| anyhow::anyhow!("[mcp.{id}]: {e}"))?;
        }
        for (id, spec) in &self.plugin {
            check_plugin(id, spec, self).map_err(|e| anyhow::anyhow!("[plugin.{id}]: {e}"))?;
        }
        for (id, spec) in &self.llm {
            check_llm(id, spec).map_err(|e| anyhow::anyhow!("[llm.{id}]: {e}"))?;
        }
        for (id, section) in &self.agent {
            check_agent(id, section, self).map_err(|e| anyhow::anyhow!("[agent.{id}]: {e}"))?;
        }
        if let Some(slack) = &self.slack {
            check_slack(slack, self)?;
        }
        Ok(())
    }

    /// The variables this document names, which a deployment cannot read. The
    /// wire copy carries none, so a server that receives one is talking to a
    /// CLI that did not strip them.
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
        llm.chain(agent).collect()
    }

    /// This document with the env-bound fields dropped — the copy that crosses
    /// the wire. Named, never sent: a deployment holds its own credentials.
    pub fn for_wire(&self) -> Self {
        let mut wire = self.clone();
        for spec in wire.llm.values_mut() {
            spec.api_key_env = None;
        }
        for section in wire.agent.values_mut() {
            section.signing_secret_env = None;
        }
        // A deployment gets the resolved bundle; the path names a directory on
        // this machine.
        for spec in wire.plugin.values_mut() {
            spec.path = None;
        }
        wire
    }

    /// Every connection this project declares, keyed by the id an agent names.
    ///
    /// The one place the per-protocol sections become one registry, and so the
    /// one place a second protocol is added: `[a2a]` folds in here with its own
    /// `ConnectorProtocol`, and ids must stay unique across sections because an
    /// agent references a bare id and tool names are prefixed from it.
    pub fn connections(&self) -> BTreeMap<String, ConnectionSpec> {
        let declared = self.mcp.iter().map(|(id, spec)| {
            let spec = ConnectionSpec {
                protocol: ConnectorProtocol::Mcp,
                ..spec.clone()
            };
            (id.clone(), spec)
        });
        // A plugin's servers are ordinary MCP connections under derived ids;
        // only the pairing with an agent knows they came from a plugin.
        let from_plugins = self.plugin.iter().flat_map(|(pid, spec)| {
            spec.bundle.iter().flat_map(move |b| {
                b.servers.iter().map(move |(name, server)| {
                    let resolved = ConnectionSpec {
                        protocol: ConnectorProtocol::Mcp,
                        auth: spec.auth.get(name).copied().or(server.auth),
                        ..server.clone()
                    };
                    (crate::plugins::server_id(pid, name), resolved)
                })
            })
        });
        declared.chain(from_plugins).collect()
    }

    /// Load each plugin's bundle from its `path`, for the halves of the CLI
    /// that need the data: the local engine at startup, `subs apply` before
    /// the wire copy. Returns the loaders' notices. Entries already carrying
    /// a bundle are left alone.
    pub fn resolve_plugins(&mut self, base: &std::path::Path) -> Result<Vec<String>> {
        let mut notices = Vec::new();
        for (id, spec) in &mut self.plugin {
            if spec.bundle.is_some() {
                continue;
            }
            let Some(path) = &spec.path else {
                continue;
            };
            let loaded = crate::plugins::load_dir(&base.join(path))
                .map_err(|e| anyhow::anyhow!("[plugin.{id}]: {e}"))?;
            notices.extend(
                loaded
                    .notices
                    .into_iter()
                    .map(|n| format!("[plugin.{id}]: {n}")),
            );
            spec.hash = Some(loaded.bundle.hash());
            spec.bundle = Some(loaded.bundle);
        }
        // What the bundles brought is part of the declaration.
        self.validate()?;
        Ok(notices)
    }

    /// The declared blocks as the engine reads them: venue and wire shape,
    /// never a credential.
    pub fn llm_blocks(&self) -> LlmBlocks {
        self.llm
            .iter()
            .map(|(name, spec)| (name.clone(), spec.block()))
            .collect()
    }

    /// Every agent this project declares, keyed by the id a client routes on.
    pub fn agents(&self) -> BTreeMap<String, AgentEntry> {
        self.agent
            .iter()
            .map(|(id, section)| (id.clone(), section.to_entry(self)))
            .collect()
    }

    pub fn agent_ids(&self) -> Vec<String> {
        self.agent.keys().cloned().collect()
    }

    /// The agent answering DMs, for the engine that serves them.
    pub fn slack_dm_agent(&self) -> Option<String> {
        self.slack.as_ref()?.dm.clone()
    }
}

/// One `[llm.<id>]` block: what runs a call on it, and — where the engine runs
/// it — how a key is bound.
///
/// The key is named, never written, for the same reason a connection's is: a
/// committed file must not be able to hold a secret. A `worker` block names no
/// variable at all — the call never leaves the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSpec {
    #[serde(rename = "type")]
    pub kind: ProviderKind,
    /// Variable holding the key. Absent ⇒ the type's own default
    /// (`ANTHROPIC_API_KEY` and so on). Local only: stripped by
    /// [`Manifest::for_wire`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Wire shape of the `llm.execute` a worker answers. Only ever valid on
    /// `type = "worker"`; absent ⇒ the engine's neutral format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<LlmFormat>,
    /// How long the vendor holds a cached prompt prefix. Each vendor spells
    /// its own lives, so [`check_llm`] reads this against the block's `type`.
    /// Absent ⇒ the vendor default, which suits a session that turns often.
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

    /// The variable this block's key is read from, for the types that need one.
    pub fn api_key_env(&self) -> Option<String> {
        self.api_key_env
            .clone()
            .or_else(|| self.kind.default_api_key_env().map(str::to_string))
    }
}

/// One `[agent.<id>]` section: what the agent is, and who decides for it.
///
/// `worker` is the whole routing switch — set, and decisions POST there; unset,
/// and the engine decides by accepting its own proposals.
///
/// The config half mirrors the wire [`AgentConfig`], but every field is
/// optional here, because a section has two jobs and only the first is
/// mandatory: it declares that the agent *exists* and where its decisions go,
/// and it may also *seed* the config. An agent that delegates everything to its
/// worker needs only the first, so `[agent.<id>]` with nothing but a `worker`
/// URL is a complete declaration — the worker authors the config on
/// `session.start`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentSection {
    /// What this agent is for, read by whoever delegates to it: naming it in
    /// another section's `sub_agents` is what puts this text in front of a
    /// model. Declared on the agent rather than on each edge pointing at it,
    /// so two parents delegating to one specialist cannot describe it
    /// differently.
    ///
    /// Not part of the wire config — it is expanded into the *parent's*
    /// `sub_agents` at load, which is why an agent that declares nothing but
    /// this and a `worker` is still a useful declaration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// How hard the model thinks. Unset leaves the provider's own default,
    /// which on the newest models is to think.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AgentTool>,
    /// The agents this one may delegate to, named by id. Each one's
    /// description comes from the section it names.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sub_agents: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mcp: Vec<McpRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<PluginRef>,
    /// The default for every tool of this agent. A tool or a connection
    /// overrides it with its own `defer`. `true` takes the defaults; a table
    /// sets them.
    #[serde(
        default,
        deserialize_with = "crate::protocol::de_defer_tools",
        skip_serializing_if = "Option::is_none"
    )]
    pub defer_tools: Option<DeferTools>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
    /// Environment variable holding the secret an engine here signs this
    /// agent's decision requests with. Named, never written. Unset means the
    /// requests go unsigned, rather than signed with a secret nobody can check.
    ///
    /// Local only: a deployment mints and holds its own secret per agent, so
    /// [`Manifest::for_wire`] strips this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signing_secret_env: Option<String>,
}

impl AgentSection {
    /// Whether the section says anything about the agent beyond its hosting.
    /// A section that does not is a declaration of existence alone.
    fn declares_config(&self) -> bool {
        self.llm.is_some()
            || self.model.is_some()
            || self.system.is_some()
            || self.retry.is_some()
            || !self.tools.is_empty()
            || !self.sub_agents.is_empty()
            || !self.mcp.is_empty()
            || !self.plugins.is_empty()
    }

    /// The wire config this section seeds, with the hosting stripped. `None`
    /// when the section seeds nothing — validation has already made sure a
    /// worker is there to author one.
    pub fn to_agent_config(&self, manifest: &Manifest) -> Option<AgentConfig> {
        Some(AgentConfig {
            llm: self.llm.clone(),
            model: self.model.clone()?,
            system: self.system.clone(),
            effort: self.effort,
            retry: self.retry.clone().map(Box::new),
            tools: self.tools.clone(),
            sub_agents: self.to_sub_agents(manifest),
            mcp: self.mcp.iter().map(McpRef::to_server).collect(),
            defer_tools: self.defer_tools,
            announce_mcp: Default::default(),
            plugins: self.plugins.iter().map(|p| p.to_wire(manifest)).collect(),
        })
    }

    /// The named agents as the wire carries them, each description read from
    /// the section it names. Resolved here rather than at delegation time
    /// because a parent builds its tool list before any child session exists,
    /// so there is nobody else to ask.
    ///
    /// An unresolved name yields an empty description rather than being
    /// dropped: validation rejects one, and a tool the model can still call is
    /// a better failure than a silently missing teammate.
    fn to_sub_agents(&self, manifest: &Manifest) -> Vec<SubAgent> {
        self.sub_agents
            .iter()
            .map(|id| SubAgent {
                id: id.clone(),
                description: manifest
                    .agent
                    .get(id)
                    .and_then(|s| s.description.clone())
                    .unwrap_or_default(),
            })
            .collect()
    }

    /// This agent as the engine routes it: the config it seeds, and the endpoint
    /// its decisions go to with whatever secret the named variable held.
    pub fn to_entry(&self, manifest: &Manifest) -> AgentEntry {
        AgentEntry {
            config: self.to_agent_config(manifest),
            worker: self.worker.clone().map(|url| WorkerEndpoint {
                url,
                signing_secret: self
                    .signing_secret_env
                    .as_deref()
                    .and_then(crate::cli::env_value),
            }),
        }
    }
}

/// One `[plugin.<id>]`: an agent-plugins directory resolved to data.
///
/// `path` is where the CLI reads it; `bundle` is what a deployment holds.
/// A file writes only `path` — `subs apply` stamps the bundle onto the wire
/// copy, and a local engine loads it at startup, so the two halves never
/// coexist in a committed file.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PluginSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle: Option<PluginBundle>,
    /// Per-server auth override, `<server> = "oauth" | "token" | "none"` —
    /// what `mcp.json` has no field for. `none` is the one that matters: it
    /// clears the standing "authorize this" notice a credential-less server
    /// would otherwise keep raising.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub auth: BTreeMap<String, AuthKind>,
    /// The bundle's content hash, stamped when the bundle is resolved. What a
    /// deployment stores and diffs on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

/// What of a plugin can be checked from what is present: the id always, the
/// bundle's servers once a bundle is there.
fn check_plugin(id: &str, spec: &PluginSpec, manifest: &Manifest) -> Result<()> {
    check_id(id)?;
    if spec.path.is_none() && spec.bundle.is_none() {
        bail!("declares nothing. Set `path` to the plugin's directory.");
    }
    let Some(bundle) = &spec.bundle else {
        return Ok(());
    };
    for name in spec.auth.keys() {
        if !bundle.servers.contains_key(name) {
            bail!(
                "`auth` names no server `{name}`. The plugin's servers: {}",
                declared(bundle.servers.keys())
            );
        }
    }
    for (name, server) in &bundle.servers {
        check_id(name).map_err(|e| anyhow::anyhow!("server `{name}`: {e}"))?;
        check_url(&server.url).map_err(|e| anyhow::anyhow!("server `{name}`: {e}"))?;
        let derived = crate::plugins::server_id(id, name);
        if manifest.mcp.contains_key(&derived) {
            bail!(
                "server `{name}` resolves to connection id `{derived}`, which `[mcp.{derived}]` \
                 already declares"
            );
        }
    }
    Ok(())
}

/// One connection an agent draws tools from: a bare id for everything the
/// connection grants, or a table where the agent narrows it.
///
/// Two spellings for two different things rather than one thing twice — the
/// filter belongs to the *pair*, not to the connection, so one `[mcp.<id>]`
/// with one credential serves an agent that gets `read_only` and one that gets
/// everything. Most agents narrow nothing, and `{ id = "sentry" }` is a table
/// wrapped around a string for them.
#[derive(Debug, Clone, PartialEq)]
pub enum McpRef {
    /// `"sentry"` — every tool the connection grants.
    All(String),
    /// `{ id = "sentry", tools = { … } }` — narrowed for this agent.
    Filtered(McpEntry),
}

/// The table form of an [`McpRef`]. `deny_unknown_fields` because the wire
/// [`McpServer`] has none: a misspelled `tools` would otherwise deserialize to
/// no filter at all, handing the model every tool the connection grants when
/// the file asked for a few.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpToolsEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_failure: Option<AuthFailure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approve: Option<Approve>,
}

/// The file's spelling of [`McpTools`], mirrored for the one reason
/// [`McpEntry`] is: `deny_unknown_fields`. A misspelled `defer` would
/// otherwise read as no setting. [`McpTools`] documents the fields.
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

    /// The wire form. The two spellings differ only in what rides along, so
    /// this is where they stop differing.
    fn to_server(&self) -> McpServer {
        match self {
            Self::All(id) => McpServer {
                id: id.clone(),
                tools: None,
                auth_failure: AuthFailure::default(),
                approve: Approve::default(),
            },
            Self::Filtered(entry) => McpServer {
                id: entry.id.clone(),
                tools: entry.tools.as_ref().map(McpToolsEntry::to_wire),
                auth_failure: entry.auth_failure.unwrap_or_default(),
                approve: entry.approve.unwrap_or_default(),
            },
        }
    }
}

/// One plugin an agent uses: a bare id, or a table with the knobs.
///
/// Same two spellings as [`McpRef`], for the same reason: the policy belongs
/// to the pair, so one `[plugin.<id>]` serves two agents that read its servers
/// differently.
#[derive(Debug, Clone, PartialEq)]
pub enum PluginRef {
    All(String),
    Configured(PluginEntry),
}

/// The table form of a [`PluginRef`]. `deny_unknown_fields` for the reason
/// [`McpEntry`] has it. `tools`, `auth_failure`, and `approve` apply to the
/// plugin's servers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpToolsEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_failure: Option<AuthFailure>,
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

    /// The wire form, stamped from the bundle the manifest holds: the model's
    /// catalog metadata and the derived server ids ride the config, so the
    /// engine never re-reads a bundle to know what an agent reaches.
    fn to_wire(&self, manifest: &Manifest) -> AgentPlugin {
        let id = self.id();
        let bundle = manifest.plugin.get(id).and_then(|s| s.bundle.as_ref());
        let (description, skills, servers) = match bundle {
            Some(b) => (
                b.description.clone(),
                b.skill_metas(),
                b.servers
                    .keys()
                    .map(|name| crate::plugins::server_id(id, name))
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
            auth_failure: entry.and_then(|e| e.auth_failure).unwrap_or_default(),
            approve: entry.and_then(|e| e.approve).unwrap_or_default(),
        }
    }
}

impl Serialize for PluginRef {
    /// Written back as it was written: a bare id stays bare.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::All(id) => s.serialize_str(id),
            Self::Configured(entry) => entry.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for PluginRef {
    /// Dispatched on the shape written, like [`McpRef`], so a misspelled field
    /// is named rather than answered with "matched no variant".
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
    /// Written back as it was written: a bare id stays bare, so a command that
    /// rewrites the file does not wrap every connection in a table.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::All(id) => s.serialize_str(id),
            Self::Filtered(entry) => entry.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for McpRef {
    /// Dispatched on the shape written, not by trying each variant: an
    /// `#[serde(untagged)]` here would answer a misspelled field with "matched
    /// no variant" instead of naming the field.
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

/// The `[slack]` section: what the bot needs that is not a secret.
///
/// The tokens are absent for the same reason they are absent from `[mcp]` — a
/// committed file must not be able to hold one — so `SLACK_APP_TOKEN` and
/// `SLACK_BOT_TOKEN` stay in the environment.
///
/// Three questions, asked separately, because they have different answers and
/// different blast radii: who takes DMs, who takes a channel nobody named, and
/// who takes each channel that is named. Every one defaults to silence, so a
/// bot answers only where it was told to.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SlackConfig {
    /// Agent for direct messages. Absent leaves DMs unanswered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dm: Option<String>,
    /// Agent for a mention in any channel that `channel` does not name. A
    /// channel only ever reaches the bot by mentioning it, which is what this
    /// is named after; a DM needs no mention, which is why it is a key of its
    /// own. Absent makes the channel table an allowlist — the bot can be
    /// invited anywhere and still answer only where it was named.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mentions: Option<String>,
    /// Where one channel differs, keyed by Slack channel id. An id is the
    /// stable identity; a name is remote state that a rename re-points, so
    /// only an id is accepted.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub channel: BTreeMap<String, SlackChannelConfig>,
}

impl SlackConfig {
    /// Whether this section configures a bot at all.
    pub fn is_configured(&self) -> bool {
        self.dm.is_some() || self.mentions.is_some() || !self.channel.is_empty()
    }
}

/// One `[slack.channel.<id>]` section: who answers there, or that nobody does.
///
/// A channel names an `agent` rather than restating a system prompt or a tool
/// list, because `[agent.<id>]` already is that bundle: pointing a channel at
/// a different agent gives it a different prompt, model, and tools at once,
/// and the tools are the agent's own rather than a request the model may
/// decline.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SlackChannelConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// The bot stays out of this channel.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub off: bool,
}

impl SlackChannelConfig {
    /// The agent that answers here, or `None` where the bot stays out.
    pub fn agent(&self) -> Option<&str> {
        match self.off {
            true => None,
            false => self.agent.as_deref(),
        }
    }
}

/// A block declares one venue, so the fields of the other are not a detail to
/// ignore — they are a misunderstanding of what the block is.
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

/// The lives one vendor spells for a cached prefix.
pub fn cache_ttls(kind: ProviderKind) -> &'static [&'static str] {
    match kind {
        ProviderKind::Anthropic | ProviderKind::Openrouter => &["5m", "1h"],
        ProviderKind::Openai => &["in_memory", "24h"],
        ProviderKind::Worker => &[],
    }
}

/// Every name an agent uses is declared in this same document, so a typo is
/// caught here rather than as a failing decision on the first turn.
pub fn check_agent(id: &str, section: &AgentSection, manifest: &Manifest) -> Result<()> {
    check_id(id)?;

    if let Some(url) = &section.worker {
        check_url(url)?;
    } else if section.signing_secret_env.is_some() {
        bail!(
            "`signing_secret_env` signs decision requests, and there is no `worker` to send any to"
        );
    }

    // Nothing but hosting: the worker authors the config. Legitimate, and the
    // only way to say "this agent is entirely my code" — but only a worker can
    // author one, so without one the engine would have nothing to propose.
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

    // Anything the document seeds has to be a config the engine can actually
    // propose from, so a half-declared one is an error rather than a proposal
    // that fails at the first model call.
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

    for server in &section.mcp {
        if !manifest.mcp.contains_key(server.id()) {
            bail!(
                "`mcp` names no connection `{}`. Declared: {}",
                server.id(),
                declared(manifest.mcp.keys())
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

    for sub in &section.sub_agents {
        check_sub_agent(sub, section, manifest)
            .map_err(|e| anyhow::anyhow!("`sub_agents`: {e}"))?;
    }

    // A worker-executed tool comes from worker code, so a document that
    // declares one would be naming a function nothing here can run.
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

/// One name in a `sub_agents` list. The description is the delegating model's
/// only account of what the teammate is for, so an absent one is checked here
/// — but only where nothing later could supply it.
fn check_sub_agent(sub: &str, section: &AgentSection, manifest: &Manifest) -> Result<()> {
    let Some(child) = manifest.agent.get(sub) else {
        bail!(
            "`{sub}` names no agent. Declared: {}",
            declared(manifest.agent.keys())
        );
    };
    // One namespace: a sub-agent reaches the model as a tool named by its id,
    // so a collision would leave one of the two unreachable.
    if section.tools.iter().any(|t| t.name == sub) {
        bail!("`{sub}` is also a tool name, and the model sees one namespace for both");
    }
    // A worker receives the expanded config as its `session.start` proposal and
    // may rewrite it, so it is a legitimate source of the description. An agent
    // the engine decides for has no such later chance — the document is the
    // whole account of it — so a missing description there is a delegation tool
    // the model cannot choose between.
    if child.description.is_none() && section.worker.is_none() {
        bail!(
            "`{sub}` has no description, and there is no `worker` to supply one. Set \
             `description` on [agent.{sub}]."
        );
    }
    Ok(())
}

/// Every agent the bot routes to is declared in this same document, so a typo
/// is caught here rather than as a bot that answers nowhere.
pub fn check_slack(slack: &SlackConfig, manifest: &Manifest) -> Result<()> {
    for (key, agent) in [("dm", &slack.dm), ("mentions", &slack.mentions)] {
        if let Some(agent) = agent {
            check_slack_agent(agent, manifest)
                .map_err(|e| anyhow::anyhow!("[slack]: `{key}`: {e}"))?;
        }
    }
    for (id, channel) in &slack.channel {
        check_channel(id, channel, manifest)
            .map_err(|e| anyhow::anyhow!("[slack.channel.{id}]: {e}"))?;
    }
    // Channels that are all `off`, no DM agent, and nothing for the rest: a bot
    // that connects, listens, and can answer nowhere.
    if slack.is_configured()
        && slack.dm.is_none()
        && slack.mentions.is_none()
        && !slack.channel.values().any(|c| !c.off)
    {
        bail!(
            "[slack]: nothing to answer with. Set `dm`, set `mentions`, or name an agent \
             in a `[slack.channel.<id>]`."
        );
    }
    Ok(())
}

pub fn check_slack_agent(agent: &str, manifest: &Manifest) -> Result<()> {
    if manifest.agent.contains_key(agent) {
        return Ok(());
    }
    bail!(
        "`agent = \"{agent}\"` names no agent. Declared: {}",
        declared(manifest.agent.keys())
    )
}

/// A channel says one of two things, and a section that says both or neither
/// is a setting with no meaning rather than one to resolve by precedence.
fn check_channel(id: &str, channel: &SlackChannelConfig, manifest: &Manifest) -> Result<()> {
    if id.is_empty() {
        bail!("the id is empty");
    }
    // The likeliest mistake, and one that would otherwise match no event and
    // report nothing: a rename re-points a name, so only an id can be pinned.
    if id.starts_with('#') || id.starts_with('@') {
        bail!(
            "`{id}` is a name, not a channel id. A rename re-points a name; use the id from the \
             channel's About tab (`C…`)."
        );
    }
    match (&channel.agent, channel.off) {
        (Some(_), true) => bail!(
            "`off` and `agent` contradict: a channel the bot stays out of has nobody to answer in it"
        ),
        (None, false) => bail!(
            "declares nothing. Name the `agent` that answers here, or set `off = true` to keep \
             the bot out."
        ),
        (Some(agent), false) => check_slack_agent(agent, manifest),
        (None, true) => Ok(()),
    }
}

pub fn declared<'a>(mut ids: impl Iterator<Item = &'a String>) -> String {
    let joined = ids.by_ref().cloned().collect::<Vec<_>>().join(", ");
    match joined.is_empty() {
        true => "none".to_string(),
        false => joined,
    }
}

/// An id becomes the prefix on every tool name the model sees, so it is held to
/// what a tool name may contain rather than to what TOML accepts as a key.
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

/// Rejected while reading the document rather than at the first fetch, where it
/// would surface as a metadata failure against a URL nobody meant to write.
/// `header` carries a static token, so it says nothing under a method that
/// binds its own. Reported rather than ignored: a file that names one means it
/// to be sent.
fn check_connection(spec: &ConnectionSpec) -> Result<()> {
    if spec.header.is_some() && spec.auth != Some(AuthKind::Token) {
        bail!("`header` carries a static token, so it needs `auth = \"token\"`");
    }
    Ok(())
}

/// Checked again where a credential is actually sent, but a URL the document
/// should not have held is better reported while reading it.
pub fn check_url(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("`{url}` is not a URL"))?;
    if parsed.scheme() == "https" || crate::connectors::oauth::is_loopback(url) {
        return Ok(());
    }
    bail!("`{url}` is not https: a credential would cross the network in the clear")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::DeferToolsStrategy;

    fn manifest(toml: &str) -> Manifest {
        toml::from_str(toml).unwrap()
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
            "#,
        );
        assert_eq!(
            m.local_bindings(),
            [
                "[llm.claude].api_key_env",
                "[agent.support].signing_secret_env"
            ]
        );

        let wire = m.for_wire();
        assert!(wire.local_bindings().is_empty());
        // Everything else survives: only the bindings are local.
        assert_eq!(wire.agent["support"].worker, m.agent["support"].worker);
        assert_eq!(wire.llm["claude"].kind, ProviderKind::Anthropic);
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

    /// A two-agent team: `assistant` delegates to `poet`, with each section's
    /// remaining lines supplied by the test.
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
    fn a_sub_agent_takes_its_description_from_the_agent_it_names() {
        let m = team(
            r#"sub_agents = ["poet"]"#,
            r#"description = "Writes a haiku.""#,
        );
        m.validate().unwrap();
        let agents = m.agents();
        let subs = &agents["assistant"]
            .config
            .as_ref()
            .expect("seeded")
            .sub_agents;
        assert_eq!(
            subs,
            &[SubAgent {
                id: "poet".to_string(),
                description: "Writes a haiku.".to_string(),
            }],
            "declared once, on the agent it describes"
        );
    }

    #[test]
    fn a_sub_agent_names_a_declared_agent() {
        let bad = team(r#"sub_agents = ["potet"]"#, r#"description = "d""#);
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("[agent.assistant]"), "{err}");
        assert!(err.contains("names no agent"), "{err}");
    }

    #[test]
    fn a_sub_agent_cannot_take_a_tool_s_name() {
        let bad = team(
            r#"sub_agents = ["poet"]
               tools = [{ name = "poet", description = "d", handler = "client" }]"#,
            r#"description = "d""#,
        );
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("one namespace"), "{err}");
    }

    #[test]
    fn an_engine_hosted_parent_needs_its_teammate_described() {
        let bad = team(r#"sub_agents = ["poet"]"#, "");
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("no description"), "{err}");

        // A worker receives the config and may write the description itself.
        team(
            r#"sub_agents = ["poet"]
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

    /// One agent drawing on one connection, its `mcp` line supplied by the test.
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
        for spelling in [r#"["sentry"]"#, r#"[{ id = "sentry" }]"#] {
            let m = connected(spelling).unwrap();
            m.validate().unwrap();
            let agents = m.agents();
            let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
            assert_eq!(mcp.len(), 1, "{spelling}");
            assert_eq!(mcp[0].id, "sentry", "{spelling}");
            assert!(mcp[0].tools.is_none(), "no filter ⇒ everything: {spelling}");
        }

        let m = connected(r#"[{ id = "sentry", tools = { read_only = true } }]"#).unwrap();
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
        let m = connected(r#"[{ id = "sentry", approve = "destructive" }]"#).unwrap();
        m.validate().unwrap();
        let agents = m.agents();
        let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
        assert_eq!(mcp[0].approve, Approve::Destructive);

        let m = connected(r#"["sentry"]"#).unwrap();
        let agents = m.agents();
        let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
        assert_eq!(mcp[0].approve, Approve::Never, "nothing asks by default");
    }

    #[test]
    fn an_unknown_approve_value_is_an_error() {
        let err = connected(r#"[{ id = "sentry", approve = "sometimes" }]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("sometimes"), "names the value: {err}");
    }

    /// The wire `McpServer` allows unknown fields, so a misspelled `tools` used
    /// to parse as *no* filter — the file asking for a few tools and the model
    /// getting all of them.
    #[test]
    fn a_misspelled_filter_is_an_error_rather_than_no_filter() {
        let err = connected(r#"[{ id = "sentry", tool = { read_only = true } }]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("tool"), "{err}");
        assert!(err.contains("unknown field"), "names the field: {err}");
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
        let m = connected(r#"[{ id = "sentry", tools = { defer = true } }]"#).unwrap();
        m.validate().unwrap();
        let agents = m.agents();
        let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
        assert_eq!(mcp[0].tools.as_ref().expect("a table").defer, Some(true));

        let m = connected(r#"[{ id = "sentry", tools = { read_only = true } }]"#).unwrap();
        let agents = m.agents();
        let mcp = &agents["support"].config.as_ref().expect("seeded").mcp;
        assert!(
            mcp[0].tools.as_ref().expect("a table").defer.is_none(),
            "unset ⇒ the agent decides"
        );
    }

    /// The same fault as a misspelled `tools`, one level down.
    #[test]
    fn a_misspelled_key_inside_the_tools_table_is_an_error() {
        let err = connected(r#"[{ id = "sentry", tools = { defr = true } }]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("defr"), "names the field: {err}");
        assert!(err.contains("unknown field"), "{err}");
    }

    #[test]
    fn a_defer_that_is_not_a_boolean_is_an_error() {
        let err = connected(r#"[{ id = "sentry", tools = { defer = "search" } }]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("boolean"), "says what it wanted: {err}");
    }

    #[test]
    fn a_bare_id_is_written_back_bare() {
        let m = connected(r#"["sentry"]"#).unwrap();
        let written = toml::to_string(&m).unwrap();
        assert!(
            written.contains(r#"mcp = ["sentry"]"#),
            "a rewrite does not wrap it in a table: {written}"
        );
    }

    #[test]
    fn a_connection_an_agent_names_is_declared() {
        let m = connected(r#"["sentyr"]"#).unwrap();
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("names no connection"), "{err}");
    }

    /// `header` only means something for a credential the file says to send
    /// itself, so it is refused rather than ignored under the other methods.
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
    fn slack_routes_only_to_declared_agents() {
        let bad = manifest(
            r#"
            [agent.support]
            llm = "claude"
            model = "m"

            [llm.claude]
            type = "anthropic"

            [slack]
            dm = "typo"
            "#,
        );
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("names no agent"), "{err}");
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
                files: Default::default(),
            }],
            servers: [(
                "renderer".to_string(),
                ConnectionSpec {
                    url: "https://pdf.example.com/mcp".into(),
                    protocol: ConnectorProtocol::Mcp,
                    auth: None,
                    header: None,
                    prefix_tools: true,
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
        assert_eq!(p.servers, ["pdf-renderer"]);
        assert!(
            m.connections().contains_key("pdf-renderer"),
            "the plugin's server joins the registry under the derived id"
        );

        // The wire copy carries the bundle, never the path.
        let wire = m.for_wire();
        assert!(wire.plugin["pdf"].path.is_none());
        assert!(wire.plugin["pdf"].bundle.is_some());
    }

    #[test]
    fn a_plugin_auth_override_reaches_the_derived_connection() {
        let mut m = plugin_manifest(r#"["pdf"]"#);
        let spec = m.plugin.get_mut("pdf").unwrap();
        spec.bundle = Some(crate::plugins::PluginBundle {
            name: "pdf-tools".into(),
            servers: [(
                "renderer".to_string(),
                ConnectionSpec {
                    url: "https://pdf.example.com/mcp".into(),
                    protocol: ConnectorProtocol::Mcp,
                    auth: None,
                    header: None,
                    prefix_tools: true,
                },
            )]
            .into(),
            ..Default::default()
        });
        spec.auth = [("renderer".to_string(), AuthKind::None)].into();
        m.validate().unwrap();
        assert_eq!(
            m.connections()["pdf-renderer"].auth,
            Some(AuthKind::None),
            "the override is how a credential-less server clears the authorize notice"
        );

        m.plugin.get_mut("pdf").unwrap().auth = [("typo".to_string(), AuthKind::None)].into();
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("names no server `typo`"), "{err}");
    }

    #[test]
    fn a_plugin_server_cannot_shadow_a_declared_connection() {
        let mut m = plugin_manifest(r#"["pdf"]"#);
        m.mcp.insert(
            "pdf-renderer".to_string(),
            ConnectionSpec {
                url: "https://other.example.com/mcp".into(),
                protocol: ConnectorProtocol::Mcp,
                auth: None,
                header: None,
                prefix_tools: true,
            },
        );
        m.plugin.get_mut("pdf").unwrap().bundle = Some(crate::plugins::PluginBundle {
            name: "pdf-tools".into(),
            servers: [(
                "renderer".to_string(),
                ConnectionSpec {
                    url: "https://pdf.example.com/mcp".into(),
                    protocol: ConnectorProtocol::Mcp,
                    auth: None,
                    header: None,
                    prefix_tools: true,
                },
            )]
            .into(),
            ..Default::default()
        });
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("already declares"), "{err}");
    }
}
