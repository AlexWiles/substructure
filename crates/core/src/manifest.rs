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

use crate::connectors::registry::ConnectionSpec;
use crate::protocol::{
    AgentConfig, AgentTool, ConnectorProtocol, Handler, LlmFormat, McpServer, RetryPolicy, SubAgent,
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
        wire
    }

    /// Every connection this project declares, keyed by the id an agent names.
    ///
    /// The one place the per-protocol sections become one registry, and so the
    /// one place a second protocol is added: `[a2a]` folds in here with its own
    /// `ConnectorProtocol`, and ids must stay unique across sections because an
    /// agent references a bare id and tool names are prefixed from it.
    pub fn connections(&self) -> BTreeMap<String, ConnectionSpec> {
        self.mcp
            .iter()
            .map(|(id, spec)| {
                let spec = ConnectionSpec {
                    protocol: ConnectorProtocol::Mcp,
                    ..spec.clone()
                };
                (id.clone(), spec)
            })
            .collect()
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
            .map(|(id, section)| (id.clone(), section.to_entry()))
            .collect()
    }

    pub fn agent_ids(&self) -> Vec<String> {
        self.agent.keys().cloned().collect()
    }

    pub fn slack_agent(&self) -> Option<String> {
        self.slack.as_ref()?.agent.clone()
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryPolicy>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AgentTool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sub_agents: Vec<SubAgent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mcp: Vec<McpServer>,
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
            || self.stream.is_some()
            || self.retry.is_some()
            || !self.tools.is_empty()
            || !self.sub_agents.is_empty()
            || !self.mcp.is_empty()
    }

    /// The wire config this section seeds, with the hosting stripped. `None`
    /// when the section seeds nothing — validation has already made sure a
    /// worker is there to author one.
    pub fn to_agent_config(&self) -> Option<AgentConfig> {
        Some(AgentConfig {
            llm: self.llm.clone(),
            model: self.model.clone()?,
            system: self.system.clone(),
            stream: self.stream.unwrap_or(false),
            retry: self.retry.clone(),
            tools: self.tools.clone(),
            sub_agents: self.sub_agents.clone(),
            mcp: self.mcp.clone(),
        })
    }

    /// This agent as the engine routes it: the config it seeds, and the endpoint
    /// its decisions go to with whatever secret the named variable held.
    pub fn to_entry(&self) -> AgentEntry {
        AgentEntry {
            config: self.to_agent_config(),
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

/// The `[slack]` section: what the Socket Mode bot needs that is not a secret.
///
/// The tokens are absent for the same reason they are absent from `[mcp]` — a
/// committed file must not be able to hold one — so `SLACK_APP_TOKEN` and
/// `SLACK_BOT_TOKEN` stay in the environment.
///
/// `agent` is the default, and `[slack.channel.<id>]` is where one channel
/// differs. An allowlist is the absence of a default rather than a second
/// setting: with no `agent` here, only the declared channels are served.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SlackConfig {
    /// Agent id the bot drives wherever the channel table says nothing.
    /// Absent, with no channel declared, leaves the bot off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// Where one channel differs, keyed by Slack channel id. An id is the
    /// stable identity; a name is remote state that a rename re-points, so
    /// only an id is accepted.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub channel: BTreeMap<String, SlackChannelConfig>,
}

impl SlackConfig {
    /// Whether this section configures a bot at all.
    pub fn is_configured(&self) -> bool {
        self.agent.is_some() || !self.channel.is_empty()
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
        }
        _ => {
            if spec.format.is_some() {
                bail!(
                    "`format` is the wire shape of an `llm.execute`, so it only applies to \
                     `type = \"worker\"`"
                );
            }
        }
    }
    Ok(())
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
        if !manifest.mcp.contains_key(&server.id) {
            bail!(
                "`mcp` names no connection `{}`. Declared: {}",
                server.id,
                declared(manifest.mcp.keys())
            );
        }
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

/// Every agent the bot routes to is declared in this same document, so a typo
/// is caught here rather than as a bot that answers nowhere.
pub fn check_slack(slack: &SlackConfig, manifest: &Manifest) -> Result<()> {
    if let Some(agent) = &slack.agent {
        check_slack_agent(agent, manifest).map_err(|e| anyhow::anyhow!("[slack]: {e}"))?;
    }
    for (id, channel) in &slack.channel {
        check_channel(id, channel, manifest)
            .map_err(|e| anyhow::anyhow!("[slack.channel.{id}]: {e}"))?;
    }
    // Channels that are all `off` and no default to fall back to: a bot that
    // connects, listens, and can answer nowhere.
    if slack.is_configured() && slack.agent.is_none() && !slack.channel.values().any(|c| !c.off) {
        bail!(
            "[slack]: nothing to answer with. Name a default `agent`, or name one in a \
             `[slack.channel.<id>]`."
        );
    }
    Ok(())
}

fn check_slack_agent(agent: &str, manifest: &Manifest) -> Result<()> {
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
/// would surface as a discovery failure against a URL nobody meant to write.
/// A token-backed connection never reaches the OAuth resolver's own check:
/// `EnvCredentials` sends `$token_env` wherever the URL points.
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
            agent = "typo"
            "#,
        );
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("names no agent"), "{err}");
    }
}
