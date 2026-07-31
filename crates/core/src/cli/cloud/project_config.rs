use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use toml_edit::{DocumentMut, Item, Table, Value};

use crate::cli::env::{OutputFormat, ProviderBinding, ProviderKind};
use crate::connectors::registry::ConnectionSpec;
use crate::protocol::{
    AgentConfig, AgentTool, ConnectorProtocol, Handler, LlmFormat, McpServer, RetryPolicy, SubAgent,
};
use crate::runtime::llm::{LlmBlock, LlmBlocks};
use crate::runtime::worker::{AgentEntry, WorkerEndpoint};

pub const FILENAME: &str = "substructure.toml";
pub const DEFAULT_DB: &str = "substructure.db";

/// One system, described once.
///
/// A file carries two roles, either or both: **an engine you run** (`db`,
/// `log`, `[run]`, `[server]`) and **a deployment you administer**
/// (`[deployment]`). What the app *is* — `name`, `[agent.<id>]`, `[llm.<id>]`,
/// `[slack]`, `[mcp.<id>]` — is one declaration whichever role reads it, so a
/// self-hosted system is served and administered from the same file rather
/// than two that have to agree.
///
/// A role is present when its keys are: `[deployment]` is what `subs apply`
/// and `subs sessions` act on, and it is also what decides that a connection's
/// credential belongs to the deployment rather than to the engine here.
///
/// Precedence for anything the CLI also accepts as a flag is
/// **flag > environment > this > default**, so pinning something here never
/// takes an override away.
///
/// Per-invocation arguments are deliberately absent: `--input`, `--session`, and
/// `-c` itself say what one run is doing, not how the environment is set up.
/// Secrets are absent for the same reason they are absent from `[mcp]` — a
/// committed file must not be able to hold one, so the signing secret is named
/// rather than written.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnvConfig {
    /// The app's name. `subs apply` creates the app from it when nothing is
    /// pinned, and renames when it changes: the file is the source of truth.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Engine state: events, sessions, and the credentials `subs mcp login`
    /// authorized [default: `substructure.db`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db: Option<String>,
    /// Log filter in `RUST_LOG` syntax: a bare level (`info`) or per-target
    /// directives (`substructure_core=debug,warn`). `$RUST_LOG` still wins.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<String>,
    /// The LLM blocks this app declares, keyed by the name an agent names. The
    /// declaration travels with the app; the credential binds per environment.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub llm: BTreeMap<String, ProviderSpec>,
    /// The agents this app declares, keyed by the id a client routes on. Each
    /// section is the wire `AgentConfig` plus where its decisions go.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub agent: BTreeMap<String, AgentSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<RunConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server: Option<ServerConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slack: Option<SlackConfig>,
    /// MCP servers this app may reach, keyed by the id an agent names. An
    /// engine here dials them itself; a deployment is told the id and URL and
    /// holds the credential, so `auth` is the engine's half alone.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, ConnectionSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deployment: Option<Deployment>,
}

/// The server this file administers — the hosted cloud, a self-hosted
/// deployment, or someone else's `subs serve` — and what it is pinned to there.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Deployment {
    /// The API to talk to [default: `https://api.substructure.ai`]. A `--url`
    /// flag still overrides it; `$SUBS_API_URL` only fills in when neither is
    /// set. `subs login` reads it too, so the token is stored under the same
    /// server the rest of the file's commands reach.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
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
    /// (`ANTHROPIC_API_KEY` and so on).
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
    fn block(&self) -> LlmBlock {
        match self.kind {
            ProviderKind::Worker => LlmBlock::worker(self.format),
            _ => LlmBlock::engine(),
        }
    }

    /// The variable this block's key is read from, for the types that need one.
    fn api_key_env(&self) -> Option<String> {
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
/// `session.start`, exactly as it did before the file could.
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

/// Defaults for `subs run`.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputFormat>,
}

/// `subs serve` only.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// Client and worker authentication [default: true]. `false` serves
    /// without issuing tokens, which is for a server nothing outside this
    /// machine can reach.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<bool>,
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

impl SlackConfig {
    /// Whether this section configures a bot at all.
    pub fn is_configured(&self) -> bool {
        self.agent.is_some() || !self.channel.is_empty()
    }
}

impl EnvConfig {
    /// Every connection this environment declares, keyed by the id an agent
    /// names.
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

    pub fn db_path(&self) -> String {
        self.db.clone().unwrap_or_else(|| DEFAULT_DB.to_string())
    }

    pub fn slack_agent(&self) -> Option<String> {
        self.slack.as_ref()?.agent.clone()
    }

    /// The declared blocks as the engine reads them: venue and wire shape,
    /// never a credential.
    pub fn llm_blocks(&self) -> LlmBlocks {
        self.llm
            .iter()
            .map(|(name, spec)| (name.clone(), spec.block()))
            .collect()
    }

    /// The blocks the engine runs itself, each with the variable its key comes
    /// from. `worker` blocks are absent: they never need a credential here.
    pub fn provider_bindings(&self) -> Vec<ProviderBinding> {
        self.llm
            .iter()
            .filter(|(_, spec)| spec.kind != ProviderKind::Worker)
            .filter_map(|(name, spec)| {
                Some(ProviderBinding {
                    name: name.clone(),
                    kind: spec.kind,
                    api_key_env: spec.api_key_env()?,
                    base_url: spec.base_url.clone(),
                })
            })
            .collect()
    }

    /// Every agent this app declares, keyed by the id a client routes on.
    pub fn agents(&self) -> BTreeMap<String, AgentEntry> {
        self.agent
            .iter()
            .map(|(id, section)| (id.clone(), section.to_entry()))
            .collect()
    }

    /// The declared agent ids, for the error that says what could have been named.
    pub fn agent_ids(&self) -> Vec<String> {
        self.agent.keys().cloned().collect()
    }

    /// Whether an engine here authenticates its clients and workers
    /// [default: yes].
    pub fn server_auth(&self) -> bool {
        self.server.as_ref().and_then(|s| s.auth).unwrap_or(true)
    }

    pub fn deployment_url(&self) -> Option<&str> {
        self.deployment.as_ref()?.url.as_deref()
    }

    pub fn org(&self) -> Option<&str> {
        self.deployment.as_ref()?.org.as_deref()
    }

    pub fn app(&self) -> Option<&str> {
        self.deployment.as_ref()?.app.as_deref()
    }

    /// The deployment section, creating it if the file has none — for the
    /// commands that pin (`subs link`, `subs apply`).
    pub fn deployment_mut(&mut self) -> &mut Deployment {
        self.deployment.get_or_insert_with(Deployment::default)
    }

    fn parse(s: &str, path: &Path) -> Result<Self> {
        let at = path.display();
        let value: toml::Value = toml::from_str(s).map_err(|e| anyhow!("parsing {at}: {e}"))?;
        moved_keys(&value, &at)?;
        let config: EnvConfig = value.try_into().map_err(|e| anyhow!("parsing {at}: {e}"))?;
        for (id, spec) in &config.mcp {
            check_id(id).map_err(|e| anyhow!("{at}: [mcp.{id}]: {e}"))?;
            check_url(&spec.url).map_err(|e| anyhow!("{at}: [mcp.{id}]: {e}"))?;
        }
        for (id, spec) in &config.llm {
            check_llm(id, spec).map_err(|e| anyhow!("{at}: [llm.{id}]: {e}"))?;
        }
        for (id, section) in &config.agent {
            check_agent(id, section, &config).map_err(|e| anyhow!("{at}: [agent.{id}]: {e}"))?;
        }
        if let Some(slack) = &config.slack {
            check_slack(slack, &config).map_err(|e| anyhow!("{at}: {e}"))?;
        }
        Ok(config)
    }
}

/// A block declares one venue, so the fields of the other are not a detail to
/// ignore — they are a misunderstanding of what the block is.
fn check_llm(id: &str, spec: &ProviderSpec) -> Result<()> {
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

/// Every name an agent uses is declared in this same file, so a typo is caught
/// here rather than as a failing decision on the first turn.
fn check_agent(id: &str, section: &AgentSection, config: &EnvConfig) -> Result<()> {
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

    // Anything the file seeds has to be a config the engine can actually
    // propose from, so a half-declared one is an error rather than a proposal
    // that fails at the first model call.
    let Some(llm) = section.llm.as_deref() else {
        bail!(
            "no `llm`. Name one of the declared blocks: {}",
            declared(config.llm.keys())
        );
    };
    let Some(block) = config.llm.get(llm) else {
        bail!(
            "`llm = \"{llm}\"` names no block. Declared: {}",
            declared(config.llm.keys())
        );
    };
    if section.model.is_none() {
        bail!("no `model`. An agent that declares an `llm` has to say which model on it.");
    }

    for server in &section.mcp {
        if !config.mcp.contains_key(&server.id) {
            bail!(
                "`mcp` names no connection `{}`. Declared: {}",
                server.id,
                declared(config.mcp.keys())
            );
        }
    }

    // A worker-executed tool comes from worker code, so a file that declares
    // one would be naming a function nothing here can run.
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

/// Every agent the bot routes to is declared in this same file, so a typo is
/// caught here rather than as a bot that answers nowhere.
fn check_slack(slack: &SlackConfig, config: &EnvConfig) -> Result<()> {
    if let Some(agent) = &slack.agent {
        check_slack_agent(agent, config).map_err(|e| anyhow!("[slack]: {e}"))?;
    }
    for (id, channel) in &slack.channel {
        check_channel(id, channel, config).map_err(|e| anyhow!("[slack.channel.{id}]: {e}"))?;
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

fn check_slack_agent(agent: &str, config: &EnvConfig) -> Result<()> {
    if config.agent.contains_key(agent) {
        return Ok(());
    }
    bail!(
        "`agent = \"{agent}\"` names no agent. Declared: {}",
        declared(config.agent.keys())
    )
}

/// A channel says one of two things, and a section that says both or neither
/// is a setting with no meaning rather than one to resolve by precedence.
fn check_channel(id: &str, channel: &SlackChannelConfig, config: &EnvConfig) -> Result<()> {
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
        (Some(agent), false) => check_slack_agent(agent, config),
        (None, true) => Ok(()),
    }
}

fn declared<'a>(mut ids: impl Iterator<Item = &'a String>) -> String {
    let joined = ids.by_ref().cloned().collect::<Vec<_>>().join(", ");
    match joined.is_empty() {
        true => "none".to_string(),
        false => joined,
    }
}

/// The keys and sections that moved, reported where they went rather than as
/// `deny_unknown_fields`' "unknown field", which says nothing about the file
/// this one is.
fn moved_keys(value: &toml::Value, at: &impl std::fmt::Display) -> Result<()> {
    let pins: Vec<&str> = ["url", "org", "app"]
        .into_iter()
        .filter(|k| value.get(k).is_some())
        .collect();
    if value.get("target").is_some() {
        let and_pins = match pins.is_empty() {
            true => String::new(),
            false => format!(", and move `{}` under `[deployment]`", pins.join("`, `")),
        };
        bail!(
            "{at}: `target` is no longer a setting. Delete it{and_pins} — a file describes an \
             engine you run (`db`, `[run]`, `[server]`), a deployment you administer \
             (`[deployment]`), or both."
        );
    }
    if !pins.is_empty() {
        bail!(
            "{at}: `{}` belongs under `[deployment]`, with the server's `url`.",
            pins.join("`, `")
        );
    }
    if value.get("worker").is_some() {
        bail!(
            "{at}: `[worker]` is no longer a setting: a worker belongs to one agent, not to \
             the app. Write `worker = \"<url>\"` under the `[agent.<id>]` it decides for, and \
             leave it off the agents the engine decides for."
        );
    }
    // `[llm] provider = "anthropic"` versus `[llm.claude] type = "anthropic"`:
    // the old form's values are scalars where the new form's are tables.
    if let Some(llm) = value.get("llm").and_then(toml::Value::as_table) {
        if llm.values().any(|v| !v.is_table()) {
            bail!(
                "{at}: `[llm]` now declares named blocks, so that a second one can be added \
                 and an agent can say which it uses. Write `[llm.<name>]` with a `type`, e.g. \
                 `[llm.claude]` / `type = \"anthropic\"`, and name it from `[agent.<id>]`."
            );
        }
    }
    Ok(())
}

/// An id becomes the prefix on every tool name the model sees, so it is held to
/// what a tool name may contain rather than to what TOML accepts as a key.
fn check_id(id: &str) -> Result<()> {
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

/// Rejected while reading the file rather than at the first fetch, where it
/// would surface as a discovery failure against a URL nobody meant to write.
/// A token-backed connection never reaches the OAuth resolver's own check:
/// `EnvCredentials` sends `$token_env` wherever the URL points.
fn check_url(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("`{url}` is not a URL"))?;
    if parsed.scheme() == "https" || crate::connectors::oauth::is_loopback(url) {
        return Ok(());
    }
    bail!("`{url}` is not https: a credential would cross the network in the clear")
}

#[derive(Debug, Clone)]
pub struct Found {
    pub config: EnvConfig,
    pub path: PathBuf,
}

/// The environment file at `path`, or the nearest one above the working
/// directory. An explicit path that does not resolve is an error; discovery
/// finding nothing is not.
pub fn resolve(path: Option<&Path>) -> Result<Option<Found>> {
    match path {
        Some(p) => load_explicit(p).map(Some),
        None => find(),
    }
}

/// What the file says, or the defaults when there is none. For the commands
/// that work without one — an engine runs on defaults, and every setting is
/// also a flag.
pub fn load(path: Option<&Path>) -> Result<EnvConfig> {
    Ok(resolve(path)?.map(|found| found.config).unwrap_or_default())
}

pub fn find_from(start: &Path) -> Result<Option<Found>> {
    let mut dir: &Path = start;
    loop {
        let candidate = dir.join(FILENAME);
        if candidate.is_file() {
            return load_explicit(&candidate).map(Some);
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => return Ok(None),
        }
    }
}

pub fn find() -> Result<Option<Found>> {
    let cwd = env::current_dir().context("could not determine cwd for substructure.toml lookup")?;
    find_from(&cwd)
}

pub fn load_explicit(path: &Path) -> Result<Found> {
    let s = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Found {
        config: EnvConfig::parse(&s, path)?,
        path: path.to_path_buf(),
    })
}

/// Write `config` back, keeping everything about the file that is not a
/// setting. A machine edit must not cost a reader their comments or their
/// layout, so the parsed document is edited in place rather than replaced.
pub fn write(path: &Path, config: &EnvConfig) -> Result<()> {
    let mut rendered: DocumentMut =
        toml_edit::ser::to_document(config).context("serializing substructure.toml")?;
    for (_, item) in rendered.as_table_mut().iter_mut() {
        expand(item, 2);
    }

    let mut doc = match fs::read_to_string(path) {
        Ok(existing) => existing
            .parse::<DocumentMut>()
            .map_err(|e| anyhow!("parsing {}: {e}", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => DocumentMut::new(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    merge(doc.as_table_mut(), rendered.as_table());

    fs::write(path, doc.to_string()).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Serde renders every struct as an inline table. Sections are what a reader
/// expects of the two outermost levels — `[worker]`, `[mcp.sentry]` — and an
/// inline table is what they expect of anything below.
fn expand(item: &mut Item, depth: usize) {
    if depth == 0 {
        return;
    }
    if let Item::Value(Value::InlineTable(inline)) = item {
        *item = Item::Table(std::mem::take(inline).into_table());
    }
    if let Item::Table(table) = item {
        for (_, child) in table.iter_mut() {
            expand(child, depth - 1);
        }
        // `[mcp]` holds nothing of its own, so it is a header nobody needs.
        if !table.is_empty() && table.iter().all(|(_, child)| child.is_table()) {
            table.set_implicit(true);
        }
    }
}

/// Overwrite `target` with `source`, key by key. A key `source` does not carry
/// is a setting that was removed, so it goes; a value that changed keeps its
/// decoration, which is where a trailing comment lives.
fn merge(target: &mut Table, source: &Table) {
    target.retain(|key, _| source.contains_key(key));
    for (key, item) in source.iter() {
        match (target.get_mut(key), item) {
            (Some(Item::Table(existing)), Item::Table(next)) => merge(existing, next),
            // A section someone wrote inline stays inline.
            (Some(Item::Value(Value::InlineTable(existing))), Item::Table(next)) => {
                let mut merged = next.clone().into_inline_table();
                *merged.decor_mut() = existing.decor().clone();
                *existing = merged;
            }
            (Some(Item::Value(existing)), Item::Value(next)) => {
                let mut next = next.clone();
                *next.decor_mut() = existing.decor().clone();
                *existing = next;
            }
            (Some(existing), next) => *existing = next.clone(),
            (None, next) => {
                target.insert(key, next.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> PathBuf {
        // Timestamp alone collides across parallel tests; the counter disambiguates.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("subs-project-test-{nanos}-{seq}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn parse(s: &str) -> Result<EnvConfig> {
        EnvConfig::parse(s, Path::new("substructure.toml"))
    }

    fn ok(s: &str) -> EnvConfig {
        parse(s).unwrap()
    }

    #[test]
    fn a_file_carries_either_role_or_both() {
        let engine = ok("db = \"dev.db\"\n[server]\nport = 9000\n");
        assert_eq!(engine.db_path(), "dev.db");
        assert!(engine.deployment.is_none());

        let deployment = ok("[deployment]\norg = \"org_1\"\n");
        assert_eq!(deployment.org(), Some("org_1"));
        assert_eq!(deployment.db_path(), DEFAULT_DB);

        // One system: served here, administered there, declared once.
        let both = ok(r#"
            name = "support-bot"
            db = "prod.db"

            [llm.claude]
            type = "anthropic"

            [agent.support]
            llm = "claude"
            model = "claude-sonnet-4-5"
            worker = "https://bot.example.com/agent"

            [deployment]
            url = "https://subs.internal"
            app = "app_1"
        "#);
        assert_eq!(both.name.as_deref(), Some("support-bot"));
        assert_eq!(both.db_path(), "prod.db");
        assert_eq!(both.deployment_url(), Some("https://subs.internal"));
        assert_eq!(both.app(), Some("app_1"));
    }

    #[test]
    fn an_empty_file_is_valid_and_is_the_defaults() {
        assert_eq!(ok(""), EnvConfig::default());
        assert_eq!(EnvConfig::default().db_path(), DEFAULT_DB);
        assert!(EnvConfig::default().server_auth());
    }

    #[test]
    fn the_engine_groups_read_back() {
        let cfg = ok(r#"
            db = "dev.substructure.db"
            log = "substructure_core=debug,warn"

            [llm.claude]
            type = "anthropic"

            [agent.support]
            llm = "claude"
            model = "claude-sonnet-4-5"
            worker = "http://localhost:4444"
            signing_secret_env = "SUBS_SIGNING_SECRET"

            [run]
            agent = "support"
            output = "pretty"

            [server]
            port = 9000
            auth = false
        "#);
        assert_eq!(cfg.db_path(), "dev.substructure.db");
        assert_eq!(cfg.log.as_deref(), Some("substructure_core=debug,warn"));
        assert_eq!(
            cfg.agent["support"].worker.as_deref(),
            Some("http://localhost:4444")
        );
        assert_eq!(cfg.llm["claude"].kind, ProviderKind::Anthropic);
        assert_eq!(cfg.run.as_ref().unwrap().agent.as_deref(), Some("support"));
        assert_eq!(cfg.run.clone().unwrap().output, Some(OutputFormat::Pretty));
        assert!(!cfg.server_auth());
        let server = cfg.server.unwrap();
        assert_eq!(server.port, Some(9000));
        // Absent is absent, not a default the flag would then have to beat.
        assert_eq!(server.host, None);
    }

    #[test]
    fn target_says_where_it_went() {
        let err = parse("target = \"local\"\ndb = \"dev.db\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("`target` is no longer"), "got {err}");
        assert!(err.contains("[deployment]"), "got {err}");

        // The pins moved with it, so one message covers the whole edit.
        let err = parse("target = \"remote\"\nurl = \"https://x\"\norg = \"o\"\napp = \"a\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("`url`, `org`, `app`"), "got {err}");

        // And on their own, without a target to hang the message on.
        let err = parse("org = \"acme\"\n").unwrap_err().to_string();
        assert!(
            err.contains("`org`") && err.contains("[deployment]"),
            "got {err}"
        );
    }

    #[test]
    fn a_misspelled_key_is_a_parse_error_not_a_silent_no_op() {
        // An agent section is the wire config plus two hosting keys, and
        // `flatten` blinds serde's own check, so the key set is checked by hand.
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\nsytem = \"be brief\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("sytem"), "got {err}");

        let err = parse("[deployment]\nnmae = \"x\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("nmae"), "got {err}");

        // There is no catalog key: a connection always declares a URL.
        let err = parse("[mcp.sentry]\ncatalog = \"sentry\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("catalog"), "got {err}");
    }

    #[test]
    fn a_connection_is_checked_where_it_was_typed() {
        // An id prefixes every tool name the model sees.
        let err = parse("[mcp.\"my server\"]\nurl = \"https://x/mcp\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot prefix a tool name"), "got {err}");

        // A credential would cross the network in the clear.
        let err = parse("[mcp.sentry]\nurl = \"http://mcp.sentry.dev/mcp\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not https"), "got {err}");

        // Loopback is exempt: nothing off-host sees it.
        ok("[mcp.issues]\nurl = \"http://localhost:4445/mcp\"\n");
    }

    #[test]
    fn an_inline_secret_is_a_parse_error() {
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\nsigning_secret = \"s3cret\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("signing_secret"),
            "a committed file must not be able to hold a secret; got {err}"
        );

        let err = parse("[llm.claude]\ntype = \"anthropic\"\napi_key = \"sk-1\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("api_key"), "got {err}");

        let err = parse("[mcp.sentry]\nurl = \"https://x/mcp\"\nauth = { token = \"t\" }\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("token"), "got {err}");
    }

    /// An agent whose worker authors its whole config declares nothing but the
    /// URL — the file has no business naming an `llm` or a `model` it will
    /// never use.
    #[test]
    fn an_agent_may_delegate_everything_to_its_worker() {
        let cfg = ok(r#"
            [agent.reggu]
            worker = "http://localhost:4000/substructure/agent"
        "#);
        let entry = &cfg.agents()["reggu"];
        assert!(entry.config.is_none(), "nothing to seed");
        assert!(entry.worker.is_some(), "the worker authors it");
    }

    /// …but only a worker can author one, so an agent that declares neither a
    /// config nor a worker is a declaration nothing can act on.
    #[test]
    fn an_agent_that_declares_nothing_at_all_is_an_error() {
        let err = parse("[agent.a]\n").unwrap_err().to_string();
        assert!(err.contains("declares nothing"), "got {err}");
    }

    /// A half-declared config would seed a proposal the engine cannot call
    /// with, so it fails here instead of at the first model call.
    #[test]
    fn a_partly_declared_config_is_an_error() {
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nworker = \"https://a/agent\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no `model`"), "got {err}");

        let err = parse("[agent.a]\nsystem = \"be brief\"\nworker = \"https://a/agent\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("no `llm`"), "got {err}");
    }

    /// Every name an agent uses is declared in this same file, so a typo is a
    /// parse error rather than a decision that fails on the first turn.
    #[test]
    fn an_agents_references_are_checked_against_the_file() {
        let err = parse("[agent.a]\nmodel = \"m\"\n").unwrap_err().to_string();
        assert!(err.contains("no `llm`"), "got {err}");

        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n[agent.a]\nllm = \"clade\"\nmodel = \"m\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("names no block"), "got {err}");
        assert!(
            err.contains("claude"),
            "and says what is declared; got {err}"
        );

        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\nmcp = [{ id = \"sentry\" }]\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no connection `sentry`"), "got {err}");
    }

    /// A `worker` block's calls are made by a worker, so an agent on one that
    /// has no worker could never make a call at all.
    #[test]
    fn an_agent_on_a_worker_block_needs_a_worker() {
        let err = parse(
            "[llm.byo]\ntype = \"worker\"\nformat = \"anthropic\"\n\n\
             [agent.a]\nllm = \"byo\"\nmodel = \"m\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("needs a `worker`"), "got {err}");

        // With one attached it parses.
        ok("[llm.byo]\ntype = \"worker\"\n\n\
            [agent.a]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"https://a/agent\"\n");
    }

    /// A field belonging to the other venue is a misunderstanding of the block,
    /// not a detail to ignore.
    #[test]
    fn a_block_is_checked_against_its_own_type() {
        let err = parse("[llm.claude]\ntype = \"anthropic\"\nformat = \"anthropic\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("only applies to"), "got {err}");

        let err = parse("[llm.byo]\ntype = \"worker\"\napi_key_env = \"K\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("never leaves your worker"), "got {err}");
    }

    /// A file can only declare tools the browser runs: a worker-run tool is
    /// worker code, and nothing here could execute one.
    #[test]
    fn a_file_declared_tool_must_be_client_handled() {
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\n\
             tools = [{ name = \"get_time\" }]\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("handler = \"client\""), "got {err}");
    }

    /// Signing a request nobody receives is a setting with no effect, which is
    /// worse than an error.
    #[test]
    fn a_signing_secret_without_a_worker_is_an_error() {
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\nsigning_secret_env = \"S\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no `worker`"), "got {err}");
    }

    /// The two sections that moved say where they went, rather than reading as
    /// "unknown field".
    #[test]
    fn the_old_worker_and_llm_forms_say_what_replaced_them() {
        let err = parse("[worker]\nurl = \"http://localhost:4444\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("belongs to one agent"), "got {err}");
        assert!(err.contains("[agent.<id>]"), "got {err}");

        let err = parse("[llm]\nprovider = \"anthropic\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("named blocks"), "got {err}");
        assert!(err.contains("[llm.<name>]"), "got {err}");
    }

    /// What the engine reads off the file: the venue per block, and one binding
    /// per block it runs itself.
    #[test]
    fn the_engine_reads_venues_and_bindings_off_the_blocks() {
        let cfg = ok(r#"
            [llm.claude]
            type = "anthropic"

            [llm.cheap]
            type = "openai"
            api_key_env = "MY_OPENAI_KEY"

            [llm.byo]
            type = "worker"
            format = "anthropic"
        "#);

        let blocks = cfg.llm_blocks();
        assert_eq!(blocks.get("claude"), Some(LlmBlock::engine()));
        assert_eq!(
            blocks.get("byo"),
            Some(LlmBlock::worker(Some(LlmFormat::Anthropic)))
        );
        assert_eq!(blocks.declared(), "byo, cheap, claude");

        // A worker block needs no credential where the engine runs.
        let bindings: Vec<(String, String)> = cfg
            .provider_bindings()
            .into_iter()
            .map(|b| (b.name, b.api_key_env))
            .collect();
        assert_eq!(
            bindings,
            [
                ("cheap".to_string(), "MY_OPENAI_KEY".to_string()),
                ("claude".to_string(), "ANTHROPIC_API_KEY".to_string()),
            ]
        );
    }

    /// The directory the engine routes on: config without hosting, hosting
    /// without config.
    #[test]
    fn agents_become_directory_entries() {
        let cfg = ok(r#"
            [llm.claude]
            type = "anthropic"

            [agent.assistant]
            llm = "claude"
            model = "claude-sonnet-4-5"

            [agent.triage]
            llm = "claude"
            model = "claude-haiku-4-5"
            worker = "https://triage.internal/agent"
        "#);
        let agents = cfg.agents();
        assert!(agents["assistant"].worker.is_none(), "the engine decides");
        assert!(agents["assistant"].config.is_some(), "and needs a config");
        assert_eq!(
            agents["triage"].worker.as_ref().map(|w| w.url.as_str()),
            Some("https://triage.internal/agent")
        );
        // The hosting never crosses the wire.
        let wire = serde_json::to_value(agents["triage"].config.as_ref().unwrap()).unwrap();
        assert!(wire.get("worker").is_none(), "got {wire}");
        assert_eq!(wire["llm"], "claude");
    }

    #[test]
    fn an_output_mode_that_does_not_exist_is_a_parse_error() {
        // It used to fall back to `ag-ui`, so a typo silently changed the mode.
        let err = parse("[run]\noutput = \"pretyy\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("pretyy"), "got {err}");
    }

    #[test]
    fn everything_set_survives_a_round_trip() {
        let cfg = ok(r#"
            name = "support-bot"
            db = "dev.db"
            log = "info"

            [llm.cheap]
            type = "openai"
            api_key_env = "MY_OPENAI_KEY"
            base_url = "https://openai.internal"

            [llm.byo]
            type = "worker"
            format = "anthropic"

            [agent.support]
            llm = "cheap"
            model = "gpt-5-mini"
            system = "Be brief."
            stream = true
            worker = "http://localhost:4444"
            signing_secret_env = "S"
            mcp = [{ id = "sentry" }]
            sub_agents = [{ id = "researcher", description = "Finds sources" }]
            tools = [{ name = "confirm", description = "Ask", handler = "client" }]

            [agent.researcher]
            llm = "cheap"
            model = "gpt-5-mini"

            [run]
            agent = "support"
            output = "jsonl"

            [server]
            host = "0.0.0.0"
            port = 9000
            auth = false

            [slack]
            agent = "support"

            [slack.channel.C0ENGOPS]
            agent = "researcher"

            [slack.channel.C0RANDOM]
            off = true

            [mcp.sentry]
            url = "https://mcp.sentry.dev/mcp"
            prefix_tools = false

            [deployment]
            url = "https://subs.internal"
            org = "org_1"
            app = "app_1"
        "#);
        let written = toml::to_string_pretty(&cfg).unwrap();
        assert_eq!(ok(&written), cfg, "written back as {written}");
    }

    /// Serde renders in declaration order, and a top-level scalar written after
    /// a section would parse back as that section's key.
    #[test]
    fn a_written_file_keeps_its_scalars_above_its_sections() {
        let path = tmpdir().join(FILENAME);
        let mut cfg = ok("db = \"dev.db\"\n[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\n");
        cfg.name = Some("support-bot".into());
        cfg.deployment_mut().app = Some("app_1".into());
        write(&path, &cfg).unwrap();

        assert_eq!(load_explicit(&path).unwrap().config, cfg);
    }

    #[test]
    fn unset_settings_are_not_written_back() {
        let cfg = ok("[deployment]\norg = \"acme\"\n");
        let out = toml::to_string_pretty(&cfg).unwrap();
        assert_eq!(out.trim(), "[deployment]\norg = \"acme\"", "got {out}");
    }

    #[test]
    fn an_empty_slack_section_is_not_a_configured_bot() {
        assert_eq!(ok("[slack]\n").slack_agent(), None);
        assert_eq!(EnvConfig::default().slack_agent(), None);
        assert!(!ok("[slack]\n").slack.unwrap().is_configured());

        // The old bare key is gone, and says so rather than doing nothing.
        let err = parse("slack_agent = \"helper\"\n").unwrap_err().to_string();
        assert!(err.contains("slack_agent"), "got {err}");
    }

    fn agents() -> String {
        "[llm.claude]\ntype = \"anthropic\"\n\n\
         [agent.support]\nllm = \"claude\"\nmodel = \"m\"\n\n\
         [agent.oncall]\nllm = \"claude\"\nmodel = \"m\"\n\n"
            .to_string()
    }

    fn slack(s: &str) -> Result<EnvConfig> {
        parse(&(agents() + s))
    }

    /// A channel names an agent rather than restating a prompt or a tool list:
    /// `[agent.<id>]` already is that bundle.
    #[test]
    fn a_channel_names_the_agent_that_answers_there() {
        let cfg = slack(
            "[slack]\nagent = \"support\"\n\n\
             [slack.channel.C0ENGOPS]\nagent = \"oncall\"\n\n\
             [slack.channel.C0RANDOM]\noff = true\n",
        )
        .unwrap();
        let s = cfg.slack.unwrap();
        assert_eq!(s.agent.as_deref(), Some("support"));
        assert_eq!(s.channel["C0ENGOPS"].agent(), Some("oncall"));
        // `off` is the absence of an agent, however the section spelled it.
        assert_eq!(s.channel["C0RANDOM"].agent(), None);
        assert!(s.is_configured());
    }

    /// A default is not required: naming channels alone is the allowlist.
    #[test]
    fn channels_without_a_default_are_a_complete_section() {
        let cfg = slack("[slack.channel.C0ENGOPS]\nagent = \"oncall\"\n").unwrap();
        let s = cfg.slack.unwrap();
        assert_eq!(s.agent, None);
        assert!(s.is_configured(), "the bot is on, in one channel");
    }

    /// Every name the bot routes to is declared in this same file, so a typo
    /// is caught here rather than as a bot that answers nowhere.
    #[test]
    fn a_channels_agent_is_checked_against_the_file() {
        let err = slack("[slack]\nagent = \"suport\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("names no agent"), "got {err}");
        assert!(err.contains("oncall, support"), "and says which; got {err}");

        let err = slack("[slack.channel.C0ENGOPS]\nagent = \"on-call\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("[slack.channel.C0ENGOPS]"), "got {err}");
        assert!(err.contains("names no agent"), "got {err}");
    }

    /// The likeliest mistake, and one that would otherwise match no event and
    /// report nothing.
    #[test]
    fn a_channel_name_is_not_a_channel_id() {
        let err = slack("[slack.channel.\"#eng-oncall\"]\nagent = \"oncall\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("is a name, not a channel id"), "got {err}");
        assert!(err.contains("About tab"), "and where to get one; got {err}");
    }

    /// A channel says one of two things. Both, or neither, is a setting with
    /// no meaning rather than one to resolve by precedence.
    #[test]
    fn a_channel_that_says_both_or_neither_is_an_error() {
        let err = slack("[slack.channel.C0RANDOM]\nagent = \"oncall\"\noff = true\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("contradict"), "got {err}");

        let err = slack("[slack.channel.C0RANDOM]\n").unwrap_err().to_string();
        assert!(err.contains("declares nothing"), "got {err}");
    }

    /// A bot that connects, listens, and can answer nowhere.
    #[test]
    fn a_section_that_serves_nowhere_is_an_error() {
        let err = slack("[slack.channel.C0RANDOM]\noff = true\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("nothing to answer with"), "got {err}");
    }

    #[test]
    fn writing_keeps_comments_layout_and_everything_it_did_not_change() {
        let path = tmpdir().join(FILENAME);
        fs::write(
            &path,
            "# how this app is deployed\n\
             name = \"support-bot\"\n\
             \n\
             [llm.claude]\n\
             type = \"anthropic\"\n\
             \n\
             [agent.support]\n\
             llm = \"claude\"\n\
             model = \"claude-sonnet-4-5\"\n\
             # where the agent runs\n\
             worker = \"https://bot.example.com/agent\"\n\
             \n\
             [mcp.sentry]\n\
             url = \"https://mcp.sentry.dev/mcp\"\n\
             \n\
             [deployment]\n\
             org = \"old\"        # pinned by hand\n",
        )
        .unwrap();

        let mut cfg = load_explicit(&path).unwrap().config;
        cfg.deployment_mut().org = Some("new".into());
        cfg.deployment_mut().app = Some("app_1".into());
        write(&path, &cfg).unwrap();

        let after = fs::read_to_string(&path).unwrap();
        assert!(after.contains("# how this app is deployed"), "{after}");
        assert!(after.contains("# where the agent runs"), "{after}");
        assert!(
            after.contains("org = \"new\"        # pinned by hand"),
            "{after}"
        );
        assert!(after.contains("app = \"app_1\""), "{after}");
        assert!(after.contains("[mcp.sentry]"), "{after}");
    }

    #[test]
    fn writing_removes_a_setting_that_is_no_longer_set() {
        let path = tmpdir().join(FILENAME);
        fs::write(&path, "[deployment]\norg = \"acme\"\napp = \"app_1\"\n").unwrap();

        let mut cfg = load_explicit(&path).unwrap().config;
        cfg.deployment_mut().app = None;
        write(&path, &cfg).unwrap();

        let after = fs::read_to_string(&path).unwrap();
        assert!(!after.contains("app"), "{after}");
        assert!(after.contains("org = \"acme\""), "{after}");
    }

    #[test]
    fn an_explicit_config_path_that_is_missing_is_an_error() {
        let missing = tmpdir().join("nope.toml");
        assert!(resolve(Some(&missing)).is_err());
    }

    #[test]
    fn load_without_a_file_is_the_defaults() {
        let root = tmpdir().join("isolated");
        fs::create_dir_all(&root).unwrap();
        assert!(find_from(&root).unwrap().is_none());
    }

    #[test]
    fn find_walks_up_from_cwd_to_first_match() {
        let root = tmpdir();
        let nested = root.join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        let cfg_path = root.join(FILENAME);
        fs::write(
            &cfg_path,
            "[deployment]\norg = \"org-x\"\napp = \"app-y\"\n",
        )
        .unwrap();

        let found = find_from(&nested).unwrap().expect("should find ancestor");
        assert_eq!(found.path, cfg_path);
        assert_eq!(found.config.org(), Some("org-x"));
        assert_eq!(found.config.app(), Some("app-y"));
    }

    #[test]
    fn nearest_subs_toml_wins_over_ancestor() {
        let root = tmpdir();
        let nested = root.join("inner");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join(FILENAME), "[deployment]\norg = \"outer\"\n").unwrap();
        fs::write(nested.join(FILENAME), "[deployment]\norg = \"inner\"\n").unwrap();

        let found = find_from(&nested).unwrap().unwrap();
        assert_eq!(found.config.org(), Some("inner"));
    }
}
