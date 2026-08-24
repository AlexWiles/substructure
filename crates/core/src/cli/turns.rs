//! Where a turn runs: an engine here, or the deployment the file names.
//!
//! The only place a command asks [`target`]. `subs run` drives one turn
//! through this seam and `subs chat` drives many; neither knows which side
//! answered.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::api::v1::RunFormat;
use crate::protocol::{ClientInput, Issuer, Requester, SessionOwner, Subject};
use crate::providers::sqlite::{SqliteBlobStore, SqliteDb};
use crate::session::index::SessionFilter;
use crate::transport::ag_ui::events::{AgUiEvent, AgUiInterrupt};
use crate::transport::ag_ui::run as ag_ui_run;
use crate::transport::ag_ui::snapshot;
use crate::transport::ag_ui::translator::run_ag_ui_translation;
use crate::transport::channel::ChannelContext;
use crate::transport::mcp_auth::McpAuthDeps;
use crate::transport::push::PushAdapter;
use crate::{Caller, Runtime};

use super::cloud::context::Context as CloudContext;
use super::cloud::project_config::{self, ProjectConfig};
use super::cloud::{CloudGlobals, ProjectScope};
use super::env::{EnvVars, OutputFormat};
use super::output::Renderer;
use super::output::{unfinished, TurnEnd, TurnRender};
use super::target::target;
use super::{local, run_remote, DEFAULT_TENANT, LOCAL_SUBJECT};

const SPAN: &str = "cli_client_input";

/// One user message, the shape a `subs run` positional argument means — and
/// the shape a line typed at `subs chat` means.
pub(crate) fn message_input(message: String, agent_id: String) -> ClientInput {
    ClientInput::Message {
        agent_id,
        turn_id: None,
        message: crate::protocol::DraftMessage {
            id: None,
            role: crate::protocol::Role::User,
            content: Some(crate::protocol::Content::Text(message)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: None,
        },
        stream: true,
        queue: false,
    }
}

/// Which agent a command drives. `--agent`, else `[run].agent` when the file
/// pins one; nothing is inferred from what happens to be declared, because an
/// engine that picks for you picks differently the day a second agent is added.
pub(crate) fn select_agent(
    flag: Option<String>,
    pinned: Option<String>,
    declared: &[String],
) -> Result<String> {
    let Some(agent_id) = flag.or(pinned) else {
        anyhow::bail!(
            "no agent given. Pass `--agent <id>` or set `[run].agent`. Declared: {}",
            crate::worker::directory::declared(declared)
        );
    };
    // A typo here would otherwise surface as a failed decision one turn later.
    if !declared.contains(&agent_id) {
        anyhow::bail!(
            "no [agent.{agent_id}] in substructure.toml. Declared: {}",
            crate::worker::directory::declared(declared)
        );
    }
    Ok(agent_id)
}

/// One turn, wherever it runs. Both sides end at the same AG-UI event stream,
/// so only the submission differs.
#[async_trait]
pub(crate) trait Turns: Send + Sync {
    async fn drive(&mut self, input: ClientInput, renderer: &mut Renderer) -> Result<TurnEnd>;

    async fn parked(&self) -> Result<Vec<AgUiInterrupt>>;

    /// Local: wait for the session index to read the turn. Remote: the
    /// deployment does this.
    async fn wait_for_index(&self) {}
}

pub(crate) struct Open {
    pub globals: CloudGlobals,
    pub session: String,
    pub db: Option<String>,
    pub output: OutputFormat,
}

pub(crate) async fn open(cfg: &ProjectConfig, o: Open) -> Result<Box<dyn Turns>> {
    if target(&o.globals)?.here().is_none() {
        let scope = ProjectScope {
            org: None,
            project: None,
            globals: o.globals,
        };
        let (ctx, project) = CloudContext::from_project(&scope).await?;
        return Ok(Box::new(RemoteTurns {
            ctx,
            project,
            session_id: o.session,
            format: run_remote::format_for(o.output),
        }));
    }

    let env = match EnvVars::load(cfg.provider_bindings(), false) {
        Some(e) => e,
        None => std::process::exit(2),
    };

    let db_path = o.db.unwrap_or_else(|| cfg.db_path());
    project_config::ensure_parent(&db_path)?;

    let db = SqliteDb::open(&db_path, std::time::Duration::from_secs(5))?;
    let blobs = Arc::new(SqliteBlobStore::new(db.clone()));
    let (rt, adapter, mcp_auth) = local::start_engine(db, blobs, env.providers, cfg).await?;

    Ok(Box::new(LocalTurns {
        ctx: ChannelContext::new(rt.clone(), CancellationToken::new()),
        rt,
        _adapter: adapter,
        _mcp_auth: mcp_auth,
        caller: Caller::System {
            tenant_id: DEFAULT_TENANT.to_string(),
        },
        owner: SessionOwner {
            tenant_id: DEFAULT_TENANT.to_string(),
            requester: Requester::private(Subject::new(Issuer::cli(), LOCAL_SUBJECT)),
            metadata: Default::default(),
        },
        session_id: o.session,
    }))
}

struct LocalTurns {
    ctx: ChannelContext,
    rt: Arc<Runtime>,
    _adapter: Arc<PushAdapter>,
    _mcp_auth: Option<Arc<McpAuthDeps>>,
    caller: Caller,
    owner: SessionOwner,
    session_id: String,
}

#[async_trait]
impl Turns for LocalTurns {
    async fn drive(&mut self, input: ClientInput, renderer: &mut Renderer) -> Result<TurnEnd> {
        let translated = !renderer.is_raw();
        let turn = ag_ui_run::start(
            &self.ctx,
            &self.caller,
            &self.owner,
            &self.session_id,
            input,
            SPAN,
            translated,
        )
        .await?;

        let mut render = TurnRender::new(renderer);
        let finished = match turn.deltas {
            Some(deltas) => {
                let mut events = run_ag_ui_translation(
                    turn.events,
                    deltas,
                    self.session_id.clone(),
                    turn.turn_id,
                    self.ctx.shutdown.clone(),
                );
                while let Some(event) = events.recv().await {
                    render.accept(vec![event])?;
                }
                render.terminated()
            }
            None => {
                let mut events = turn.events;
                let mut ended = false;
                while let Some(event) = events.recv().await {
                    ended |= event.ends_run();
                    render.raw(&event)?;
                }
                ended
            }
        };

        if !finished {
            return Err(unfinished());
        }
        Ok(render.into_end())
    }

    async fn parked(&self) -> Result<Vec<AgUiInterrupt>> {
        let Ok(session) = self.rt.get_session(DEFAULT_TENANT, &self.session_id).await else {
            return Ok(Vec::new());
        };
        Ok(snapshot::open_interrupts(&session.state))
    }

    /// Best effort, so `subs sessions list` shows the turn. The index polls,
    /// and nothing runs after this process stops.
    async fn wait_for_index(&self) {
        const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);
        const POLL: std::time::Duration = std::time::Duration::from_millis(20);

        let Ok(session) = self.rt.get_session(DEFAULT_TENANT, &self.session_id).await else {
            return;
        };
        let filter = SessionFilter {
            tenant_id: Some(DEFAULT_TENANT.to_string()),
            session_id: Some(self.session_id.clone()),
            ..Default::default()
        };

        let deadline = tokio::time::Instant::now() + TIMEOUT;
        while tokio::time::Instant::now() < deadline {
            if let Ok(page) = self.rt.list_sessions(&filter).await {
                if page.items.iter().any(|s| s.seq >= session.seq) {
                    return;
                }
            }
            tokio::time::sleep(POLL).await;
        }
        tracing::debug!(
            session_id = %self.session_id,
            "session index did not catch up before exit"
        );
    }
}

struct RemoteTurns {
    ctx: CloudContext,
    project: String,
    session_id: String,
    format: RunFormat,
}

#[async_trait]
impl Turns for RemoteTurns {
    async fn drive(&mut self, input: ClientInput, renderer: &mut Renderer) -> Result<TurnEnd> {
        run_remote::drive(
            &self.ctx,
            &self.project,
            &self.session_id,
            input,
            renderer,
            self.format,
        )
        .await
    }

    async fn parked(&self) -> Result<Vec<AgUiInterrupt>> {
        let path = format!(
            "/api/v1/projects/{}/sessions/{}/ag-ui/connect",
            self.project, self.session_id
        );
        let body = serde_json::json!({ "threadId": self.session_id, "runId": "reattach" });
        let mut end = TurnEnd::default();
        let mut failed: Option<anyhow::Error> = None;
        self.ctx
            .client
            .post_sse(&path, &body, |line| {
                let Some(data) = line.strip_prefix("data:") else {
                    return;
                };
                let data = data.trim();
                if data.is_empty() {
                    return;
                }
                match serde_json::from_str::<AgUiEvent>(data) {
                    Ok(event) => end.note(&[event]),
                    Err(e) => {
                        failed = Some(anyhow::anyhow!(
                            "the deployment sent an AG-UI event this CLI cannot read: {e}"
                        ))
                    }
                }
            })
            .await?;
        if let Some(e) = failed {
            return Err(e);
        }
        Ok(end.interrupts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Content, Role};

    fn declared() -> Vec<String> {
        vec!["assistant".to_string(), "researcher".to_string()]
    }

    #[test]
    fn the_flag_wins_over_the_pinned_agent() {
        let agent = select_agent(
            Some("researcher".to_string()),
            Some("assistant".to_string()),
            &declared(),
        )
        .unwrap();
        assert_eq!(agent, "researcher");
        assert_eq!(
            select_agent(None, Some("assistant".to_string()), &declared()).unwrap(),
            "assistant"
        );
    }

    /// Nothing is inferred from what happens to be declared: an engine that
    /// picks for you picks differently the day a second agent is added.
    #[test]
    fn no_agent_anywhere_is_an_error_listing_the_declared_ones() {
        let err = select_agent(None, None, &declared())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no agent given"), "got {err}");
        assert!(err.contains("assistant, researcher"), "got {err}");
    }

    /// The preflight is what makes a typo cost nothing: without it the run
    /// creates a session and fails one decision later.
    #[test]
    fn an_undeclared_agent_fails_before_the_session_exists() {
        let err = select_agent(Some("assistnat".to_string()), None, &declared())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no [agent.assistnat]"), "got {err}");
        assert!(err.contains("assistant, researcher"), "got {err}");
    }

    #[test]
    fn a_positional_message_is_a_user_turn_for_the_selected_agent() {
        match message_input("hi".to_string(), "assistant".to_string()) {
            ClientInput::Message {
                agent_id, message, ..
            } => {
                assert_eq!(agent_id, "assistant");
                assert_eq!(message.role, Role::User);
                assert_eq!(message.content.as_ref().and_then(Content::text), Some("hi"));
            }
            other => panic!("expected client.message, got {other:?}"),
        }
    }
}
