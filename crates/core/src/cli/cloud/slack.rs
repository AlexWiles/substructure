//! `subs slack connect`: putting a workspace's bot token where the bot runs.
//!
//! Consent is a human in a browser either way; what differs is who is left
//! holding the token, and the file says which. A file naming a `[remote]`
//! asks that server to run Slack's install flow, and the token never touches
//! this machine. An engine you run here has no install flow to offer — Socket
//! Mode reads its two tokens out of the environment — so that side is
//! explained rather than pretended.
//!
//! The browser is where consent lands, and it lands on the deployment rather
//! than here, so the outcome is read from what the org holds: the workspaces
//! before, against the workspaces after. A Slack app installs once per
//! workspace, so a second install of one already connected refreshes its token
//! and shows up as nothing new — which is what the wait says when it ends
//! having seen none.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{bail, Result};
use clap::Subcommand;
use serde::Serialize;

use crate::api::v1::{SlackConnection, SlackInstallUrl, SlackStatus};

use super::context::Context;
use super::project_config::{self, Found};
use super::{notices, print, OrgScope};

/// How long the CLI waits for the browser before saying where to finish.
const INSTALL_TIMEOUT: Duration = Duration::from_secs(300);
const POLL_INTERVAL: Duration = Duration::from_secs(3);

/// Where the Slack app is built, which is a page rather than a command.
pub const SLACK_NEW_APP: &str = "https://api.slack.com/apps?new_app=1";
pub const SLACK_DOCS: &str = "https://substructure.ai/docs/slack";

#[derive(Subcommand)]
pub enum SlackCommand {
    /// Connect a Slack workspace to this org, opening a browser for consent.
    Connect {
        /// Print the install URL instead of opening a browser.
        #[arg(long)]
        no_browser: bool,
        #[command(flatten)]
        scope: OrgScope,
    },
}

pub async fn run(command: SlackCommand) -> Result<()> {
    match command {
        SlackCommand::Connect { no_browser, scope } => connect(no_browser, scope).await,
    }
}

/// What `--json` emits.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Connected<'a> {
    org_id: &'a str,
    /// `connected` when a workspace arrived, `pending` when the wait ended
    /// without one.
    status: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace: Option<&'a SlackConnection>,
    /// What the org already held, which is also what a refresh of one of these
    /// leaves unchanged.
    connected: &'a [SlackConnection],
}

async fn connect(no_browser: bool, scope: OrgScope) -> Result<()> {
    let found = project_config::resolve(scope.globals.config.as_deref())?;
    // A file that names no remote is an engine run from here, which holds its
    // own credential. No file at all names no engine either: the command means
    // the server the flags and the default resolve to.
    if let Some(found) = &found {
        if found.config.remote.is_none() {
            bail!("{}", engine_here(found));
        }
    }

    // A credential the deployment refuses is the client's problem: it logs in
    // and carries on, so the feature probe is also this command's login.
    let ctx = Context::load(&scope.globals)?;
    // Before the org is resolved: an org picker is a question worth not asking
    // when the answer cannot be acted on anyway.
    require_slack(&ctx).await?;
    let org = ctx.require_org(scope.org.as_deref()).await?;

    // What the org holds before consent, which is what the outcome is read
    // against. Also the deployment's own answer about whether it connects
    // workspaces at all, which `meta` only says for the deployment as a whole.
    let before = status(&ctx, &org).await?;
    if !before.configured {
        bail!("{}", no_slack_app());
    }

    let install: SlackInstallUrl = ctx
        .client
        .get(&format!("/api/v1/orgs/{org}/slack/install-url"))
        .await?;

    // The URL is the one thing the reader cannot do without, so `--json` puts
    // it where it does not spoil the document on stdout.
    let say = |line: String| match scope.globals.json {
        true => eprintln!("{line}"),
        false => println!("{line}"),
    };
    say(format!(
        "Open this URL to add the bot to your workspace:\n  {}",
        install.url
    ));
    if !no_browser && webbrowser::open(&install.url).is_ok() {
        say("Opened your browser to install.".into());
    }

    let arrived = wait_for_workspace(&ctx, &org, &before.connections).await?;

    if scope.globals.json {
        return print::json(&Connected {
            org_id: &org,
            status: match arrived.is_some() {
                true => "connected",
                false => "pending",
            },
            workspace: arrived.as_ref(),
            connected: &before.connections,
        });
    }

    let Some(workspace) = arrived else {
        waiting(&before.connections);
        return Ok(());
    };
    println!("Connected {} to {org}.", workspace.label());
    routing_note(found.as_ref());
    // Org-scoped, so there may be no project to report on: this command
    // connects a workspace whether or not this file pins one.
    if let Ok(Some(project)) = ctx.pinned_project(None).await {
        notices::remaining(&ctx, &project, &scope.globals).await;
    }
    Ok(())
}

/// A deployment that does not install Slack apps says so before a browser is
/// opened, rather than 404-ing halfway.
///
/// The gate is the feature, not its absence: a deployment that advertises
/// nothing is one this CLI predates, and there is no older shape to fall back
/// to. Only the deployment's own answer decides that — see [`Context::meta`].
async fn require_slack(ctx: &Context) -> Result<()> {
    if ctx.meta().await?.has("slack") {
        return Ok(());
    }
    bail!("{}", no_slack_app())
}

async fn status(ctx: &Context, org: &str) -> Result<SlackStatus> {
    ctx.client.get(&format!("/api/v1/orgs/{org}/slack")).await
}

/// The deployment holds no Slack app, so it has no workspace to install one
/// into — whether it said so in `meta` or in the status it answered.
fn no_slack_app() -> String {
    format!(
        "this deployment does not install Slack apps, so it has no workspace to connect.\n\
         An engine you run answers Slack over Socket Mode instead, with its own credential:\n{}",
        socket_mode_steps()
    )
}

/// The file describes an engine run from here, so there is no deployment to
/// hold a token and nothing for this command to do.
fn engine_here(found: &Found) -> String {
    format!(
        "{} names no `[remote]`, so the engine that answers Slack is the one you run here, \
         and it holds its own credential:\n{}\n\n\
         To connect a workspace to a deployment instead, add `[remote]` and `subs apply`.",
        found.path.display(),
        socket_mode_steps()
    )
}

/// The Socket Mode path, which is a Slack app you own rather than one a
/// deployment installs for you.
fn socket_mode_steps() -> String {
    format!(
        "\n  1. Create a Slack app: {SLACK_NEW_APP}\n     \
         the manifest to paste: {SLACK_DOCS}\n  \
         2. export SLACK_APP_TOKEN=xapp-...\n     \
         export SLACK_BOT_TOKEN=xoxb-...\n  \
         3. subs serve --slack-agent <agent>"
    )
}

/// Nothing new arrived. Consent may still be in a browser, and a workspace
/// already connected may have been reinstalled — that refreshes a token and
/// changes nothing anyone can see from here, so it is said rather than left to
/// read as a failure.
fn waiting(before: &[SlackConnection]) {
    println!("No new workspace connected.");
    if before.is_empty() {
        println!("Finish in the browser; the workspace is connected once it lands.");
        return;
    }
    println!("Already connected:");
    for connection in before {
        println!("  {}", connection.label());
    }
    println!();
    println!(
        "Installing again into one of these refreshes its token, which shows as no change here."
    );
}

/// A workspace with nothing routed to it is a bot that never answers, which is
/// the next thing this reader would otherwise discover in Slack.
fn routing_note(found: Option<&Found>) {
    if routed(found) {
        return;
    }
    println!();
    match found {
        Some(found) => println!(
            "{} declares no `[slack]`, so the bot answers nowhere yet. Add one and `subs apply`:",
            found.path.display()
        ),
        None => println!("Nothing routes to the bot yet. Declare it and `subs apply`:"),
    }
    println!();
    println!("  [slack]");
    println!("  dm = \"<agent>\"");
    println!("  mentions = \"<agent>\"");
}

/// Whether anything in the file sends a Slack message to an agent. A file that
/// was never read routes nothing, same as one with no `[slack]`.
fn routed(found: Option<&Found>) -> bool {
    found
        .and_then(|f| f.config.slack.as_ref())
        .is_some_and(|slack| slack.is_configured())
}

/// Poll the org's workspaces until one of them is new, or until there is more
/// value in saying what is there than in waiting longer.
async fn wait_for_workspace(
    ctx: &Context,
    org: &str,
    before: &[SlackConnection],
) -> Result<Option<SlackConnection>> {
    let deadline = std::time::Instant::now() + INSTALL_TIMEOUT;
    loop {
        if let Some(arrived) = arrival(before, &status(ctx, org).await?.connections) {
            return Ok(Some(arrived));
        }
        if std::time::Instant::now() + POLL_INTERVAL > deadline {
            return Ok(None);
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// The workspace this install added, if the org now holds one it did not.
///
/// A connection whose status has changed counts too: a workspace that was
/// revoked and is installed again is the same row coming back to life, and
/// reads as an arrival to the person who just consented.
fn arrival(before: &[SlackConnection], now: &[SlackConnection]) -> Option<SlackConnection> {
    let was: HashMap<&str, &str> = before
        .iter()
        .map(|c| (c.id.as_str(), c.status.as_str()))
        .collect();
    now.iter()
        .find(|c| was.get(c.id.as_str()) != Some(&c.status.as_str()))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::v1::Meta;
    use std::path::PathBuf;

    fn tmpdir() -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("subs-slack-test-{nanos}-{seq}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn found(body: &str) -> Found {
        let path = tmpdir().join(project_config::FILENAME);
        std::fs::write(&path, body).unwrap();
        project_config::load_explicit(&path).unwrap()
    }

    /// A file whose `[slack]` keys have agents to name, which is what the
    /// manifest checks when it is read.
    fn with_agents(slack: &str) -> Found {
        found(&format!(
            "db = \"dev.db\"\n\
             [llm.claude]\ntype = \"anthropic\"\n\
             [agent.support]\nllm = \"claude\"\nmodel = \"m\"\n\
             [agent.oncall]\nllm = \"claude\"\nmodel = \"m\"\n\
             {slack}"
        ))
    }

    fn connection(id: &str, team: &str, status: &str) -> SlackConnection {
        SlackConnection {
            id: id.into(),
            team_id: format!("T{team}"),
            team_name: team.into(),
            status: status.into(),
        }
    }

    /// The file that describes an engine run from here is answered with the
    /// path that machine has, not with a browser.
    #[tokio::test]
    async fn a_file_naming_no_remote_is_told_how_that_side_works() {
        let file = with_agents("[slack]\ndm = \"support\"\n");
        let scope = OrgScope {
            org: None,
            globals: super::super::CloudGlobals {
                config: Some(file.path.clone()),
                // A browser must not open on the way to this error.
                no_interaction: true,
                ..Default::default()
            },
        };

        let err = connect(true, scope).await.unwrap_err().to_string();
        assert!(err.contains("SLACK_APP_TOKEN"), "{err}");
        assert!(err.contains("[remote]"), "{err}");
    }

    /// The deployment decides whether it can do this, and a CLI older than the
    /// deployment must not decide for it.
    #[test]
    fn only_a_deployment_that_advertises_slack_is_asked_for_an_install() {
        let with = Meta {
            features: vec!["agents".into(), "slack".into()],
            ..Default::default()
        };
        assert!(with.has("slack"));
        assert!(!Meta::default().has("slack"));
        assert!(!Meta {
            features: vec!["sessions".into(), "projects".into()],
            ..Default::default()
        }
        .has("slack"));
    }

    /// What the whole wait rests on: the workspace that was not there before.
    #[test]
    fn the_workspace_that_arrived_is_the_one_the_org_did_not_hold() {
        let acme = connection("c1", "Acme", "active");
        let other = connection("c2", "Other", "active");

        assert_eq!(
            arrival(&[], std::slice::from_ref(&acme)).map(|c| c.id),
            Some("c1".to_string())
        );
        // A workspace that was already there is not this install's outcome.
        let held = std::slice::from_ref(&acme);
        assert!(arrival(held, held).is_none());
        assert_eq!(
            arrival(held, &[acme.clone(), other]).map(|c| c.id),
            Some("c2".to_string())
        );
        // Nothing at all is not an arrival either.
        assert!(arrival(&[], &[]).is_none());
    }

    /// A workspace that was revoked and installed again is the same row, so
    /// the status is part of what "new" means.
    #[test]
    fn a_revoked_workspace_installed_again_reads_as_an_arrival() {
        let revoked = connection("c1", "Acme", "revoked");
        let active = connection("c1", "Acme", "active");
        let live = std::slice::from_ref(&active);
        assert_eq!(
            arrival(&[revoked], live).map(|c| c.id),
            Some("c1".to_string())
        );
        assert!(arrival(live, live).is_none());
    }

    /// A connected workspace nothing routes to is a bot that never answers.
    #[test]
    fn a_file_that_routes_nowhere_is_the_case_worth_a_note() {
        let routes = |slack: &str| routed(Some(&with_agents(slack)));
        assert!(!routes(""));
        assert!(!routes("[slack]\n"));
        assert!(routes("[slack]\ndm = \"support\"\n"));
        assert!(routes("[slack.channel.C0ENGOPS]\nagent = \"oncall\"\n"));
        // Nowhere to read a file is nowhere for a message to be routed.
        assert!(!routed(None));
    }

    /// The id is what a reader recognizes when the deployment sends no name.
    #[test]
    fn a_workspace_reads_as_its_name_and_falls_back_to_its_id() {
        assert_eq!(connection("c1", "Acme", "active").label(), "Acme (TAcme)");
        assert_eq!(connection("c1", "", "active").label(), "T");
    }

    /// The engine-here message names the file it read, since `-c` makes that a
    /// real question.
    #[test]
    fn the_local_path_names_the_file_it_read() {
        let file = found("db = \"dev.db\"\n");
        let message = engine_here(&file);
        assert!(
            message.contains(&file.path.display().to_string()),
            "{message}"
        );
        assert!(message.contains("subs serve --slack-agent"), "{message}");
    }
}
