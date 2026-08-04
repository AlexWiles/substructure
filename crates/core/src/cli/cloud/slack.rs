//! `subs slack connect`: putting a workspace's bot token where the bot runs.
//!
//! Consent is a human in a browser either way; what differs is who is left
//! holding the token, and the file says which. A file naming a `[remote]`
//! asks that server to run Slack's install flow, and the token never touches
//! this machine. An engine you run here has no install flow to offer — Socket
//! Mode reads its two tokens out of the environment — so that side is
//! explained rather than pretended.
//!
//! An org holds one workspace: a Slack app installs once per workspace, so
//! connecting again is a refresh, and connecting a different workspace is a
//! replacement.

use std::time::Duration;

use anyhow::{bail, Result};
use clap::Subcommand;
use reqwest::StatusCode;
use serde::Serialize;

use crate::api::v1::{Meta, SlackInstall, SlackInstallStarted, SlackWorkspace};

use super::context::Context;
use super::project_config::{self, Found};
use super::{http, print, OrgScope};

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
    status: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace: Option<&'a SlackWorkspace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    replaced: Option<&'a SlackWorkspace>,
}

async fn connect(no_browser: bool, scope: OrgScope) -> Result<()> {
    let found = project_config::resolve(scope.globals.config.as_deref())?;
    // A file that names no deployment is an engine run from here, which holds
    // its own credential. No file at all names no engine either: the command
    // means the server the flags and the default resolve to.
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

    let started: SlackInstallStarted = ctx
        .client
        .post_empty(&format!("/api/v1/orgs/{org}/slack/install"))
        .await?;

    // The URL is the one thing the reader cannot do without, so `--json` puts
    // it where it does not spoil the document on stdout.
    let say = |line: String| match scope.globals.json {
        true => eprintln!("{line}"),
        false => println!("{line}"),
    };
    if let Some(scopes) = &started.scopes {
        say(format!("Requesting: {scopes}"));
    }
    say(format!(
        "Open this URL to add the bot to your workspace:\n  {}",
        started.install_url
    ));
    if !no_browser && webbrowser::open(&started.install_url).is_ok() {
        say("Opened your browser to install.".into());
    }

    let install = wait_for_install(&ctx, &org, &started.id).await?;

    if scope.globals.json {
        return print::json(&Connected {
            org_id: &org,
            status: install.as_ref().map_or("pending", |i| i.status.as_str()),
            workspace: install.as_ref().and_then(|i| i.workspace.as_ref()),
            replaced: install.as_ref().and_then(|i| i.replaced.as_ref()),
        });
    }

    let Some(install) = install else {
        println!(
            "Still waiting on consent. Finish in the browser; the workspace is \
             connected once it lands."
        );
        return Ok(());
    };
    match &install.workspace {
        Some(workspace) => println!("Connected {} to {org}.", workspace.label()),
        None => println!("Connected to {org}."),
    }
    if let Some(replaced) = &install.replaced {
        println!("Replaced the connection to {}.", replaced.label());
    }
    routing_note(found.as_ref());
    Ok(())
}

/// A deployment that does not install Slack apps says so before a browser is
/// opened, rather than 404-ing halfway.
///
/// The gate is the feature, not its absence: a deployment that advertises
/// nothing is one this CLI predates, and there is no older shape to fall back
/// to. A request that failed says nothing about what the deployment offers, so
/// it is reported as itself.
async fn require_slack(ctx: &Context) -> Result<()> {
    let meta: Meta = match ctx.client.get("/api/v1/meta").await {
        Ok(meta) => meta,
        Err(e) if http::status_of(&e) == Some(StatusCode::NOT_FOUND) => Default::default(),
        Err(e) => return Err(e),
    };
    if meta.has("slack") {
        return Ok(());
    }
    bail!(
        "this deployment does not install Slack apps, so it has no workspace to connect.\n\
         An engine you run answers Slack over Socket Mode instead, with its own credential:\n\
         {}",
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

/// Poll until Slack has sent the deployment a token, or until there is more
/// value in saying where to finish than in waiting longer.
///
/// A status this build has no name for keeps the wait going: a deployment that
/// learns a new step of its own must not read as an outcome.
async fn wait_for_install(ctx: &Context, org: &str, id: &str) -> Result<Option<SlackInstall>> {
    let deadline = std::time::Instant::now() + INSTALL_TIMEOUT;
    loop {
        let install: SlackInstall = ctx
            .client
            .get(&format!("/api/v1/orgs/{org}/slack/installs/{id}"))
            .await?;
        match install.status.as_str() {
            "active" => return Ok(Some(install)),
            "denied" => bail!("the install was declined in Slack{}", detail(&install)),
            "expired" => bail!(
                "the install expired before it was approved{}. Run `subs slack connect` again.",
                detail(&install)
            ),
            _ => {}
        }
        if std::time::Instant::now() + POLL_INTERVAL > deadline {
            return Ok(None);
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// What the deployment said about a failure, when it said anything.
fn detail(install: &SlackInstall) -> String {
    match install
        .error
        .as_deref()
        .map(str::trim)
        .filter(|e| !e.is_empty())
    {
        Some(error) => format!(": {error}"),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// The file that describes an engine run from here is answered with the
    /// path that machine has, not with a browser.
    #[tokio::test]
    async fn a_file_naming_no_deployment_is_told_how_that_side_works() {
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

    /// A failure the deployment explained reads as that explanation; one it
    /// did not is still a whole sentence.
    #[test]
    fn a_declined_install_says_what_the_deployment_said() {
        let install = |error: Option<&str>| SlackInstall {
            id: "i1".into(),
            status: "denied".into(),
            error: error.map(str::to_string),
            ..Default::default()
        };
        assert_eq!(
            detail(&install(Some("user is not an admin"))),
            ": user is not an admin"
        );
        assert_eq!(detail(&install(None)), "");
        assert_eq!(detail(&install(Some("  "))), "");
    }

    /// The id is what a reader recognizes when the deployment sends no name.
    #[test]
    fn a_workspace_reads_as_its_name_and_falls_back_to_its_id() {
        let workspace = |name: &str| SlackWorkspace {
            team_id: "T0123".into(),
            team_name: name.into(),
            ..Default::default()
        };
        assert_eq!(workspace("Acme").label(), "Acme (T0123)");
        assert_eq!(workspace("").label(), "T0123");
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
