use std::io::{IsTerminal as _, Read as _};

use anyhow::{bail, Context as _, Result};

use crate::api::v1::{SlackApp, SlackCredentials, SlackManifest};
use crate::transport::slack::env_var;

use super::cloud::context::Context;
use super::cloud::pickers;
use super::cloud::project_config::ProjectConfig;
use super::cloud::{print, ProjectScope};
use super::connections::Row;

pub const SLACK_NEW_APP: &str = "https://api.slack.com/apps";
pub const SLACK_DOCS: &str = "https://substructure.ai/docs/slack";

pub fn path_agent(path: &str) -> Option<&str> {
    path.strip_prefix("agent.")?.strip_suffix(".slack")
}

pub async fn set_credentials(agent_id: String, scope: ProjectScope) -> Result<()> {
    let cfg = super::connections::environment(&scope.globals)?;
    if cfg.remote.is_none() {
        bail!("{}", engine_here(&agent_id));
    }
    declared(&cfg, &agent_id)?;

    let (ctx, project) = Context::from_project(&scope).await?;
    let existing: Vec<SlackApp> = ctx
        .client
        .get(&format!("/api/v1/projects/{project}/slack"))
        .await?;
    let live = existing
        .iter()
        .find(|app| app.agent_id == agent_id)
        .is_some_and(|app| app.installed.is_some());

    if !live {
        let rendered: SlackManifest = ctx
            .client
            .get(&format!(
                "/api/v1/projects/{project}/slack/{agent_id}/manifest"
            ))
            .await?;
        print_manifest(&agent_id, &rendered);
    } else {
        println!();
        println!("Replacing the credentials for `{agent_id}`.");
        println!();
    }

    let (bot_token, signing_secret) = read_pair(&scope)?;
    let app: SlackApp = ctx
        .client
        .put_json(
            &format!("/api/v1/projects/{project}/slack/{agent_id}"),
            &SlackCredentials {
                bot_token,
                signing_secret,
            },
        )
        .await?;

    if scope.globals.json {
        return print::json(&app);
    }
    let team = app
        .installed
        .as_ref()
        .map(|i| i.team_name.clone())
        .unwrap_or_default();
    println!();
    println!("Installed {} into {team}.", app.name);
    if !live {
        println!("It answers in its DMs, and in any channel you invite it to.");
        println!();
        println!("  /invite @{}", app.name);
    }
    Ok(())
}

pub async fn delete_credentials(agent_id: String, scope: ProjectScope) -> Result<()> {
    let cfg = super::connections::environment(&scope.globals)?;
    if cfg.remote.is_none() {
        bail!("{}", engine_here(&agent_id));
    }
    let (ctx, project) = Context::from_project(&scope).await?;
    ctx.client
        .delete_discard(&format!("/api/v1/projects/{project}/slack/{agent_id}"))
        .await?;
    println!("Removed the Slack app for `{agent_id}`.");
    Ok(())
}

pub async fn rows(cfg: &ProjectConfig, scope: &ProjectScope, here: bool) -> Result<Vec<Row>> {
    let manifest = cfg.manifest();
    let declared = manifest.slack_apps();
    if declared.is_empty() {
        return Ok(Vec::new());
    }
    if here {
        return Ok(declared
            .iter()
            .map(|(agent_id, _)| Row {
                path: format!("agent.{agent_id}.slack"),
                what: "slack app".to_string(),
                credential: local_cell(agent_id),
            })
            .collect());
    }

    let (ctx, project) = Context::from_project(scope).await?;
    let apps: Vec<SlackApp> = ctx
        .client
        .get(&format!("/api/v1/projects/{project}/slack"))
        .await?;
    Ok(apps
        .iter()
        .map(|app| Row {
            path: format!("agent.{}.slack", app.agent_id),
            what: "slack app".to_string(),
            credential: app.label(),
        })
        .collect())
}

fn read_pair(scope: &ProjectScope) -> Result<(String, String)> {
    if !std::io::stdin().is_terminal() {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("reading from stdin")?;
        let mut lines = buf.lines().map(str::trim).filter(|l| !l.is_empty());
        let (Some(bot), Some(secret)) = (lines.next(), lines.next()) else {
            bail!(
                "a Slack app takes two secrets, so stdin needs two lines: the bot token, then \
                 the signing secret.\n  printf '%s\\n%s\\n' \"$BOT_TOKEN\" \"$SIGNING_SECRET\" | \
                 subs auth …"
            );
        };
        return Ok((bot.to_string(), secret.to_string()));
    }
    if !pickers::interactive(&scope.globals) {
        bail!("nothing on stdin. Pipe the bot token and the signing secret, one per line.");
    }
    let bot = pickers::prompt_secret("Bot token (xoxb-)")?;
    let secret = pickers::prompt_secret("Signing secret")?;
    if bot.is_empty() || secret.is_empty() {
        bail!("both a bot token and a signing secret are needed.");
    }
    Ok((bot, secret))
}

fn local_cell(agent_id: &str) -> String {
    let missing: Vec<String> = ["SLACK_APP_TOKEN", "SLACK_BOT_TOKEN"]
        .into_iter()
        .map(|prefix| env_var(prefix, agent_id))
        .filter(|var| super::env_value(var).is_none())
        .map(|var| format!("${var}"))
        .collect();
    match missing.is_empty() {
        true => "set".to_string(),
        false => format!("{} not set", missing.join(" and ")),
    }
}

fn declared(cfg: &ProjectConfig, agent_id: &str) -> Result<()> {
    let manifest = cfg.manifest();
    if manifest.slack_apps().iter().any(|(id, _)| *id == agent_id) {
        return Ok(());
    }
    bail!(
        "no [agent.{agent_id}.slack] in subs.toml. An agent gets its own Slack app by \
         declaring one:\n\n  [agent.{agent_id}.slack]\n  name = \"…\"\n\nThen `subs apply`."
    )
}

fn print_manifest(agent_id: &str, rendered: &SlackManifest) {
    let body = serde_json::to_string_pretty(&rendered.manifest)
        .unwrap_or_else(|_| rendered.manifest.to_string());
    println!();
    println!("`{agent_id}` has no Slack app yet. Create one from this manifest:");
    println!();
    println!("  {SLACK_NEW_APP}  →  Create New App  →  From a manifest");
    println!();
    for line in body.lines() {
        println!("  {line}");
    }
    println!();
    println!("Then Install to Workspace, and paste what it gives you.");
    println!();
    println!("  OAuth & Permissions   → Bot User OAuth Token");
    println!("  Basic Information     → Signing Secret");
    println!();
}

fn engine_here(agent_id: &str) -> String {
    let app = env_var("SLACK_APP_TOKEN", agent_id);
    let bot = env_var("SLACK_BOT_TOKEN", agent_id);
    format!(
        "an engine here answers Slack over Socket Mode, so there is no install to credential. \
         It reads its tokens from the environment:\n\n  \
         1. Create a Slack app: {SLACK_NEW_APP}\n     \
         the manifest to paste: {SLACK_DOCS}\n  \
         2. export {app}=xapp-...\n     \
         export {bot}=xoxb-...\n  \
         3. subs serve\n\n\
         To install a per-agent app on a deployment instead, add `[remote]` and `subs apply`."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_agents_slack_block_is_this_path() {
        assert_eq!(path_agent("agent.support.slack"), Some("support"));
        assert_eq!(path_agent("agent.support"), None);
        assert_eq!(path_agent("mcp.sentry"), None);
        assert_eq!(path_agent("llm.openrouter"), None);
    }

    #[test]
    fn the_local_refusal_names_this_agents_variables() {
        let said = engine_here("support");
        assert!(said.contains("SLACK_APP_TOKEN_SUPPORT"), "{said}");
        assert!(said.contains("SLACK_BOT_TOKEN_SUPPORT"), "{said}");
    }

    #[test]
    fn a_missing_variable_is_named_on_its_own() {
        let said = local_cell("support");
        assert!(said.contains("$SLACK_APP_TOKEN_SUPPORT"), "{said}");
        assert!(said.contains("$SLACK_BOT_TOKEN_SUPPORT"), "{said}");
        assert!(said.ends_with("not set"), "{said}");
    }
}
