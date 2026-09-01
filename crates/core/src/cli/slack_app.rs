use std::io::{IsTerminal as _, Read as _};

use anyhow::{bail, Context as _, Result};

use serde_json::Value;

use crate::api::v1::{SlackApp, SlackCredentials, SlackManifest};
use crate::manifest::AgentSlackConfig;
use crate::transport::slack::env_var;
use crate::transport::slack::manifest::{render, Delivery};

use super::cloud::context::Context;
use super::cloud::pickers;
use super::cloud::project_config::ProjectConfig;
use super::cloud::{print, ProjectScope};
use super::connections::Row;

pub const SLACK_NEW_APP: &str = "https://api.slack.com/apps";
pub const SLACK_DOCS: &str = "https://substructure.ai/docs/slack";

pub async fn set_credentials(agent_id: String, scope: ProjectScope) -> Result<()> {
    let cfg = super::connections::environment(&scope.globals)?;
    let app = declared(&cfg, &agent_id)?;
    if cfg.remote.is_none() {
        return setup_here(&agent_id, &app);
    }

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
        println!();
        println!("`{agent_id}` has no Slack app yet. Create one from this manifest:");
        print_manifest(&rendered.manifest);
        println!("Install it to your workspace, then paste what it gives you.");
        println!();
        println!("  OAuth & Permissions   → Bot User OAuth Token");
        println!("  Basic Information     → Signing Secret");
        println!();
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
        let app = env_var("SLACK_APP_TOKEN", &agent_id);
        let bot = env_var("SLACK_BOT_TOKEN", &agent_id);
        bail!(
            "This project runs locally and reads its Slack tokens from the environment. \
             Remove them there:\n\n  unset {app}\n  unset {bot}"
        );
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

fn declared(cfg: &ProjectConfig, agent_id: &str) -> Result<AgentSlackConfig> {
    let manifest = cfg.manifest();
    if let Some((_, app)) = manifest
        .slack_apps()
        .into_iter()
        .find(|(id, _)| *id == agent_id)
    {
        return Ok(app.clone());
    }
    bail!(
        "no [agent.{agent_id}.slack] in subs.toml. An agent gets its own Slack app by \
         declaring one:\n\n  [agent.{agent_id}.slack]\n  name = \"…\"\n\nThen `subs apply`."
    )
}

fn print_manifest(rendered: &Value) {
    let body = serde_json::to_string_pretty(rendered).unwrap_or_else(|_| rendered.to_string());
    println!();
    println!("  {SLACK_NEW_APP}  →  Create New App  →  From a manifest");
    println!();
    for line in body.lines() {
        println!("  {line}");
    }
    println!();
}

fn setup_here(agent_id: &str, app: &AgentSlackConfig) -> Result<()> {
    let app_token = env_var("SLACK_APP_TOKEN", agent_id);
    let bot_token = env_var("SLACK_BOT_TOKEN", agent_id);
    if super::env_value(&app_token).is_some() && super::env_value(&bot_token).is_some() {
        println!();
        println!("`{agent_id}` already has its tokens: ${app_token} and ${bot_token} are set.");
        println!();
        println!("To use a different app, export the two again and restart `subs serve`.");
        println!();
        return Ok(());
    }

    println!();
    println!("This project runs locally, so Slack connects over Socket Mode.");
    println!("Create a Slack app for `{agent_id}` from this manifest:");
    print_manifest(&render(agent_id, app, Delivery::Socket));
    println!("Install it to your workspace, then set two variables:");
    println!();
    println!("  export {app_token}=xapp-...");
    println!("  export {bot_token}=xoxb-...");
    println!();
    println!("The app token is under Basic Information → App-Level Tokens. Create one with");
    println!("the connections:write scope. The bot token is under OAuth & Permissions.");
    println!();
    println!("Then run `subs serve`.");
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_agent_with_no_block_is_told_how_to_declare_one() {
        let cfg: ProjectConfig = toml::from_str("[agent.support.slack]\nname = \"Support\"\n")
            .expect("a config declaring one app");
        assert_eq!(
            declared(&cfg, "support").unwrap().name,
            Some("Support".into())
        );

        let err = declared(&cfg, "billing").unwrap_err().to_string();
        assert!(err.contains("[agent.billing.slack]"), "{err}");
    }

    #[test]
    fn what_is_printed_here_is_an_app_slack_dials_out_from() {
        let app = AgentSlackConfig {
            name: Some("Support".into()),
            ..Default::default()
        };
        let m = render("support", &app, Delivery::Socket);
        assert_eq!(
            m["settings"]["socket_mode_enabled"],
            serde_json::json!(true)
        );
        assert!(m["settings"]["event_subscriptions"]["request_url"].is_null());
    }

    #[test]
    fn a_missing_variable_is_named_on_its_own() {
        let said = local_cell("support");
        assert!(said.contains("$SLACK_APP_TOKEN_SUPPORT"), "{said}");
        assert!(said.contains("$SLACK_BOT_TOKEN_SUPPORT"), "{said}");
        assert!(said.ends_with("not set"), "{said}");
    }
}
