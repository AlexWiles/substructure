//! `subs doctor`: what this project still needs before it works.
//!
//! Two sources, one list. A file naming a `[remote]` asks the deployment, which
//! holds the keys, the consents and the workspaces and is therefore the only
//! thing that knows what is missing. A file naming none describes an engine you
//! run here, whose credentials are this machine's environment and this machine's
//! database — so the same notices are read off both, and printed by the same
//! code that prints them after `subs apply`.

use anyhow::{Context as _, Result};

use crate::api::v1::{Notice, NoticesResponse};

use super::cloud::context::Context as CloudContext;
use super::cloud::project_config::{self, Found};
use super::cloud::{notices, print, slack, ProjectScope};
use super::{env_value, mcp};

/// The two variables a bot answering over Socket Mode reads, with an example of
/// what each holds.
const SLACK_TOKENS: [(&str, &str); 2] = [
    ("SLACK_APP_TOKEN", "xapp-..."),
    ("SLACK_BOT_TOKEN", "xoxb-..."),
];

pub async fn run(scope: ProjectScope) -> Result<()> {
    let found = project_config::resolve(scope.globals.config.as_deref())?
        .context("no substructure.toml found. Write one with `subs init`, or pass -c.")?;

    let notices = match found.config.remote.is_none() {
        true => here(&found).await?,
        false => deployed(&scope).await?,
    };
    report(&notices, &scope)
}

fn report(notices: &[Notice], scope: &ProjectScope) -> Result<()> {
    if scope.globals.json {
        return print::json(&NoticesResponse {
            notices: notices.to_vec(),
        });
    }
    if notices.is_empty() {
        println!("Nothing to do.");
        return Ok(());
    }
    notices::print(notices, &notices::flag(&scope.globals));
    Ok(())
}

/// What the deployment says. One that does not report it is older than this
/// CLI, and there is no earlier shape to read instead.
async fn deployed(scope: &ProjectScope) -> Result<Vec<Notice>> {
    let (ctx, project) = CloudContext::from_project(scope).await?;
    notices::fetch(&ctx, &project).await?.context(
        "this deployment does not report what a project still needs. \
         Upgrade it, or use the CLI it shipped with.",
    )
}

/// What an engine you run here still needs: the credentials it reads from this
/// machine's environment, and the consents in its own database.
async fn here(found: &Found) -> Result<Vec<Notice>> {
    let config = &found.config;
    let mut out = Vec::new();

    // A block the engine calls itself, whose key is a variable that holds
    // nothing. Where the key is issued is part of the sentence: a reader who
    // has to go and get one should not have to go and find that too.
    for block in config.provider_bindings() {
        if env_value(&block.api_key_env).is_some() {
            continue;
        }
        let issued = block
            .kind
            .console_url()
            .map(|url| format!(" ({url})"))
            .unwrap_or_default();
        out.push(
            Notice::action(format!(
                "Set ${}, the key for [llm.{}]{issued}",
                block.api_key_env, block.name
            ))
            .with_command(format!("export {}=...", block.api_key_env)),
        );
    }

    for (id, spec) in config.connections() {
        let Some(auth) = &spec.auth else { continue };
        if env_value(&auth.token_env).is_none() {
            out.push(
                Notice::action(format!(
                    "Set ${}, the credential for [mcp.{id}]",
                    auth.token_env
                ))
                .with_command(format!("export {}=...", auth.token_env)),
            );
        }
    }
    for id in mcp::unauthorized_local(config).await? {
        out.push(
            Notice::action(format!("Authorize the [mcp.{id}] connection"))
                .with_command(format!("subs mcp login {id}")),
        );
    }

    // Socket Mode is the engine's own Slack app, so its two tokens are this
    // machine's to hold. Nothing routes anywhere without them.
    if config.slack.as_ref().is_some_and(|s| s.is_configured()) {
        for (var, example) in SLACK_TOKENS {
            if env_value(var).is_some() {
                continue;
            }
            out.push(
                Notice::action(format!("Set ${var}, which the bot answers Slack with"))
                    .with_command(format!("export {var}={example}"))
                    .with_url(slack::SLACK_DOCS),
            );
        }
    }

    // A named secret that holds nothing sends decisions unsigned, which the
    // worker on the other end is entitled to refuse.
    for (id, section) in &config.agent {
        let Some(var) = &section.signing_secret_env else {
            continue;
        };
        if env_value(var).is_none() {
            out.push(
                Notice::action(format!(
                    "Set ${var}, which signs the decisions [agent.{id}] sends its worker"
                ))
                .with_command(format!("export {var}=...")),
            );
        }
    }

    Ok(out)
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
        let dir = std::env::temp_dir().join(format!("subs-doctor-test-{nanos}-{seq}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn found(body: &str) -> Found {
        let dir = tmpdir();
        let path = dir.join(project_config::FILENAME);
        std::fs::write(&path, format!("db = {:?}\n{body}", dir.join("t.db"))).unwrap();
        project_config::load_explicit(&path).unwrap()
    }

    fn messages(notices: &[Notice]) -> Vec<&str> {
        notices.iter().map(|n| n.message.as_str()).collect()
    }

    /// Every credential an engine here reads, named with the variable it is
    /// missing from — the whole point of running this before `subs serve`.
    #[tokio::test]
    async fn an_engine_here_reports_the_variables_that_hold_nothing() {
        let file = found(
            "[llm.claude]\ntype = \"anthropic\"\napi_key_env = \"NOT_SET_ANTHROPIC\"\n\
             [agent.support]\nllm = \"claude\"\nmodel = \"m\"\n\
             worker = \"https://w.test\"\nsigning_secret_env = \"NOT_SET_SECRET\"\n\
             [mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\n\
             auth = { token_env = \"NOT_SET_SENTRY\" }\n\
             [mcp.linear]\nurl = \"https://mcp.linear.app/mcp\"\n\
             [slack]\ndm = \"support\"\n",
        );

        let notices = here(&file).await.unwrap();
        let said = messages(&notices).join("\n");
        assert!(said.contains("$NOT_SET_ANTHROPIC"), "{said}");
        assert!(
            said.contains("console.anthropic.com"),
            "the key is issued somewhere: {said}"
        );
        assert!(said.contains("$NOT_SET_SENTRY"), "{said}");
        assert!(
            said.contains("Authorize the [mcp.linear] connection"),
            "{said}"
        );
        assert!(said.contains("$SLACK_APP_TOKEN"), "{said}");
        assert!(said.contains("$SLACK_BOT_TOKEN"), "{said}");
        assert!(said.contains("$NOT_SET_SECRET"), "{said}");

        // A connection the engine authorizes is done from the command line;
        // a variable is not something a command can set.
        let authorize = notices
            .iter()
            .find(|n| n.message.contains("[mcp.linear]"))
            .unwrap();
        assert_eq!(authorize.command.as_deref(), Some("subs mcp login linear"));
    }

    /// A file that declares nothing to hold a credential for needs nothing.
    #[tokio::test]
    async fn a_project_with_everything_it_needs_says_so() {
        let file = found(
            "[llm.byo]\ntype = \"worker\"\n\
             [agent.support]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"https://w.test\"\n",
        );
        assert!(here(&file).await.unwrap().is_empty());
    }

    /// Slack tokens are only this machine's business when something routes to
    /// the bot: an empty `[slack]` asks for nothing.
    #[tokio::test]
    async fn slack_tokens_are_asked_for_only_when_the_bot_answers_somewhere() {
        let file = found(
            "[llm.byo]\ntype = \"worker\"\n\
             [agent.support]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"https://w.test\"\n[slack]\n",
        );
        assert!(here(&file).await.unwrap().is_empty());
    }
}
