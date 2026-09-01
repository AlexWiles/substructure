use anyhow::{Context as _, Result};

use crate::api::v1::{Notice, NoticesResponse};
use crate::cli::connections::Needs;

use super::cloud::context::Context as CloudContext;
use super::cloud::project_config::{self, Found};
use super::cloud::{notices, print, ProjectScope};
use super::slack_app;
use super::target::target;
use super::{connections, env_value};

const SLACK_TOKENS: [(&str, &str); 2] = [
    ("SLACK_APP_TOKEN", "xapp-..."),
    ("SLACK_BOT_TOKEN", "xoxb-..."),
];

pub async fn run(scope: ProjectScope) -> Result<()> {
    let found = project_config::resolve(scope.globals.config.as_deref())?
        .context("no subs.toml found. Write one, or pass -c.")?;

    let notices = match target(&scope.globals)?.here().is_some() {
        true => here(&found, env_value).await?,
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

async fn deployed(scope: &ProjectScope) -> Result<Vec<Notice>> {
    let (ctx, project) = CloudContext::from_project(scope).await?;
    notices::fetch(&ctx, &project).await?.context(
        "this deployment does not report what a project still needs. \
         Upgrade it, or use the CLI it shipped with.",
    )
}

async fn here(found: &Found, env: impl Fn(&str) -> Option<String>) -> Result<Vec<Notice>> {
    let config = &found.config;
    let mut out = Vec::new();

    for block in config.provider_bindings() {
        if env(&block.api_key_env).is_some() {
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

    for (path, needs) in connections::unauthorized_local(config).await? {
        out.push(match needs {
            Needs::Token => Notice::action(format!("Set the token for [{path}]"))
                .with_command(format!("subs auth {path}")),
            Needs::Login => Notice::action(format!("Authorize the [{path}] connection"))
                .with_command(format!("subs auth {path}")),
            Needs::Declaration => Notice::action(format!(
                "[{path}] wants a credential it publishes no way to get; declare \
                 `auth = \"token\"` and set one"
            ))
            .with_command(format!("subs auth {path}")),
        });
    }

    for (agent_id, _) in config.manifest().slack_apps() {
        for (prefix, example) in SLACK_TOKENS {
            let var = crate::transport::slack::env_var(prefix, agent_id);
            if env(&var).is_some() {
                continue;
            }
            out.push(
                Notice::action(format!(
                    "Set ${var}, which [agent.{agent_id}] answers Slack with"
                ))
                .with_command(format!("export {var}={example}"))
                .with_url(slack_app::SLACK_DOCS),
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

    fn empty(_: &str) -> Option<String> {
        None
    }

    #[tokio::test]
    async fn an_engine_here_reports_the_variables_that_hold_nothing() {
        let file = found(
            "[llm.claude]\ntype = \"anthropic\"\napi_key_env = \"NOT_SET_ANTHROPIC\"\n\
             [worker.main]\nurl = \"https://w.test\"\n\
             [agent.support]\nllm = \"claude\"\nmodel = \"m\"\nworker = \"main\"\n\
             [mcp.github]\nurl = \"https://api.github.test/mcp\"\nauth = \"token\"\n\
             [mcp.linear]\nurl = \"https://mcp.linear.app/mcp\"\n\
             [agent.support.slack]\n",
        );

        let notices = here(&file, empty).await.unwrap();
        let said = messages(&notices).join("\n");
        assert!(said.contains("$NOT_SET_ANTHROPIC"), "{said}");
        assert!(
            said.contains("console.anthropic.com"),
            "the key is issued somewhere: {said}"
        );
        assert!(said.contains("Set the token for [mcp.github]"), "{said}");
        assert!(
            said.contains("Authorize the [mcp.linear] connection"),
            "{said}"
        );
        assert!(said.contains("$SLACK_APP_TOKEN_SUPPORT"), "{said}");
        assert!(said.contains("$SLACK_BOT_TOKEN_SUPPORT"), "{said}");

        let command = |needle: &str| {
            notices
                .iter()
                .find(|n| n.message.contains(needle))
                .unwrap()
                .command
                .clone()
        };
        assert_eq!(
            command("[mcp.linear]").as_deref(),
            Some("subs auth mcp.linear")
        );
        assert_eq!(
            command("[mcp.github]").as_deref(),
            Some("subs auth mcp.github")
        );
    }

    #[tokio::test]
    async fn a_project_with_everything_it_needs_says_so() {
        let file = found(
            "[llm.byo]\ntype = \"worker\"\n\
             [worker.w]\nurl = \"https://w.test\"\n\
             [agent.support]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"w\"\n",
        );
        assert!(here(&file, empty).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn slack_tokens_are_asked_for_only_by_an_agent_with_its_own_app() {
        let file = found(
            "[llm.byo]\ntype = \"worker\"\n\
             [worker.w]\nurl = \"https://w.test\"\n\
             [agent.support]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"w\"\n",
        );
        assert!(here(&file, empty).await.unwrap().is_empty());
    }
}
