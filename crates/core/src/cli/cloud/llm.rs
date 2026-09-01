use anyhow::{bail, Result};
use serde::Serialize;

use crate::api::v1::{LlmBlockView, LlmKeyRequest};
use crate::cli::env::ProviderKind;
use crate::cli::env_value;
use crate::cli::target::target;

use super::context::Context;
use super::notices;
use super::pickers;
use super::print;
use super::project_config::ProjectConfig;
use super::ProjectScope;
use crate::cli::connections::Row;

#[derive(Debug, Serialize)]
struct KeyResult<'a> {
    block: &'a str,
    key_bound: bool,
}

pub(crate) async fn set_key_at(block: String, scope: ProjectScope) -> Result<()> {
    match target(&scope.globals)?.here() {
        Some(config) => key_is_a_variable(&block, &config, "set"),
        None => set_key(block, scope).await,
    }
}

pub(crate) async fn delete_key_at(block: String, scope: ProjectScope) -> Result<()> {
    match target(&scope.globals)?.here() {
        Some(config) => key_is_a_variable(&block, &config, "delete"),
        None => delete_key(block, scope).await,
    }
}

pub(crate) async fn rows(
    cfg: &ProjectConfig,
    scope: &ProjectScope,
    here: bool,
) -> Result<Vec<Row>> {
    if cfg.llm.is_empty() {
        return Ok(Vec::new());
    }
    if here {
        return Ok(cfg
            .llm
            .keys()
            .filter_map(|name| declared(name, cfg))
            .collect());
    }

    let (ctx, project) = Context::from_project(scope).await?;
    let blocks: Vec<LlmBlockView> = ctx
        .client
        .get(&format!("/api/v1/projects/{project}/llm"))
        .await?;
    Ok(blocks
        .iter()
        .map(|b| Row {
            path: format!("llm.{}", b.name),
            what: b.kind.clone(),
            credential: key_cell(&b.kind, b.key_bound).to_string(),
        })
        .collect())
}

const WORKER_RUNS_IT: &str = "n/a (your worker holds it)";

fn key_cell(kind: &str, bound: bool) -> &'static str {
    match (kind, bound) {
        ("worker", _) => WORKER_RUNS_IT,
        (_, true) => "set",
        (_, false) => "not set",
    }
}

fn declared(name: &str, config: &ProjectConfig) -> Option<Row> {
    let spec = config.llm.get(name)?;
    let credential = match spec.kind {
        ProviderKind::Worker => WORKER_RUNS_IT.to_string(),
        _ => match spec.api_key_env() {
            None => "no variable named".to_string(),
            Some(var) => match env_value(&var).is_some() {
                true => format!("${var}"),
                false => format!("${var} (not set)"),
            },
        },
    };
    Some(Row {
        path: format!("llm.{name}"),
        what: spec.kind.name().to_string(),
        credential,
    })
}

fn key_is_a_variable(block: &str, config: &ProjectConfig, verb: &str) -> Result<()> {
    let Some(spec) = config.llm.get(block) else {
        bail!(
            "no [llm.{block}] in subs.toml. Declared: {}",
            crate::copy::declared(config.llm.keys())
        );
    };
    if spec.kind == ProviderKind::Worker {
        bail!(
            "[llm.{block}] does not use a key. Your own worker makes the model calls, so it \
             holds the key."
        );
    }
    let var = spec
        .api_key_env()
        .unwrap_or_else(|| "the block's variable".to_string());
    let how = match verb {
        "delete" => format!("Remove it there:\n\n  unset {var}"),
        _ => format!("Set it there:\n\n  export {var}=..."),
    };
    bail!("This project runs locally and reads the key from the environment. {how}")
}

async fn set_key(block: String, scope: ProjectScope) -> Result<()> {
    let key = pickers::read_secret(&scope.globals, "Paste the key")?;
    if key.is_empty() {
        bail!("no key given. Pipe it in.");
    }

    let (ctx, project) = Context::from_project(&scope).await?;
    ctx.client
        .put_discard(
            &format!("/api/v1/projects/{project}/llm/{block}/key"),
            &LlmKeyRequest { key },
        )
        .await?;

    if scope.globals.json {
        return print::json(&KeyResult {
            block: &block,
            key_bound: true,
        });
    }
    println!("Key set for [llm.{block}].");
    notices::remaining(&ctx, &project, &scope.globals).await;
    Ok(())
}

async fn delete_key(block: String, scope: ProjectScope) -> Result<()> {
    let (ctx, project) = Context::from_project(&scope).await?;
    ctx.client
        .delete_discard(&format!("/api/v1/projects/{project}/llm/{block}/key"))
        .await?;

    if scope.globals.json {
        return print::json(&KeyResult {
            block: &block,
            key_bound: false,
        });
    }
    println!("Key removed for [llm.{block}]. Calls on it will fail until another is set.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(body: &str) -> ProjectConfig {
        toml::from_str(body).unwrap()
    }

    const BLOCKS: &str =
        "[llm.claude]\ntype = \"anthropic\"\napi_key_env = \"NOT_SET_ANTHROPIC\"\n\
         [llm.byo]\ntype = \"worker\"\n\
         [worker.w]\nurl = \"https://w.test\"\n\
         [agent.a]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"w\"\n";

    #[test]
    fn a_block_here_reports_the_variable_it_reads() {
        let config = config(BLOCKS);
        let row = declared("claude", &config).unwrap();
        let (kind, key) = (row.what, row.credential);
        assert_eq!(row.path, "llm.claude");
        assert_eq!(kind, "anthropic");
        assert_eq!(key, "$NOT_SET_ANTHROPIC (not set)");
    }

    #[test]
    fn a_worker_block_needs_no_key_here() {
        let config = config(BLOCKS);
        let row = declared("byo", &config).unwrap();
        let (kind, key) = (row.what, row.credential);
        assert_eq!(kind, "worker");
        assert!(key.contains("your worker holds it"), "{key}");
    }

    #[test]
    fn setting_a_key_here_names_the_variable() {
        let config = config(BLOCKS);
        let err = key_is_a_variable("claude", &config, "set")
            .unwrap_err()
            .to_string();
        assert!(err.contains("export NOT_SET_ANTHROPIC=..."), "{err}");

        let err = key_is_a_variable("claude", &config, "delete")
            .unwrap_err()
            .to_string();
        assert!(err.contains("unset NOT_SET_ANTHROPIC"), "{err}");

        let err = key_is_a_variable("byo", &config, "set")
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not use a key"), "{err}");
    }
}
