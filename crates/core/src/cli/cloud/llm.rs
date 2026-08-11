//! `subs llm`: the keys behind the `[llm.*]` blocks the file declares.
//!
//! A block is declared in `substructure.toml`; the key for it is uploaded here.
//! Pasting a key is configuration rather than money or consent, so it belongs
//! on this side of the write partition — and it never appears in argv, where a
//! shell history would keep it.
//!
//! An engine here holds no key. The file *names* the variable each block reads,
//! and this machine's environment holds it, so `list` reports which of those
//! variables are set and the two binding commands say which one to export.

use anyhow::{bail, Context as _, Result};
use clap::Subcommand;
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

#[derive(Subcommand)]
pub enum LlmCommand {
    /// List the blocks this project declares and whether each has a key.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Upload the key for one block. Typed at a prompt, piped in on stdin, or
    /// read from the environment variable `--env` names.
    SetKey {
        block: String,
        /// Read the key from this environment variable instead of stdin.
        #[arg(long, value_name = "VAR")]
        env: Option<String>,
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Remove the key for one block. Calls on it fail until another is set.
    DeleteKey {
        block: String,
        #[command(flatten)]
        scope: ProjectScope,
    },
}

#[derive(Debug, Serialize)]
struct KeyResult<'a> {
    block: &'a str,
    key_bound: bool,
}

pub async fn run(command: LlmCommand) -> Result<()> {
    match command {
        LlmCommand::List { scope } => match target(&scope.globals)?.here() {
            Some(config) => list_here(&config, &scope),
            None => list(scope).await,
        },
        LlmCommand::SetKey { block, env, scope } => match target(&scope.globals)?.here() {
            Some(config) => key_is_a_variable(&block, &config, "set"),
            None => set_key(block, env, scope).await,
        },
        LlmCommand::DeleteKey { block, scope } => match target(&scope.globals)?.here() {
            Some(config) => key_is_a_variable(&block, &config, "delete"),
            None => delete_key(block, scope).await,
        },
    }
}

// ---------------------------------------------------------------------------
// An engine here: the file declares the blocks, and this machine's environment
// holds their keys.
// ---------------------------------------------------------------------------

/// One declared block, as this machine can answer for it: the name, the venue,
/// and whether the variable it reads holds anything.
fn declared(name: &str, config: &ProjectConfig) -> Option<[String; 3]> {
    let spec = config.llm.get(name)?;
    // A worker block runs the call itself, so there is no key here to bind and
    // an empty cell would read as one that is missing.
    let key = match spec.kind {
        ProviderKind::Worker => "n/a (your worker runs it)".to_string(),
        _ => match spec.api_key_env() {
            None => "no variable named".to_string(),
            Some(var) => match env_value(&var).is_some() {
                true => format!("${var}"),
                false => format!("${var} (not set)"),
            },
        },
    };
    Some([name.to_string(), spec.kind.name().to_string(), key])
}

fn list_here(config: &ProjectConfig, scope: &ProjectScope) -> Result<()> {
    let rows: Vec<[String; 3]> = config
        .llm
        .keys()
        .filter_map(|name| declared(name, config))
        .collect();

    if scope.globals.json {
        let blocks: Vec<serde_json::Value> = rows
            .iter()
            .map(|[name, kind, key]| serde_json::json!({"name": name, "type": kind, "key": key}))
            .collect();
        return print::json(&blocks);
    }

    let columns = [
        print::Column::left("NAME"),
        print::Column::left("TYPE"),
        print::Column::left("KEY"),
    ];
    let rows: Vec<Vec<String>> = rows.iter().map(|r| r.to_vec()).collect();
    print::table(&columns, &rows);
    Ok(())
}

/// A key here is a variable this machine holds, so there is nothing to upload
/// or remove. The command says which variable, which is the thing you actually
/// needed to know.
fn key_is_a_variable(block: &str, config: &ProjectConfig, verb: &str) -> Result<()> {
    let Some(spec) = config.llm.get(block) else {
        bail!(
            "no [llm.{block}] in substructure.toml. Declared: {}",
            crate::worker::directory::declared(&config.llm.keys().cloned().collect::<Vec<_>>())
        );
    };
    if spec.kind == ProviderKind::Worker {
        bail!("[llm.{block}] is a worker block: your worker runs its calls and holds its key.");
    }
    let var = spec
        .api_key_env()
        .unwrap_or_else(|| "the block's variable".to_string());
    match verb {
        "delete" => bail!(
            "an engine here reads the key for [llm.{block}] from ${var}, so there is none to \
             delete.\n  unset {var}"
        ),
        _ => bail!(
            "an engine here reads the key for [llm.{block}] from ${var}, so there is none to \
             set.\n  export {var}=..."
        ),
    }
}

async fn list(scope: ProjectScope) -> Result<()> {
    let (ctx, project) = Context::from_project(&scope).await?;
    let blocks: Vec<LlmBlockView> = ctx
        .client
        .get(&format!("/api/v1/projects/{project}/llm"))
        .await?;

    if scope.globals.json {
        return print::json(&blocks);
    }

    let columns = [
        print::Column::left("NAME"),
        print::Column::left("TYPE"),
        print::Column::left("KEY"),
    ];
    let rows: Vec<Vec<String>> = blocks
        .iter()
        .map(|b| {
            // A worker block runs the call itself, so there is no key to bind
            // here and an empty cell would read as one that is missing.
            let key = match (b.kind.as_str(), b.key_bound) {
                ("worker", _) => "n/a (your worker runs it)",
                (_, true) => "set",
                (_, false) => "not set",
            };
            vec![b.name.clone(), b.kind.clone(), key.into()]
        })
        .collect();
    print::table(&columns, &rows);
    Ok(())
}

async fn set_key(block: String, env: Option<String>, scope: ProjectScope) -> Result<()> {
    let key = match &env {
        Some(var) => std::env::var(var)
            .with_context(|| format!("${var} is not set"))?
            .trim()
            .to_string(),
        None => pickers::read_secret(&scope.globals, "Paste the key")?,
    };
    if key.is_empty() {
        bail!("no key given. Pipe it in, or pass --env <VAR>.");
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
         [agent.a]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"https://w.test\"\n";

    /// The block's key here is a variable, so the answer is its name and
    /// whether it holds anything.
    #[test]
    fn a_block_here_reports_the_variable_it_reads() {
        let config = config(BLOCKS);
        let [name, kind, key] = declared("claude", &config).unwrap();
        assert_eq!(name, "claude");
        assert_eq!(kind, "anthropic");
        assert_eq!(key, "$NOT_SET_ANTHROPIC (not set)");
    }

    /// A worker block never needs a key where the engine runs, so an empty cell
    /// would read as a missing one.
    #[test]
    fn a_worker_block_needs_no_key_here() {
        let config = config(BLOCKS);
        let [_, kind, key] = declared("byo", &config).unwrap();
        assert_eq!(kind, "worker");
        assert!(key.contains("your worker runs it"), "{key}");
    }

    /// Nothing to upload: the message names the variable to export instead.
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
        assert!(err.contains("worker block"), "{err}");
    }
}
