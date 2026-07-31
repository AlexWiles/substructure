//! `subs llm`: the keys behind the `[llm.*]` blocks the file declares.
//!
//! A block is declared in `substructure.toml`; the key for it is uploaded here.
//! Pasting a key is configuration rather than money or consent, so it belongs
//! on this side of the write partition — and it never appears in argv, where a
//! shell history would keep it.

use anyhow::{bail, Context as _, Result};
use clap::Subcommand;
use serde::Serialize;
use std::io::Read as _;

use crate::api::v1::{LlmBlockView, LlmKeyRequest};

use super::context::Context;
use super::print;
use super::ProjectScope;

#[derive(Subcommand)]
pub enum LlmCommand {
    /// List the blocks this project declares and whether each has a key.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Upload the key for one block. Read from stdin, or from the environment
    /// variable `--env` names.
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
        LlmCommand::List { scope } => list(scope).await,
        LlmCommand::SetKey { block, env, scope } => set_key(block, env, scope).await,
        LlmCommand::DeleteKey { block, scope } => delete_key(block, scope).await,
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
        None => read_stdin()?,
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

/// The key from stdin, whether it was piped or typed. A trailing newline is the
/// shell's, not the key's.
fn read_stdin() -> Result<String> {
    if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        eprintln!("Paste the key, then press Ctrl-D:");
    }
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("reading the key from stdin")?;
    Ok(buf.trim().to_string())
}
