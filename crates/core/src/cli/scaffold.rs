use std::io::{Cursor, IsTerminal};
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context as _, Result};
use dialoguer::{theme::ColorfulTheme, Select};
use flate2::read::GzDecoder;
use reqwest::StatusCode;
use serde::Deserialize;
use tar::Archive;

const OWNER: &str = "substructureai";
const REPO: &str = "substructure";

/// The git ref the binary fetches templates from. Templates are pure-copy
/// starters the binary never introspects, so we track `main` rather than
/// pinning to a release tag — that keeps template edits live without cutting a
/// binary release. Flip this to `v{version}` if templates ever start depending
/// on binary behavior.
const TEMPLATES_REF: &str = "main";

/// Highest `index.toml` schema this build understands.
const SUPPORTED_SCHEMA: u32 = 1;

#[derive(Debug, clap::Args)]
pub struct NewCommand {
    /// Template id to scaffold. Omit to launch the interactive picker.
    pub template: Option<String>,
    /// Target directory the template is copied into (default: ./<template-id>).
    pub dir: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Catalog {
    schema: u32,
    #[serde(default, rename = "template")]
    templates: Vec<Template>,
}

#[derive(Debug, Deserialize)]
struct Template {
    id: String,
    description: String,
    language: String,
    #[serde(default)]
    #[allow(dead_code)]
    requires_remote: bool,
    next_steps: String,
}

pub async fn run(cmd: NewCommand) -> Result<()> {
    let client = http_client()?;
    let catalog = fetch_catalog(&client).await?;
    if catalog.templates.is_empty() {
        bail!("the templates catalog is empty.");
    }

    let template = match &cmd.template {
        Some(id) => catalog
            .templates
            .iter()
            .find(|t| &t.id == id)
            .ok_or_else(|| anyhow!("unknown template '{id}'.\n\n{}", available(&catalog)))?,
        None => pick(&catalog)?,
    };

    let target = PathBuf::from(cmd.dir.clone().unwrap_or_else(|| template.id.clone()));
    ensure_writable(&target)?;

    println!("Fetching template '{}'…", template.id);
    let count = fetch_and_extract(&client, &template.id, &target).await?;
    if count == 0 {
        bail!(
            "template '{}' has no files at templates/{}/ in the source archive.",
            template.id,
            template.id
        );
    }

    let dir_display = target.display().to_string();
    println!();
    println!("Created {dir_display} from template '{}'.", template.id);
    println!();
    println!("Next steps:");
    println!("{}", render_next_steps(&template.next_steps, &dir_display));
    Ok(())
}

fn http_client() -> Result<reqwest::Client> {
    let user_agent = format!(
        "subs/{} ({}; {})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    reqwest::Client::builder()
        .user_agent(user_agent)
        .build()
        .context("building http client")
}

fn index_url() -> String {
    format!("https://raw.githubusercontent.com/{OWNER}/{REPO}/{TEMPLATES_REF}/templates/index.toml")
}

fn tarball_url() -> String {
    format!("https://codeload.github.com/{OWNER}/{REPO}/tar.gz/{TEMPLATES_REF}")
}

async fn fetch_catalog(client: &reqwest::Client) -> Result<Catalog> {
    let url = index_url();
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow!("can't reach the templates source ({url}): {e}"))?;
    if resp.status() == StatusCode::NOT_FOUND {
        bail!("templates catalog not found at {url}.");
    }
    let resp = resp
        .error_for_status()
        .with_context(|| format!("fetching templates catalog from {url}"))?;
    let body = resp
        .text()
        .await
        .context("reading templates catalog body")?;
    parse_catalog(&body)
}

fn parse_catalog(body: &str) -> Result<Catalog> {
    let catalog: Catalog = toml::from_str(body).context("parsing templates/index.toml")?;
    if catalog.schema != SUPPORTED_SCHEMA {
        bail!(
            "templates catalog schema {} is not supported by this build (understands {SUPPORTED_SCHEMA}). Upgrade substructure.",
            catalog.schema
        );
    }
    Ok(catalog)
}

fn available(catalog: &Catalog) -> String {
    let mut out = String::from("Available templates:");
    for t in &catalog.templates {
        out.push_str(&format!(
            "\n  {}  —  {} [{}]",
            t.id, t.description, t.language
        ));
    }
    out
}

fn pick(catalog: &Catalog) -> Result<&Template> {
    if !std::io::stdin().is_terminal() {
        bail!("no template specified and not running interactively. Pass a template id: substructure new <template>");
    }
    let items: Vec<String> = catalog
        .templates
        .iter()
        .map(|t| format!("{}  —  {} [{}]", t.id, t.description, t.language))
        .collect();
    let idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a template")
        .items(&items)
        .default(0)
        .interact()
        .context("template picker")?;
    Ok(&catalog.templates[idx])
}

/// Refuse to scaffold into a path that already holds files — we never clobber.
fn ensure_writable(target: &Path) -> Result<()> {
    if target.exists() && !target.is_dir() {
        bail!("target {} exists and is not a directory.", target.display());
    }
    match std::fs::read_dir(target) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                bail!(
                    "target directory {} is not empty. Refusing to overwrite existing files.",
                    target.display()
                );
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("checking target directory {}", target.display())),
    }
}

async fn fetch_and_extract(client: &reqwest::Client, id: &str, target: &Path) -> Result<usize> {
    let url = tarball_url();
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow!("can't reach the templates source ({url}): {e}"))?
        .error_for_status()
        .with_context(|| format!("fetching template archive from {url}"))?;
    let bytes = resp.bytes().await.context("downloading template archive")?;
    extract_subdir(&bytes, id, target)
}

/// Extract every file under `templates/<id>/` from a gzipped tar into `target`,
/// returning the number of files written.
fn extract_subdir(gzipped_tar: &[u8], id: &str, target: &Path) -> Result<usize> {
    std::fs::create_dir_all(target)
        .with_context(|| format!("creating target directory {}", target.display()))?;

    let mut archive = Archive::new(GzDecoder::new(Cursor::new(gzipped_tar)));
    let mut count = 0usize;
    for entry in archive.entries().context("reading template archive")? {
        let mut entry = entry.context("reading archive entry")?;
        let path = entry.path().context("archive entry path")?.into_owned();
        let Some(rel) = entry_target_rel(&path, id) else {
            continue;
        };
        let out = target.join(&rel);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        entry
            .unpack(&out)
            .with_context(|| format!("extracting {}", out.display()))?;
        count += 1;
    }
    Ok(count)
}

/// Map a tar entry path to its path under the target dir, or `None` if it
/// doesn't live under `templates/<id>/`. The archive's top-level prefix
/// component (e.g. `substructure-main/`) is stripped generically so we don't
/// depend on GitHub's archive naming. Entries with traversal components are
/// rejected.
fn entry_target_rel(entry_path: &Path, id: &str) -> Option<PathBuf> {
    let mut comps = entry_path.components();
    comps.next()?; // drop the archive's top-level prefix dir
    let rest = comps.as_path();
    let prefix = Path::new("templates").join(id);
    let rel = rest.strip_prefix(&prefix).ok()?;
    if rel.as_os_str().is_empty() {
        return None; // the template dir entry itself
    }
    if rel.components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return None;
    }
    Some(rel.to_path_buf())
}

fn render_next_steps(next_steps: &str, dir: &str) -> String {
    next_steps.replace("{dir}", dir).trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_catalog() {
        let cat = parse_catalog(
            r#"
            schema = 1

            [[template]]
            id = "bare-worker"
            description = "Minimal worker"
            language = "typescript"
            requires_remote = false
            next_steps = "cd {dir}"
            "#,
        )
        .unwrap();
        assert_eq!(cat.schema, 1);
        assert_eq!(cat.templates.len(), 1);
        assert_eq!(cat.templates[0].id, "bare-worker");
    }

    #[test]
    fn rejects_unknown_schema() {
        let err = parse_catalog("schema = 2\n").unwrap_err().to_string();
        assert!(err.contains("not supported"), "{err}");
    }

    #[test]
    fn maps_entries_under_template_subdir() {
        let rel = entry_target_rel(
            Path::new("substructure-main/templates/bare-worker/src/index.ts"),
            "bare-worker",
        );
        assert_eq!(rel, Some(PathBuf::from("src/index.ts")));
    }

    #[test]
    fn ignores_other_templates_and_the_dir_itself() {
        assert_eq!(
            entry_target_rel(
                Path::new("substructure-main/templates/other/x.ts"),
                "bare-worker"
            ),
            None
        );
        assert_eq!(
            entry_target_rel(
                Path::new("substructure-main/templates/bare-worker"),
                "bare-worker"
            ),
            None
        );
        assert_eq!(
            entry_target_rel(Path::new("substructure-main/README.md"), "bare-worker"),
            None
        );
    }

    #[test]
    fn rejects_traversal() {
        assert_eq!(
            entry_target_rel(
                Path::new("substructure-main/templates/bare-worker/../../etc/passwd"),
                "bare-worker"
            ),
            None
        );
    }

    #[test]
    fn fills_dir_token_in_next_steps() {
        assert_eq!(
            render_next_steps("  cd {dir}\n  pnpm install\n", "./bare-worker"),
            "  cd ./bare-worker\n  pnpm install"
        );
    }

    fn gzipped_tar(files: &[(&str, &str)]) -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write as _;

        let mut builder = tar::Builder::new(Vec::new());
        for (path, contents) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, path, contents.as_bytes())
                .unwrap();
        }
        let tar_bytes = builder.into_inner().unwrap();
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&tar_bytes).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn extracts_only_the_chosen_template_subdir() {
        let archive = gzipped_tar(&[
            ("substructure-main/README.md", "repo readme"),
            ("substructure-main/templates/bare-worker/package.json", "{}"),
            (
                "substructure-main/templates/bare-worker/src/index.ts",
                "export {}",
            ),
            ("substructure-main/templates/other/ignored.txt", "nope"),
        ]);

        let target = std::env::temp_dir().join("subs-scaffold-extract-test");
        let _ = std::fs::remove_dir_all(&target);

        let count = extract_subdir(&archive, "bare-worker", &target).unwrap();
        assert_eq!(count, 2);
        assert_eq!(
            std::fs::read_to_string(target.join("package.json")).unwrap(),
            "{}"
        );
        assert_eq!(
            std::fs::read_to_string(target.join("src/index.ts")).unwrap(),
            "export {}"
        );
        assert!(!target.join("ignored.txt").exists());
        assert!(!target.join("README.md").exists());

        std::fs::remove_dir_all(&target).unwrap();
    }
}
