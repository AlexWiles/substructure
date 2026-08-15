//! Agent plugins (<https://agent-plugins.org>) as the engine holds them: a
//! directory resolved into data once, at load or at `subs apply`, and never
//! read as files again. No code execution: `scripts/` and stdio servers are
//! dropped with a notice, per the spec's failure isolation — a component that
//! cannot load never takes its siblings with it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::connectors::registry::ConnectionSpec;
use crate::protocol::{ConnectorProtocol, SkillMeta};

/// One plugin, resolved to data. Immutable once built: a session reads the
/// bundle its config was loaded with, so a re-apply cannot rewrite a session
/// underneath itself.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PluginBundle {
    /// From `plugin.json`. Metadata, not the id: the config key an agent
    /// references is the id, the same way `[mcp.<id>]` works.
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<Skill>,
    /// `mcp.json`'s remote servers, keyed by their name in the file. stdio
    /// entries are not here: nothing in a deployment runs plugin code.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub servers: BTreeMap<String, ConnectionSpec>,
}

/// One skill: metadata the model discovers, a body it loads on use, and the
/// files the body references.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// The SKILL.md markdown after the frontmatter.
    pub body: String,
    /// Skill-relative path → UTF-8 content. Binary files are not carried:
    /// a notice says what was left behind.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub files: BTreeMap<String, String>,
}

impl PluginBundle {
    pub fn skill(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.name == name)
    }

    /// The metadata half the wire config carries.
    pub fn skill_metas(&self) -> Vec<SkillMeta> {
        self.skills
            .iter()
            .map(|s| SkillMeta {
                name: s.name.clone(),
                description: s.description.clone(),
            })
            .collect()
    }

    /// Content hash, stable across field order and machines.
    pub fn hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        format!("{:x}", Sha256::digest(bytes))
    }
}

/// The bundles an engine serves, keyed by plugin id — the runtime's read-only
/// copy of what the config declared.
pub type PluginSet = BTreeMap<String, PluginBundle>;

/// Where a plugin id turns into its bundle, per tenant — the seam the cloud
/// swaps for a database, the way `ConnectionRegistry` works.
#[async_trait::async_trait]
pub trait PluginResolver: Send + Sync {
    async fn resolve(&self, tenant_id: &str, plugin_id: &str) -> Option<PluginBundle>;

    /// Whether any plugin could resolve here, read once at startup to decide
    /// whether the answering subsystem runs at all.
    fn serves_any(&self) -> bool {
        true
    }
}

/// Every bundle a local engine loaded at startup, served to its single
/// tenant.
#[derive(Default)]
pub struct StaticPlugins(PluginSet);

impl StaticPlugins {
    pub fn new(set: PluginSet) -> Self {
        Self(set)
    }
}

#[async_trait::async_trait]
impl PluginResolver for StaticPlugins {
    async fn resolve(&self, _tenant_id: &str, plugin_id: &str) -> Option<PluginBundle> {
        self.0.get(plugin_id).cloned()
    }

    fn serves_any(&self) -> bool {
        !self.0.is_empty()
    }
}

/// What loading dropped and why. Every notice is a fact for the operator, not
/// an error: the bundle that rides along is still whole.
#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    pub bundle: PluginBundle,
    pub notices: Vec<String>,
}

/// The id a plugin's server resolves under in the connection registry.
/// `<plugin>-<server>`: an id prefixes tool names, so `.` is not available.
pub fn server_id(plugin_id: &str, server_name: &str) -> String {
    format!("{plugin_id}-{server_name}")
}

/// Read one plugin directory into a bundle, per the agent-plugins layout.
/// A broken component is skipped with a notice; only a broken `plugin.json`
/// rejects the plugin.
pub fn load_dir(root: &Path) -> Result<Loaded> {
    let mut notices = Vec::new();
    let manifest_path = root.join("plugin.json");
    let manifest = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: PluginManifest = serde_json::from_str(&manifest)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    if manifest.name.is_empty() {
        bail!("{}: `name` is empty", manifest_path.display());
    }

    let skills = load_skills(&root.join("skills"), &mut notices)?;
    let servers = load_servers(&root.join("mcp.json"), &mut notices)?;

    if root.join("hooks").is_dir() {
        notices.push("hooks/ is not supported and was ignored".to_string());
    }

    Ok(Loaded {
        bundle: PluginBundle {
            name: sanitize_line(&manifest.name),
            version: manifest.version.map(|v| sanitize_line(&v)),
            description: sanitize_line(&manifest.description),
            skills,
            servers,
        },
        notices,
    })
}

/// `plugin.json`. Unknown fields are allowed, per the spec: they warn there,
/// and here they are simply not read.
#[derive(Debug, Deserialize)]
struct PluginManifest {
    name: String,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    description: String,
}

fn load_skills(dir: &Path, notices: &mut Vec<String>) -> Result<Vec<Skill>> {
    let mut skills = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(skills);
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for skill_dir in dirs {
        match load_skill(&skill_dir, notices) {
            Ok(skill) => skills.push(skill),
            Err(e) => notices.push(format!("skill {} skipped: {e}", skill_dir.display())),
        }
    }
    Ok(skills)
}

fn load_skill(dir: &Path, notices: &mut Vec<String>) -> Result<Skill> {
    let text = std::fs::read_to_string(dir.join("SKILL.md")).context("no SKILL.md")?;
    let (front, body) = split_frontmatter(&text)?;
    let name = frontmatter_value(front, "name").context("frontmatter has no `name`")?;
    let description =
        frontmatter_value(front, "description").context("frontmatter has no `description`")?;
    check_skill_name(&name)?;
    if dir.file_name().and_then(|n| n.to_str()) != Some(name.as_str()) {
        bail!("`name: {name}` does not match the directory name");
    }
    if description.is_empty() {
        bail!("`description` is empty");
    }

    let mut files = BTreeMap::new();
    collect_files(dir, dir, &mut files, notices)?;
    files.remove("SKILL.md");

    Ok(Skill {
        name,
        description: sanitize_line(&description),
        body: sanitize_text(body),
        files,
    })
}

/// The frontmatter block between the leading `---` fences.
fn split_frontmatter(text: &str) -> Result<(&str, &str)> {
    let rest = text
        .strip_prefix("---")
        .context("SKILL.md does not start with frontmatter")?;
    let end = rest.find("\n---").context("frontmatter never closes")?;
    let body = rest[end + 4..].trim_start_matches(['\r', '\n']);
    Ok((&rest[..end], body))
}

/// One top-level `key: value` scalar. The spec's frontmatter is YAML; the two
/// fields read here are required to be plain scalars, and a skill that writes
/// them another way is skipped with a notice rather than half-read.
fn frontmatter_value(front: &str, key: &str) -> Option<String> {
    front.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?.trim_start().strip_prefix(':')?;
        let value = rest.trim().trim_matches('"').trim_matches('\'');
        (!value.is_empty()).then(|| value.to_string())
    })
}

/// The Agent Skills `name` rules, which also keep it usable in a tool-ish name.
fn check_skill_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--");
    if !ok {
        bail!("`{name}` is not a valid skill name (lowercase, digits, single hyphens)");
    }
    Ok(())
}

fn collect_files(
    root: &Path,
    dir: &Path,
    files: &mut BTreeMap<String, String>,
    notices: &mut Vec<String>,
) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for path in entries {
        // A symlink may point outside the plugin; the spec allows only
        // inside-root targets, and not following any is the simple safe read.
        if path.is_symlink() {
            notices.push(format!("{} is a symlink and was ignored", path.display()));
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, files, notices)?;
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                files.insert(rel, sanitize_text(&text));
            }
            Err(_) => notices.push(format!("{rel} is not UTF-8 text and was left behind")),
        }
    }
    Ok(())
}

/// `mcp.json`. Remote servers become [`ConnectionSpec`]s; everything a
/// deployment cannot honor is a notice.
fn load_servers(
    path: &Path,
    notices: &mut Vec<String>,
) -> Result<BTreeMap<String, ConnectionSpec>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(BTreeMap::new());
    };
    let file: McpFile =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let mut servers = BTreeMap::new();
    for (name, server) in file.mcp_servers {
        match server.kind.as_str() {
            "streamable-http" | "sse" => {
                let Some(url) = server.url else {
                    notices.push(format!("mcp server `{name}` has no `url`; skipped"));
                    continue;
                };
                if !server.headers.is_empty() {
                    notices.push(format!(
                        "mcp server `{name}` declares `headers`; static credentials are set \
                         with `subs mcp set-token`, not carried in a plugin"
                    ));
                }
                servers.insert(
                    name,
                    ConnectionSpec {
                        url,
                        protocol: ConnectorProtocol::Mcp,
                        auth: None,
                        header: None,
                        prefix_tools: true,
                    },
                );
            }
            "stdio" => notices.push(format!(
                "mcp server `{name}` is stdio; plugin code does not run here, so it was skipped"
            )),
            other => notices.push(format!("mcp server `{name}` has unknown type `{other}`")),
        }
    }
    Ok(servers)
}

#[derive(Debug, Deserialize)]
struct McpFile {
    #[serde(rename = "mcpServers", default)]
    mcp_servers: BTreeMap<String, McpFileServer>,
}

#[derive(Debug, Deserialize)]
struct McpFileServer {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
}

/// Author-controlled text that lands near a prompt: control characters out,
/// one line only.
fn sanitize_line(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>()
        .trim()
        .to_string()
}

/// Body text keeps its shape; only the invisible is removed.
fn sanitize_text(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, content: &str) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn plugin_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(
            root,
            "plugin.json",
            r#"{ "name": "pdf-tools", "version": "1.2.0", "description": "PDF work." }"#,
        );
        write(
            root,
            "skills/form-filling/SKILL.md",
            "---\nname: form-filling\ndescription: Fill out PDF forms.\n---\n\nRead references/FORMS.md first.\n",
        );
        write(root, "skills/form-filling/references/FORMS.md", "fields…");
        write(
            root,
            "mcp.json",
            r#"{ "mcpServers": {
                "renderer": { "type": "streamable-http", "url": "https://pdf.example.com/mcp" },
                "helper": { "type": "stdio", "command": "./run.py" }
            }}"#,
        );
        dir
    }

    #[test]
    fn a_directory_loads_into_a_bundle() {
        let dir = plugin_dir();
        let loaded = load_dir(dir.path()).unwrap();
        let b = &loaded.bundle;
        assert_eq!(b.name, "pdf-tools");
        assert_eq!(b.version.as_deref(), Some("1.2.0"));
        assert_eq!(b.skills.len(), 1);
        let skill = &b.skills[0];
        assert_eq!(skill.name, "form-filling");
        assert_eq!(skill.description, "Fill out PDF forms.");
        assert!(skill.body.starts_with("Read references/FORMS.md"));
        assert_eq!(skill.files["references/FORMS.md"], "fields…");
        assert!(
            !skill.files.contains_key("SKILL.md"),
            "the body is not a file"
        );
    }

    #[test]
    fn a_stdio_server_is_a_notice_not_an_error() {
        let dir = plugin_dir();
        let loaded = load_dir(dir.path()).unwrap();
        assert_eq!(loaded.bundle.servers.len(), 1, "only the remote one");
        assert!(loaded.bundle.servers.contains_key("renderer"));
        assert!(
            loaded.notices.iter().any(|n| n.contains("stdio")),
            "got {:?}",
            loaded.notices
        );
    }

    #[test]
    fn a_broken_skill_is_skipped_and_the_rest_load() {
        let dir = plugin_dir();
        write(
            dir.path(),
            "skills/broken/SKILL.md",
            "---\nname: mismatch\ndescription: d.\n---\nbody",
        );
        let loaded = load_dir(dir.path()).unwrap();
        assert_eq!(loaded.bundle.skills.len(), 1);
        assert!(
            loaded.notices.iter().any(|n| n.contains("broken")),
            "got {:?}",
            loaded.notices
        );
    }

    #[test]
    fn descriptions_cannot_carry_control_characters() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "plugin.json",
            "{ \"name\": \"x\", \"description\": \"a\\u001b[31mb\\nc\" }",
        );
        let loaded = load_dir(dir.path()).unwrap();
        assert_eq!(loaded.bundle.description, "a [31mb c");
    }

    #[test]
    fn the_hash_is_stable_and_content_bound() {
        let dir = plugin_dir();
        let a = load_dir(dir.path()).unwrap().bundle;
        let b = load_dir(dir.path()).unwrap().bundle;
        assert_eq!(a.hash(), b.hash());
        let mut c = a.clone();
        c.skills[0].body.push('!');
        assert_ne!(a.hash(), c.hash());
    }

    #[test]
    fn server_ids_are_tool_prefix_safe() {
        assert_eq!(server_id("pdf", "renderer"), "pdf-renderer");
        crate::manifest::check_id(&server_id("pdf", "renderer")).unwrap();
    }
}
