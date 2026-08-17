//! Agent plugins (<https://agent-plugins.org>), read from a directory into
//! data. Plugin code does not run here: `scripts/` and stdio servers are
//! dropped with a notice.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::connectors::registry::ConnectionSpec;
use crate::protocol::{ConnectorProtocol, SkillMeta, OCTET_STREAM};
use crate::runtime::blob::{BlobStore, NewBlob};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PluginBundle {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<Skill>,
    /// `mcp.json`'s remote servers, keyed by their name in the file.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub servers: BTreeMap<String, ConnectionSpec>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// SKILL.md after the frontmatter.
    pub body: String,
    /// Skill-relative path → UTF-8 content.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub files: BTreeMap<String, String>,
    /// Skill-relative path → a non-text file, by blob ref.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub binaries: BTreeMap<String, Binary>,
}

/// A stored non-text file.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Binary {
    /// `blob://…`
    pub uri: String,
    pub mime: String,
    pub size: u64,
}

/// A binary read from the directory, not yet stored.
#[derive(Debug, Clone, PartialEq)]
pub struct Pending {
    pub skill: String,
    pub path: String,
    pub mime: String,
    pub bytes: Vec<u8>,
}

impl PluginBundle {
    pub fn skill(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.name == name)
    }

    pub fn skill_metas(&self) -> Vec<SkillMeta> {
        self.skills
            .iter()
            .map(|s| SkillMeta {
                name: s.name.clone(),
                description: s.description.clone(),
            })
            .collect()
    }

    pub fn hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        format!("{:x}", Sha256::digest(bytes))
    }
}

/// Bundles keyed by plugin id.
pub type PluginSet = BTreeMap<String, PluginBundle>;

#[async_trait::async_trait]
pub trait PluginResolver: Send + Sync {
    async fn resolve(&self, tenant_id: &str, plugin_id: &str) -> Option<PluginBundle>;

    fn serves_any(&self) -> bool {
        true
    }
}

/// Serves one fixed set to every tenant.
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

#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    pub bundle: PluginBundle,
    /// What loading dropped. Not errors.
    pub notices: Vec<String>,
    /// Binaries waiting for a blob store.
    pub pending: Vec<Pending>,
}

impl Loaded {
    /// Taken before the binaries are stored: a ref is minted per put, so a
    /// hash over refs would change on every load.
    pub fn hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(self.bundle.hash());
        for file in &self.pending {
            digest.update(&file.skill);
            digest.update(&file.path);
            digest.update(&file.mime);
            digest.update(&file.bytes);
        }
        format!("{:x}", digest.finalize())
    }
}

/// The connection id of a plugin's server.
pub fn server_id(plugin_id: &str, server_name: &str) -> String {
    format!("{plugin_id}-{server_name}")
}

/// Read a plugin directory. A broken component gives a notice; only a broken
/// `plugin.json` is an error.
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

    let mut pending = Vec::new();
    let skills = load_skills(&root.join("skills"), &mut pending, &mut notices)?;
    let servers = load_servers(&root.join("mcp.json"), &mut notices)?;

    if root.join("hooks").is_dir() {
        notices.push("hooks/ is not supported and was ignored".to_string());
    }

    let bundle = PluginBundle {
        name: sanitize_line(&manifest.name),
        version: manifest.version.map(|v| sanitize_line(&v)),
        description: sanitize_line(&manifest.description),
        skills,
        servers,
    };
    let bytes = content_bytes(&bundle);
    if bytes > MAX_BUNDLE_BYTES {
        bail!(
            "{} holds {bytes} bytes of skill content; the limit is {MAX_BUNDLE_BYTES}. \
             Move large files out of the skills, or split the plugin.",
            root.display()
        );
    }
    let binary_bytes: usize = pending.iter().map(|p| p.bytes.len()).sum();
    if binary_bytes > MAX_BINARY_BYTES {
        bail!(
            "{} holds {binary_bytes} bytes of binary skill files; the limit is \
             {MAX_BINARY_BYTES}. Serve files this large from a server instead.",
            root.display()
        );
    }
    Ok(Loaded {
        bundle,
        notices,
        pending,
    })
}

/// Skill text goes in the prompt, so this budget is small.
const MAX_BUNDLE_BYTES: usize = 10 * 1024 * 1024;

/// A binary does not go in the prompt; it only has to be stored.
const MAX_BINARY_BYTES: usize = 100 * 1024 * 1024;

/// Put each pending binary in blob storage and stamp the ref on its skill.
/// One that will not store is dropped with a notice.
pub async fn store_binaries(
    bundle: &mut PluginBundle,
    pending: Vec<Pending>,
    tenant_id: &str,
    blobs: &dyn BlobStore,
) -> Vec<String> {
    let mut notices = Vec::new();
    for file in pending {
        let size = file.bytes.len() as u64;
        let put = blobs
            .put(NewBlob {
                tenant_id: tenant_id.to_string(),
                mime: file.mime.clone(),
                name: file.path.rsplit('/').next().map(str::to_string),
                bytes: file.bytes,
            })
            .await;
        match put {
            Ok(stored) => {
                if let Some(skill) = bundle.skills.iter_mut().find(|s| s.name == file.skill) {
                    skill.binaries.insert(
                        file.path,
                        Binary {
                            uri: stored.uri(),
                            mime: file.mime,
                            size,
                        },
                    );
                }
            }
            Err(e) => notices.push(format!(
                "{} in skill `{}` was not stored and is unreadable: {e}",
                file.path, file.skill
            )),
        }
    }
    notices
}

fn content_bytes(bundle: &PluginBundle) -> usize {
    bundle
        .skills
        .iter()
        .map(|s| s.body.len() + s.files.values().map(String::len).sum::<usize>())
        .sum()
}

#[derive(Debug, Deserialize)]
struct PluginManifest {
    name: String,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    description: String,
}

fn load_skills(
    dir: &Path,
    pending: &mut Vec<Pending>,
    notices: &mut Vec<String>,
) -> Result<Vec<Skill>> {
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
        // A failed skill leaves nothing behind.
        let mut found = Vec::new();
        match load_skill(&skill_dir, &mut found, notices) {
            Ok(skill) => {
                skills.push(skill);
                pending.append(&mut found);
            }
            Err(e) => notices.push(format!("skill {} skipped: {e}", skill_dir.display())),
        }
    }
    Ok(skills)
}

fn load_skill(dir: &Path, pending: &mut Vec<Pending>, notices: &mut Vec<String>) -> Result<Skill> {
    let text = std::fs::read_to_string(dir.join("SKILL.md")).context("no SKILL.md")?;
    let (front, body) = split_frontmatter(&text)?;
    let name = frontmatter_value(front, "name").ok_or_else(|| unreadable_key(front, "name"))?;
    let description = frontmatter_value(front, "description")
        .ok_or_else(|| unreadable_key(front, "description"))?;
    check_skill_name(&name)?;
    if dir.file_name().and_then(|n| n.to_str()) != Some(name.as_str()) {
        bail!("`name: {name}` does not match the directory name");
    }
    if description.is_empty() {
        bail!("`description` is empty");
    }

    let mut files = BTreeMap::new();
    let mut found = Vec::new();
    collect_files(dir, dir, &mut files, &mut found, notices)?;
    files.remove("SKILL.md");
    pending.extend(found.into_iter().map(|(path, mime, bytes)| Pending {
        skill: name.clone(),
        path,
        mime,
        bytes,
    }));

    Ok(Skill {
        name,
        description: sanitize_line(&description),
        body: sanitize_text(body),
        files,
        binaries: BTreeMap::new(),
    })
}

fn split_frontmatter(text: &str) -> Result<(&str, &str)> {
    let rest = text
        .strip_prefix("---")
        .context("SKILL.md does not start with frontmatter")?;
    let end = rest.find("\n---").context("frontmatter never closes")?;
    let body = rest[end + 4..].trim_start_matches(['\r', '\n']);
    Ok((&rest[..end], body))
}

/// One top-level `key: value`. Plain scalars only.
fn frontmatter_value(front: &str, key: &str) -> Option<String> {
    front.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?.trim_start().strip_prefix(':')?;
        let value = rest.trim().trim_matches('"').trim_matches('\'');
        // A YAML block indicator is not a value.
        if matches!(value, ">" | "|" | ">-" | "|-" | ">+" | "|+") {
            return None;
        }
        (!value.is_empty()).then(|| value.to_string())
    })
}

fn unreadable_key(front: &str, key: &str) -> anyhow::Error {
    let written = front.lines().any(|l| {
        l.strip_prefix(key)
            .is_some_and(|r| r.trim_start().starts_with(':'))
    });
    match written {
        true => anyhow::anyhow!(
            "frontmatter `{key}` is not a plain value; folded and multiline strings are not \
             supported"
        ),
        false => anyhow::anyhow!("frontmatter has no `{key}`"),
    }
}

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
    binaries: &mut Vec<(String, String, Vec<u8>)>,
    notices: &mut Vec<String>,
) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for path in entries {
        // A symlink can point out of the plugin.
        if path.is_symlink() {
            notices.push(format!("{} is a symlink and was ignored", path.display()));
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, files, binaries, notices)?;
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) => {
                notices.push(format!("{rel} was not read and was left behind: {e}"));
                continue;
            }
        };
        // The name decides first: a small PDF is valid UTF-8 and still not text.
        if let Some(mime) = named_mime(&rel) {
            binaries.push((rel, mime.to_string(), bytes));
            continue;
        }
        match String::from_utf8(bytes) {
            Ok(text) => {
                files.insert(rel, sanitize_text(&text));
            }
            Err(e) => binaries.push((rel, OCTET_STREAM.to_string(), e.into_bytes())),
        }
    }
    Ok(())
}

/// The mime of a skill file whose extension says it is not text.
fn named_mime(path: &str) -> Option<&'static str> {
    let ext = path.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase())?;
    let mime = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "ico" => "image/vnd.microsoft.icon",
        "tif" | "tiff" => "image/tiff",
        "pdf" => "application/pdf",
        "zip" => "application/zip",
        "gz" => "application/gzip",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "mp4" => "video/mp4",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        _ => return None,
    };
    Some(mime)
}

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
                        credential: None,
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

/// Plugin text goes into prompts. Remove escapes and control characters, and
/// keep one line.
fn sanitize_line(text: &str) -> String {
    strip_escapes(text)
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>()
        .trim()
        .to_string()
}

fn sanitize_text(text: &str) -> String {
    strip_escapes(text)
        .chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .collect()
}

fn strip_escapes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        if chars.peek() == Some(&'[') {
            chars.next();
            for d in chars.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            chars.next();
        }
    }
    out
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

    fn png(root: &Path, rel: &str) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, [0x89, b'P', b'N', b'G', 0xff, 0xfe]).unwrap();
    }

    #[test]
    fn a_binary_skill_file_waits_for_a_blob_store() {
        let dir = plugin_dir();
        png(dir.path(), "skills/form-filling/references/form.png");
        let loaded = load_dir(dir.path()).unwrap();
        let skill = &loaded.bundle.skills[0];
        assert!(
            !skill.files.contains_key("references/form.png"),
            "a binary is not skill text"
        );
        assert!(
            skill.binaries.is_empty(),
            "nothing is stamped before it is stored"
        );
        assert_eq!(loaded.pending.len(), 1);
        let pending = &loaded.pending[0];
        assert_eq!(pending.skill, "form-filling");
        assert_eq!(pending.path, "references/form.png");
        assert_eq!(pending.mime, "image/png");
    }

    #[tokio::test]
    async fn storing_a_binary_leaves_the_bundle_holding_a_ref() {
        let dir = plugin_dir();
        png(dir.path(), "skills/form-filling/references/form.png");
        let mut loaded = load_dir(dir.path()).unwrap();
        let blobs = crate::runtime::blob::MemoryBlobStore::new();
        let notices = store_binaries(&mut loaded.bundle, loaded.pending, "t1", &blobs).await;
        assert!(notices.is_empty(), "got {notices:?}");

        let binary = &loaded.bundle.skills[0].binaries["references/form.png"];
        assert_eq!(binary.mime, "image/png");
        assert_eq!(binary.size, 6);
        let stored = crate::runtime::blob::BlobRef::parse(&binary.uri).expect("a ref");
        assert_eq!(blobs.get(&stored).await.unwrap().len(), 6);
        assert!(
            !serde_json::to_string(&loaded.bundle)
                .unwrap()
                .contains("PNG"),
            "the bytes do not travel with the config"
        );
    }

    #[test]
    fn a_named_format_is_binary_whatever_its_bytes_say() {
        let dir = plugin_dir();
        // Valid UTF-8, and still a document.
        write(
            dir.path(),
            "skills/form-filling/references/policy.pdf",
            "%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\n",
        );
        let loaded = load_dir(dir.path()).unwrap();
        assert!(
            !loaded.bundle.skills[0]
                .files
                .contains_key("references/policy.pdf"),
            "a PDF does not inline as text"
        );
        assert_eq!(loaded.pending[0].mime, "application/pdf");
    }

    #[test]
    fn the_hash_follows_a_binary_that_changed() {
        let dir = plugin_dir();
        png(dir.path(), "skills/form-filling/references/form.png");
        let before = load_dir(dir.path()).unwrap().hash();
        std::fs::write(
            dir.path().join("skills/form-filling/references/form.png"),
            [0x89, b'P', b'N', b'G', 0x00, 0x01],
        )
        .unwrap();
        assert_ne!(before, load_dir(dir.path()).unwrap().hash());
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
    fn descriptions_cannot_carry_escapes_or_control_characters() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "plugin.json",
            "{ \"name\": \"x\", \"description\": \"a\\u001b[31mb\\nc\" }",
        );
        let loaded = load_dir(dir.path()).unwrap();
        assert_eq!(
            loaded.bundle.description, "ab c",
            "the whole sequence goes, not just the escape"
        );
    }

    #[test]
    fn a_bundle_over_the_size_limit_is_an_error_naming_the_limit() {
        let dir = plugin_dir();
        write(
            dir.path(),
            "skills/form-filling/references/huge.md",
            &"x".repeat(MAX_BUNDLE_BYTES + 1),
        );
        let err = load_dir(dir.path()).unwrap_err().to_string();
        assert!(err.contains("limit"), "{err}");
    }

    #[test]
    fn a_folded_frontmatter_value_is_named_as_the_reason() {
        let dir = plugin_dir();
        write(
            dir.path(),
            "skills/folded/SKILL.md",
            "---\nname: folded\ndescription: >-\n  long text\n---\nbody",
        );
        let loaded = load_dir(dir.path()).unwrap();
        assert!(
            loaded
                .notices
                .iter()
                .any(|n| n.contains("not a plain value")),
            "{:?}",
            loaded.notices
        );
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
