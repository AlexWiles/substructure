use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// What an `[llm.<id>]` block's `type` says: which vendor the engine calls with
/// a key of its own, or that the agent's own worker makes the call.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Openrouter,
    Anthropic,
    Openai,
    /// The agent's worker executes calls on this block itself, answering
    /// `llm.execute`. Never needs a credential where the engine runs.
    Worker,
}

impl ProviderKind {
    /// The variable a key is read from when the file names none. Named here
    /// rather than at each use so every command tells you the same name the
    /// engine will look for.
    pub fn default_api_key_env(self) -> Option<&'static str> {
        match self {
            Self::Openrouter => Some("OPENROUTER_API_KEY"),
            Self::Anthropic => Some("ANTHROPIC_API_KEY"),
            Self::Openai => Some("OPENAI_API_KEY"),
            Self::Worker => None,
        }
    }

    /// The variable holding the base URL override, for a vendor that has one.
    pub fn base_url_env(self) -> Option<&'static str> {
        match self {
            Self::Openrouter => Some("OPENROUTER_BASE_URL"),
            Self::Anthropic => Some("ANTHROPIC_BASE_URL"),
            Self::Openai => Some("OPENAI_BASE_URL"),
            Self::Worker => None,
        }
    }

    /// Where the key is issued.
    pub fn console_url(self) -> Option<&'static str> {
        match self {
            Self::Openrouter => Some("https://openrouter.ai/keys"),
            Self::Anthropic => Some("https://console.anthropic.com/settings/keys"),
            Self::Openai => Some("https://platform.openai.com/api-keys"),
            Self::Worker => None,
        }
    }

    /// The spelling the file uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Openrouter => "openrouter",
            Self::Anthropic => "anthropic",
            Self::Openai => "openai",
            Self::Worker => "worker",
        }
    }
}

/// How `subs run` renders a turn. The file spells these the way `--output`
/// does, so one type is both the flag's values and the manifest's.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OutputFormat {
    /// Stream AG-UI protocol events, one JSON object per line.
    AgUi,
    /// Stream raw persisted engine events, one JSON object per line.
    Jsonl,
    /// Human-readable text: streamed replies, tool calls, and results.
    Pretty,
}

/// One engine-run llm block, bound: the block's name, what it calls, and the
/// key the environment held for it.
pub struct ProviderEnv {
    pub name: String,
    pub kind: ProviderKind,
    pub api_key: String,
    pub base_url: Option<String>,
    pub cache_ttl: Option<String>,
}

/// One engine-run llm block as the file declares it, before binding.
pub struct ProviderBinding {
    pub name: String,
    pub kind: ProviderKind,
    pub api_key_env: String,
    pub base_url: Option<String>,
    pub cache_ttl: Option<String>,
}

pub struct AuthEnvVars {
    pub client_token_issuer: String,
    pub client_token_audience: String,
    pub client_token_hs256_secret: String,
    pub substructure_api_key: String,
}

pub struct EnvVars {
    pub providers: Vec<ProviderEnv>,
    pub auth: Option<AuthEnvVars>,
}

impl EnvVars {
    /// Returns `None` after printing the missing variables to stderr — every
    /// one of them, so a file declaring three blocks reports three names
    /// rather than one per run. `authenticate` false skips the token variables
    /// an unauthenticated server has no use for.
    pub fn load(bindings: Vec<ProviderBinding>, authenticate: bool) -> Option<Self> {
        let provider_specs: Vec<(String, String)> = bindings
            .iter()
            .map(|b| {
                let console = b
                    .kind
                    .console_url()
                    .map(|u| format!(" ({u})"))
                    .unwrap_or_default();
                (
                    b.api_key_env.clone(),
                    format!(
                        "API key for [llm.{}], a {} block{console}",
                        b.name,
                        b.kind.as_str()
                    ),
                )
            })
            .collect();

        let auth_specs: &[(&str, &str)] = &[
            ("CLIENT_TOKEN_ISSUER", "JWT 'iss' claim for client tokens"),
            ("CLIENT_TOKEN_AUDIENCE", "JWT 'aud' claim for client tokens"),
            (
                "CLIENT_TOKEN_HS256_SECRET",
                "HS256 secret used to sign client tokens",
            ),
            (
                "SUBSTRUCTURE_API_KEY",
                "Bearer API key SDK clients present to reach worker and admin HTTP APIs",
            ),
        ];

        let mut missing: Vec<(String, String)> = Vec::new();
        let mut provider_values: Vec<String> = Vec::with_capacity(provider_specs.len());
        let mut auth_values: Vec<String> = Vec::with_capacity(auth_specs.len());

        for (name, desc) in &provider_specs {
            match std::env::var(name) {
                Ok(v) => provider_values.push(v),
                Err(_) => missing.push((name.clone(), desc.clone())),
            }
        }
        if authenticate {
            for (name, desc) in auth_specs {
                match std::env::var(name) {
                    Ok(v) => auth_values.push(v),
                    Err(_) => missing.push((name.to_string(), desc.to_string())),
                }
            }
        }

        if !missing.is_empty() {
            eprintln!("error: missing required environment variable(s):");
            for (name, desc) in &missing {
                eprintln!("  - {name}: {desc}");
            }
            eprintln!("\nSet them and try again, e.g.:");
            for (name, _) in &missing {
                eprintln!("  export {name}=...");
            }
            return None;
        }

        let providers = bindings
            .into_iter()
            .zip(provider_values)
            .map(|(b, api_key)| ProviderEnv {
                name: b.name,
                kind: b.kind,
                api_key,
                base_url: b.base_url,
                cache_ttl: b.cache_ttl,
            })
            .collect();

        let auth = if !authenticate {
            None
        } else {
            let mut it = auth_values.into_iter();
            Some(AuthEnvVars {
                client_token_issuer: it.next().unwrap(),
                client_token_audience: it.next().unwrap(),
                client_token_hs256_secret: it.next().unwrap(),
                substructure_api_key: it.next().unwrap(),
            })
        };

        Some(Self { providers, auth })
    }
}
