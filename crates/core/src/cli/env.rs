use clap::ValueEnum;

#[derive(Copy, Clone, ValueEnum)]
pub enum LlmProviderArg {
    Openrouter,
}

pub enum ProviderEnv {
    Openrouter { api_key: String },
}

pub struct AuthEnvVars {
    pub client_token_issuer: String,
    pub client_token_audience: String,
    pub client_token_hs256_secret: String,
    pub substructure_api_key: String,
}

pub struct EnvVars {
    pub provider: Option<ProviderEnv>,
    pub auth: Option<AuthEnvVars>,
}

impl EnvVars {
    pub fn load(provider: Option<LlmProviderArg>, dev: bool) -> Result<Self, ()> {
        let provider_specs: &[(&str, &str)] = match provider {
            Some(LlmProviderArg::Openrouter) => &[(
                "OPENROUTER_API_KEY",
                "API key for OpenRouter (https://openrouter.ai/keys)",
            )],
            None => &[],
        };

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

        let mut missing: Vec<(&'static str, &'static str)> = Vec::new();
        let mut provider_values: Vec<String> = Vec::with_capacity(provider_specs.len());
        let mut auth_values: Vec<String> = Vec::with_capacity(auth_specs.len());

        for (name, desc) in provider_specs {
            match std::env::var(name) {
                Ok(v) => provider_values.push(v),
                Err(_) => missing.push((name, desc)),
            }
        }
        if !dev {
            for (name, desc) in auth_specs {
                match std::env::var(name) {
                    Ok(v) => auth_values.push(v),
                    Err(_) => missing.push((name, desc)),
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
            return Err(());
        }

        let provider = match provider {
            Some(LlmProviderArg::Openrouter) => Some(ProviderEnv::Openrouter {
                api_key: provider_values.into_iter().next().unwrap(),
            }),
            None => None,
        };

        let auth = if dev {
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

        Ok(Self { provider, auth })
    }
}
