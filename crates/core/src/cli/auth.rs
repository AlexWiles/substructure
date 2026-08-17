use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::transport::auth::{
    ApiKeyBinding, AuthResolver, BearerHashedApiKeyAuthResolver, JwtHs256ClientTokenAuthResolver,
    NoopAuthResolver, SOURCE_API_KEY,
};

use super::env::AuthEnvVars;
use super::DEFAULT_TENANT;

pub struct AuthWiring {
    pub client: Arc<dyn AuthResolver>,
    pub worker: Arc<dyn AuthResolver>,
    pub admin: Arc<dyn AuthResolver>,
    pub issuer: Arc<JwtHs256ClientTokenAuthResolver>,
}

impl AuthWiring {
    /// No client or worker authentication, for a server nothing off this
    /// machine can reach.
    pub fn unauthenticated() -> Self {
        let secret = hex::encode(rand::random::<[u8; 32]>());
        let issuer = Arc::new(JwtHs256ClientTokenAuthResolver::new(
            "substructure-dev",
            "substructure-dev",
            &secret,
        ));
        // A dev server has no credential to read, so it presents a key: the
        // machine-only operations need a key. Client endpoints
        // still validate the JWT `issuer` mints.
        Self {
            client: issuer.clone(),
            worker: Arc::new(
                NoopAuthResolver::new(DEFAULT_TENANT).with_subject(SOURCE_API_KEY, "dev-worker"),
            ),
            admin: Arc::new(
                NoopAuthResolver::new(DEFAULT_TENANT).with_subject(SOURCE_API_KEY, "dev-admin"),
            ),
            issuer,
        }
    }

    pub fn from_env(env: AuthEnvVars) -> anyhow::Result<Self> {
        let issuer = Arc::new(JwtHs256ClientTokenAuthResolver::new(
            env.client_token_issuer,
            env.client_token_audience,
            env.client_token_hs256_secret,
        ));
        let resolver = bearer_resolver(&env.substructure_api_key, "sdk")?;
        Ok(Self {
            client: issuer.clone(),
            worker: resolver.clone(),
            admin: resolver,
            issuer,
        })
    }
}

fn bearer_resolver(api_key: &str, role: &'static str) -> anyhow::Result<Arc<dyn AuthResolver>> {
    let hash = hex::encode(Sha256::digest(api_key.as_bytes()));
    let bindings = vec![ApiKeyBinding::new(DEFAULT_TENANT, hash, role)];
    Ok(Arc::new(
        BearerHashedApiKeyAuthResolver::new(bindings).map_err(anyhow::Error::msg)?,
    ))
}
